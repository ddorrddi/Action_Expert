#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Same-checkpoint Euler-stage intervention for Reasoning RAW-KV Flow/DiT Action Expert.

RESEARCH QUESTION
-----------------
Is generated reasoning KV more useful during early/coarse trajectory formation
than during late geometric refinement?

STRICT CONTROL
--------------
ALL four conditions use the EXACT SAME reasoning-trained checkpoint, model instance,
normalizer, full RAW K/V tensors, physical prefix length, Euler solver, and the exact
same per-sample Gaussian x0. The only intervention is whether generated-reasoning
KV positions are readable in memory_mask at each Euler step.

Default 10-step protocol (split_step=5):

  ALL_ON     : reasoning readable at steps 0..9
  ALL_MASKED : reasoning masked   at steps 0..9
  EARLY_ON   : reasoning readable at steps 0..4, masked at 5..9
  LATE_ON    : reasoning masked   at steps 0..4, readable at 5..9

Primary paired comparisons:

  ALL_ON   vs ALL_MASKED
  EARLY_ON vs ALL_MASKED
  LATE_ON  vs ALL_MASKED
  EARLY_ON vs LATE_ON

Positive EARLY_ON-vs-LATE_ON ADE gain means EARLY_ON has lower ADE.
Every comparison includes a paired-bootstrap 95% confidence interval.

DEFAULT INPUTS
--------------
Dataset:
  /home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl

Trace cache:
  /home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit

Single checkpoint for ALL conditions:
  /home/lhh/lab/models/action_expert/
  reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt

Output:
  /home/lhh/lab/Action_Expert/results/reasoning_kv_euler_stage_same_checkpoint

OUTPUTS
-------
report.md
summary.json
attention_diagnostics.json
samples.jsonl
samples.csv
counterexamples.json
plots/*.png

No attention_vectors.pt is generated. Cross-scene collapse is accumulated online.

EXAMPLES
--------
cd ~/lab/Action_Expert/scripts
python3 test_reasoning_kv_euler_stage_same_checkpoint.py --overwrite

Debug first 10 samples:
python3 test_reasoning_kv_euler_stage_same_checkpoint.py --limit 10 --overwrite

Custom split (e.g. first 3 steps early, last 7 late):
python3 test_reasoning_kv_euler_stage_same_checkpoint.py --split-step 3 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DEFAULTS
# =============================================================================

TEST_JSONL = Path("/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl")
TRACE_CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit"
)
REASONING_CKPT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt"
)
RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/results/reasoning_kv_mask_same_checkpoint"
)

NUM_STEPS = 10
DT = 0.5
SEED = 20260823
FLOW_NOISE_SEED = 20260841
CACHE_DTYPE = torch.bfloat16


# =============================================================================
# BASIC HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def fmt(x: Any, digits: int = 4) -> str:
    v = finite_float(x)
    return "N/A" if not math.isfinite(v) else f"{v:.{digits}f}"


def pct(x: Any) -> str:
    v = finite_float(x)
    return "N/A" if not math.isfinite(v) else f"{100.0 * v:.2f}%"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSONL parse error {path}:{line_no}") from exc
    return out


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((value + int(base_seed)) % (2**63 - 1))


def make_noise(sample_id: str, base_seed: int, num_steps: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_sample_seed(sample_id, base_seed))
    return torch.randn((1, num_steps, 3), generator=g, dtype=torch.float32)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Empty sample id")
    row["id"] = sid

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (NUM_STEPS, 3) or not np.isfinite(gt).all():
        raise ValueError(
            f"trajectory must be finite [{NUM_STEPS},3] id={sid}, got={gt.shape}"
        )
    row["trajectory"] = gt.tolist()

    command = str(row.get("command") or row.get("mission_command") or "").strip()
    row["command"] = command
    row["mission_command"] = command

    for key in ("speed_mps", "acceleration_mps2", "heading_rad"):
        v = finite_float(row.get(key))
        row[key] = v

    images = row.get("images")
    row["images"] = [str(x) for x in images] if isinstance(images, list) else []
    return row


def row_fingerprint(row: Dict[str, Any]) -> Optional[str]:
    # Match the trace-cache generator whenever all original fields are present.
    if (
        not row.get("command")
        or len(row.get("images", [])) != 3
        or not all(math.isfinite(finite_float(row.get(k))) for k in (
            "speed_mps", "acceleration_mps2", "heading_rad"
        ))
    ):
        return None
    payload = {
        "id": row["id"],
        "images": row["images"],
        "command": row["command"],
        "speed_mps": float(row["speed_mps"]),
        "acceleration_mps2": float(row["acceleration_mps2"]),
        "heading_rad": float(row["heading_rad"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# =============================================================================
# TRACE CACHE
# =============================================================================

def validate_trace_cache(
    root: Path,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    manifest_path = root / "test" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Trace manifest not found: {manifest_path}")

    manifest = read_jsonl(manifest_path)
    if len(manifest) < len(rows):
        raise RuntimeError(
            f"Trace manifest too short: {len(manifest)} < requested {len(rows)}"
        )
    manifest = manifest[: len(rows)]

    expected_ids = [r["id"] for r in rows]
    got_ids = [str(m.get("id", "")) for m in manifest]
    if got_ids != expected_ids:
        for i, (a, b) in enumerate(zip(expected_ids, got_ids)):
            if a != b:
                raise RuntimeError(
                    f"Trace manifest/sample ID mismatch at index={i}: {a} != {b}"
                )
        raise RuntimeError("Trace manifest/sample ID mismatch")

    for row, item in zip(rows, manifest):
        cache_path = Path(str(item.get("cache_file", ""))).expanduser()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing trace cache for id={row['id']}: {cache_path}")

        fp = row_fingerprint(row)
        manifest_fp = str(item.get("input_fingerprint", ""))
        if fp is not None and manifest_fp and fp != manifest_fp:
            raise RuntimeError(f"Trace fingerprint mismatch id={row['id']}")

    return manifest


def load_full_reasoning_memory(
    cache: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    required = (
        "reasoning_direct_key",
        "reasoning_direct_value",
        "reasoning_delta_key",
        "reasoning_delta_value",
    )
    for key in required:
        if key not in cache:
            raise KeyError(f"Trace cache missing {key}")

    direct_k = cache["reasoning_direct_key"]
    direct_v = cache["reasoning_direct_value"]
    delta_k = cache["reasoning_delta_key"]
    delta_v = cache["reasoning_delta_value"]

    if direct_k.ndim != 3 or direct_v.shape != direct_k.shape:
        raise ValueError(f"Bad direct K/V shape: {direct_k.shape}/{direct_v.shape}")
    if delta_k.ndim != 3 or delta_v.shape != delta_k.shape:
        raise ValueError(f"Bad reasoning delta K/V shape: {delta_k.shape}/{delta_v.shape}")
    if direct_k.shape[0] != delta_k.shape[0] or direct_k.shape[2] != delta_k.shape[2]:
        raise ValueError(
            f"Direct/delta KV geometry mismatch: {direct_k.shape} vs {delta_k.shape}"
        )

    direct_tokens = int(direct_k.shape[1])
    reasoning_tokens = int(delta_k.shape[1])
    if reasoning_tokens <= 0:
        raise RuntimeError("Reasoning delta contains zero tokens")

    full_k = torch.cat([direct_k, delta_k], dim=1).unsqueeze(0)
    full_v = torch.cat([direct_v, delta_v], dim=1).unsqueeze(0)
    return full_k, full_v, direct_tokens, reasoning_tokens


# =============================================================================
# TRAJECTORY METRICS
# =============================================================================

def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def xyz_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_xyz, dtype=np.float64)
    gt = np.asarray(gt_xyz, dtype=np.float64)
    if pred.shape != (NUM_STEPS, 3) or gt.shape != (NUM_STEPS, 3):
        raise ValueError(f"Bad xyz metric shapes {pred.shape}, {gt.shape}")
    err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
    heading = np.abs(wrap_angle_np(pred[:, 2] - gt[:, 2]))
    return {
        "ade_m": float(err.mean()),
        "fde_m": float(err[-1]),
        "heading_mae_rad": float(heading.mean()),
    }


# =============================================================================
# ONLINE CROSS-SCENE COLLAPSE ACCUMULATOR
# =============================================================================

class VectorCollapseAccumulator:
    """Compute cross-scene cosine collapse without storing raw sample vectors."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_unit: Optional[np.ndarray] = None
        self.sum_raw: Optional[np.ndarray] = None
        self.sum_sq: Optional[np.ndarray] = None
        self.sum_norm = 0.0

    def update(self, vector: torch.Tensor | np.ndarray) -> None:
        if torch.is_tensor(vector):
            x = vector.detach().float().cpu().numpy().astype(np.float64, copy=False)
        else:
            x = np.asarray(vector, dtype=np.float64)
        x = x.reshape(-1)
        if not np.isfinite(x).all():
            return
        norm = float(np.linalg.norm(x))
        if norm <= 1.0e-12:
            return
        unit = x / norm
        if self.sum_unit is None:
            self.sum_unit = np.zeros_like(unit)
            self.sum_raw = np.zeros_like(x)
            self.sum_sq = np.zeros_like(x)
        self.n += 1
        self.sum_unit += unit
        self.sum_raw += x
        self.sum_sq += x * x
        self.sum_norm += norm

    def summary(self) -> Dict[str, Any]:
        if self.n <= 0 or self.sum_unit is None or self.sum_raw is None or self.sum_sq is None:
            return {
                "n_samples": 0,
                "cross_scene_mean_cosine": float("nan"),
                "centroid_norm_ratio": float("nan"),
                "mean_vector_norm": float("nan"),
                "mean_feature_std": float("nan"),
            }

        if self.n >= 2:
            sum_unit_norm_sq = float(np.dot(self.sum_unit, self.sum_unit))
            mean_pair_cos = (sum_unit_norm_sq - self.n) / (self.n * (self.n - 1))
        else:
            mean_pair_cos = float("nan")

        mean_vec = self.sum_raw / float(self.n)
        mean_norm = self.sum_norm / float(self.n)
        centroid_ratio = (
            float(np.linalg.norm(mean_vec)) / mean_norm if mean_norm > 1.0e-12 else float("nan")
        )
        var = np.maximum(self.sum_sq / float(self.n) - mean_vec * mean_vec, 0.0)
        return {
            "n_samples": int(self.n),
            "cross_scene_mean_cosine": float(mean_pair_cos),
            "centroid_norm_ratio": float(centroid_ratio),
            "mean_vector_norm": float(mean_norm),
            "mean_feature_std": float(np.sqrt(var).mean()),
        }


# =============================================================================
# ATTENTION DIAGNOSTICS
# =============================================================================

ATTENTION_METRICS = (
    "vlm_attention_mass",
    "self_attention_mass",
    "vlm_attention_entropy_nats",
    "vlm_effective_token_fraction",
    "reasoning_attention_mass",
    "reasoning_attention_share_within_vlm",
    "reasoning_attention_enrichment",
    "reasoning_token_fraction_physical",
    "reasoning_attention_entropy_nats",
    "reasoning_attention_entropy_normalized",
    "reasoning_effective_token_fraction",
    "top1_reasoning_token_share",
    "top1_reasoning_token_share_within_vlm",
    "attended_context_uniform_mean_v_cosine",
)


class AttentionDiagnosticRecorder:
    """Recorder-only attention diagnostic; does not alter model output."""

    def __init__(self, role: str, num_layers: int, solver_steps: int) -> None:
        self.role = str(role)
        self.num_layers = int(num_layers)
        self.solver_steps = int(solver_steps)

        self.sample_ids: List[str] = []
        self.records: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
        self.sample_token_regions: Dict[str, Dict[str, int]] = {}

        self._sample_id: Optional[str] = None
        self._step: Optional[int] = None
        self._t: Optional[float] = None
        self._direct_tokens = 0
        self._reasoning_tokens = 0

        # Online collapse accumulators: no attention_vectors.pt required.
        self.collapse_overall = VectorCollapseAccumulator()
        self.collapse_by_layer = {
            i: VectorCollapseAccumulator() for i in range(self.num_layers)
        }
        self.collapse_by_step = {
            i: VectorCollapseAccumulator() for i in range(self.solver_steps)
        }
        self.collapse_by_layer_step = {
            (l, s): VectorCollapseAccumulator()
            for l in range(self.num_layers)
            for s in range(self.solver_steps)
        }
        self.reasoning_collapse_overall = VectorCollapseAccumulator()
        self.reasoning_collapse_by_layer = {
            i: VectorCollapseAccumulator() for i in range(self.num_layers)
        }
        self.reasoning_collapse_by_step = {
            i: VectorCollapseAccumulator() for i in range(self.solver_steps)
        }
        self.reasoning_collapse_by_layer_step = {
            (l, s): VectorCollapseAccumulator()
            for l in range(self.num_layers)
            for s in range(self.solver_steps)
        }

        # Per-active-sample vector sums, so overall/layer/step collapse is based
        # on one vector per scene rather than treating every Euler cell as a scene.
        self._sample_attended_sum: Optional[np.ndarray] = None
        self._sample_attended_count = 0
        self._sample_layer_sum: Dict[int, np.ndarray] = {}
        self._sample_layer_count: Dict[int, int] = defaultdict(int)
        self._sample_step_sum: Dict[int, np.ndarray] = {}
        self._sample_step_count: Dict[int, int] = defaultdict(int)

        self._sample_reasoning_sum: Optional[np.ndarray] = None
        self._sample_reasoning_count = 0
        self._sample_reasoning_layer_sum: Dict[int, np.ndarray] = {}
        self._sample_reasoning_layer_count: Dict[int, int] = defaultdict(int)
        self._sample_reasoning_step_sum: Dict[int, np.ndarray] = {}
        self._sample_reasoning_step_count: Dict[int, int] = defaultdict(int)

    @staticmethod
    def _safe_mean(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return float("nan")
        v = float(x.detach().float().mean().cpu().item())
        return v if math.isfinite(v) else float("nan")

    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * torch.log(p.clamp_min(1.0e-12))).sum(dim=-1)

    def begin_sample(self, sample_id: str, direct_tokens: int, reasoning_tokens: int) -> None:
        if self._sample_id is not None:
            raise RuntimeError(f"Recorder already active: {self._sample_id}")
        sid = str(sample_id)
        if sid in self.records:
            raise RuntimeError(f"Duplicate recorder sample: {sid}")
        self._sample_id = sid
        self._step = None
        self._t = None
        self._direct_tokens = int(direct_tokens)
        self._reasoning_tokens = int(reasoning_tokens)
        self.sample_ids.append(sid)
        self.records[sid] = {}
        self.sample_token_regions[sid] = {
            "direct_tokens": int(direct_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "physical_prefix_tokens": int(direct_tokens + reasoning_tokens),
        }

        self._sample_attended_sum = None
        self._sample_attended_count = 0
        self._sample_layer_sum = {}
        self._sample_layer_count = defaultdict(int)
        self._sample_step_sum = {}
        self._sample_step_count = defaultdict(int)
        self._sample_reasoning_sum = None
        self._sample_reasoning_count = 0
        self._sample_reasoning_layer_sum = {}
        self._sample_reasoning_layer_count = defaultdict(int)
        self._sample_reasoning_step_sum = {}
        self._sample_reasoning_step_count = defaultdict(int)

    def set_euler_step(self, step: int, t: float) -> None:
        if self._sample_id is None:
            raise RuntimeError("set_euler_step without active sample")
        self._step = int(step)
        self._t = float(t)

    @staticmethod
    def _accumulate_np(
        current: Optional[np.ndarray],
        vector: np.ndarray,
    ) -> np.ndarray:
        if current is None:
            return vector.astype(np.float64, copy=True)
        current += vector
        return current

    def record(
        self,
        layer_index: int,
        attn: torch.Tensor,
        expanded_vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
        prefix_len: int,
    ) -> None:
        if self._sample_id is None or self._step is None:
            raise RuntimeError("record outside active sample/step")
        if attn.ndim != 4 or expanded_vlm_value.ndim != 4:
            raise ValueError(f"Bad attn/V ranks: {attn.shape}/{expanded_vlm_value.shape}")
        if int(attn.shape[0]) != 1:
            raise ValueError("Only batch size 1 is supported")

        layer = int(layer_index)
        step = int(self._step)
        cell = (layer, step)
        if cell in self.records[self._sample_id]:
            raise RuntimeError(f"Duplicate recorder cell {self._sample_id} {cell}")

        prefix_len = int(prefix_len)
        a = attn[..., :prefix_len].detach().float()
        v = expanded_vlm_value.detach().float()
        valid = memory_mask[:, None, None, :prefix_len].bool()
        valid_f = valid.float()
        masked_a = a * valid_f

        vlm_mass = masked_a.sum(dim=-1)
        self_mass = (1.0 - vlm_mass).clamp(0.0, 1.0)
        p_vlm = masked_a / vlm_mass.unsqueeze(-1).clamp_min(1.0e-12)
        vlm_entropy = self._entropy(p_vlm)
        valid_tokens = memory_mask[:, :prefix_len].bool().sum(dim=-1).float().clamp_min(1.0)
        vlm_eff = (torch.exp(vlm_entropy) / valid_tokens[:, None, None]).clamp(0.0, 1.0)

        direct_end = min(max(self._direct_tokens, 0), prefix_len)
        r_start = direct_end
        r_end = min(r_start + max(self._reasoning_tokens, 0), prefix_len)
        r_count = max(0, r_end - r_start)
        physical_fraction = float(r_count) / float(prefix_len) if prefix_len > 0 else float("nan")

        metrics = {
            "euler_t": float(self._t),
            "vlm_attention_mass": self._safe_mean(vlm_mass),
            "self_attention_mass": self._safe_mean(self_mass),
            "vlm_attention_entropy_nats": self._safe_mean(vlm_entropy),
            "vlm_effective_token_fraction": self._safe_mean(vlm_eff),
            "reasoning_attention_mass": float("nan"),
            "reasoning_attention_share_within_vlm": float("nan"),
            "reasoning_attention_enrichment": float("nan"),
            "reasoning_token_fraction_physical": physical_fraction,
            "reasoning_attention_entropy_nats": float("nan"),
            "reasoning_attention_entropy_normalized": float("nan"),
            "reasoning_effective_token_fraction": float("nan"),
            "top1_reasoning_token_share": float("nan"),
            "top1_reasoning_token_share_within_vlm": float("nan"),
            "attended_context_uniform_mean_v_cosine": float("nan"),
        }

        # Attention-weighted VLM context over valid prefix tokens.
        attended = torch.matmul(p_vlm, v)  # [1,Hq,Nq,D]
        attended_vec = attended.mean(dim=2).reshape(-1)
        token_valid_f = memory_mask[:, None, :prefix_len, None].float()
        uniform_v = (v * token_valid_f).sum(dim=2) / token_valid_f.sum(dim=2).clamp_min(1.0)
        uniform_vec = uniform_v.reshape(-1)
        metrics["attended_context_uniform_mean_v_cosine"] = self._safe_mean(
            F.cosine_similarity(
                attended_vec.unsqueeze(0), uniform_vec.unsqueeze(0), dim=-1, eps=1.0e-8
            )
        )

        attended_np = attended_vec.detach().float().cpu().numpy().astype(np.float64, copy=False)
        self.collapse_by_layer_step[cell].update(attended_np)
        self._sample_attended_sum = self._accumulate_np(self._sample_attended_sum, attended_np)
        self._sample_attended_count += 1
        self._sample_layer_sum[layer] = self._accumulate_np(
            self._sample_layer_sum.get(layer), attended_np
        )
        self._sample_layer_count[layer] += 1
        self._sample_step_sum[step] = self._accumulate_np(
            self._sample_step_sum.get(step), attended_np
        )
        self._sample_step_count[step] += 1

        if r_count > 0:
            r = masked_a[..., r_start:r_end]
            r_mass = r.sum(dim=-1)
            r_share = r_mass / vlm_mass.clamp_min(1.0e-12)
            if physical_fraction > 0.0:
                enrichment = r_share / physical_fraction
            else:
                enrichment = torch.full_like(r_share, float("nan"))

            metrics["reasoning_attention_mass"] = self._safe_mean(r_mass)
            metrics["reasoning_attention_share_within_vlm"] = self._safe_mean(r_share)
            metrics["reasoning_attention_enrichment"] = self._safe_mean(enrichment)

            # Distribution metrics are meaningful only when reasoning receives mass.
            positive = r_mass > 1.0e-12
            if bool(positive.any().item()):
                p_r = r / r_mass.unsqueeze(-1).clamp_min(1.0e-12)
                r_entropy = self._entropy(p_r)
                if r_count > 1:
                    r_entropy_norm = r_entropy / math.log(float(r_count))
                else:
                    r_entropy_norm = torch.zeros_like(r_entropy)
                r_eff = (torch.exp(r_entropy) / float(max(r_count, 1))).clamp(0.0, 1.0)
                top1_abs = r.max(dim=-1).values
                top1_share = top1_abs / r_mass.clamp_min(1.0e-12)
                top1_vlm = top1_abs / vlm_mass.clamp_min(1.0e-12)

                # Exclude zero-mass query/head entries from entropy-type averages.
                def masked_mean(x: torch.Tensor) -> float:
                    vals = x[positive]
                    return self._safe_mean(vals) if vals.numel() else float("nan")

                metrics.update({
                    "reasoning_attention_entropy_nats": masked_mean(r_entropy),
                    "reasoning_attention_entropy_normalized": masked_mean(r_entropy_norm),
                    "reasoning_effective_token_fraction": masked_mean(r_eff),
                    "top1_reasoning_token_share": masked_mean(top1_share),
                    "top1_reasoning_token_share_within_vlm": masked_mean(top1_vlm),
                })

                # Reasoning-only context. For MASKED this branch naturally does not run.
                r_v = v[:, :, r_start:r_end, :]
                r_context = torch.matmul(p_r, r_v).mean(dim=2).reshape(-1)
                r_np = r_context.detach().float().cpu().numpy().astype(np.float64, copy=False)
                self.reasoning_collapse_by_layer_step[cell].update(r_np)
                self._sample_reasoning_sum = self._accumulate_np(
                    self._sample_reasoning_sum, r_np
                )
                self._sample_reasoning_count += 1
                self._sample_reasoning_layer_sum[layer] = self._accumulate_np(
                    self._sample_reasoning_layer_sum.get(layer), r_np
                )
                self._sample_reasoning_layer_count[layer] += 1
                self._sample_reasoning_step_sum[step] = self._accumulate_np(
                    self._sample_reasoning_step_sum.get(step), r_np
                )
                self._sample_reasoning_step_count[step] += 1

        self.records[self._sample_id][cell] = metrics

    @staticmethod
    def _aggregate(cells: Iterable[Dict[str, float]]) -> Dict[str, float]:
        cells = list(cells)
        out: Dict[str, float] = {}
        for key in ATTENTION_METRICS:
            vals = np.asarray([finite_float(c.get(key)) for c in cells], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[key] = float(vals.mean()) if len(vals) else float("nan")

        # Correct aggregate definitions: ratio-of-sums rather than mean-of-ratios.
        r_mass = np.asarray(
            [finite_float(c.get("reasoning_attention_mass")) for c in cells], dtype=np.float64
        )
        v_mass = np.asarray(
            [finite_float(c.get("vlm_attention_mass")) for c in cells], dtype=np.float64
        )
        frac = np.asarray(
            [finite_float(c.get("reasoning_token_fraction_physical")) for c in cells],
            dtype=np.float64,
        )
        mask = np.isfinite(r_mass) & np.isfinite(v_mass) & np.isfinite(frac)
        if mask.any() and float(v_mass[mask].sum()) > 1.0e-12:
            rsum = float(r_mass[mask].sum())
            vsum = float(v_mass[mask].sum())
            null_mass = float((v_mass[mask] * frac[mask]).sum())
            out["reasoning_attention_share_within_vlm_mass_weighted"] = rsum / vsum
            out["reasoning_attention_enrichment_mass_weighted"] = (
                rsum / null_mass if null_mass > 1.0e-12 else float("nan")
            )
        else:
            out["reasoning_attention_share_within_vlm_mass_weighted"] = float("nan")
            out["reasoning_attention_enrichment_mass_weighted"] = float("nan")
        return out

    def end_sample(self) -> Dict[str, float]:
        if self._sample_id is None:
            raise RuntimeError("end_sample without active sample")
        sid = self._sample_id
        expected = self.num_layers * self.solver_steps
        if len(self.records[sid]) != expected:
            raise RuntimeError(
                f"Incomplete attention recording id={sid}: {len(self.records[sid])}/{expected}"
            )

        # Update cross-scene accumulators using one scene vector per aggregation unit.
        if self._sample_attended_sum is not None and self._sample_attended_count > 0:
            self.collapse_overall.update(
                self._sample_attended_sum / float(self._sample_attended_count)
            )
        for layer, vec in self._sample_layer_sum.items():
            self.collapse_by_layer[layer].update(
                vec / float(self._sample_layer_count[layer])
            )
        for step, vec in self._sample_step_sum.items():
            self.collapse_by_step[step].update(
                vec / float(self._sample_step_count[step])
            )

        if self._sample_reasoning_sum is not None and self._sample_reasoning_count > 0:
            self.reasoning_collapse_overall.update(
                self._sample_reasoning_sum / float(self._sample_reasoning_count)
            )
        for layer, vec in self._sample_reasoning_layer_sum.items():
            self.reasoning_collapse_by_layer[layer].update(
                vec / float(self._sample_reasoning_layer_count[layer])
            )
        for step, vec in self._sample_reasoning_step_sum.items():
            self.reasoning_collapse_by_step[step].update(
                vec / float(self._sample_reasoning_step_count[step])
            )

        summary = self.sample_summary(sid)
        self._sample_id = None
        self._step = None
        self._t = None
        return summary

    def sample_summary(self, sample_id: str) -> Dict[str, float]:
        return self._aggregate(self.records[str(sample_id)].values())

    def sample_layer_summary(self, sample_id: str, layer: int) -> Dict[str, float]:
        return self._aggregate(
            rec for (l, _), rec in self.records[str(sample_id)].items() if l == int(layer)
        )

    def sample_step_summary(self, sample_id: str, step: int) -> Dict[str, float]:
        return self._aggregate(
            rec for (_, s), rec in self.records[str(sample_id)].items() if s == int(step)
        )

    def overall_summary(self) -> Dict[str, float]:
        return self._aggregate(
            rec for sid in self.sample_ids for rec in self.records[sid].values()
        )

    def by_layer_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            str(layer): self._aggregate(
                rec
                for sid in self.sample_ids
                for (l, _), rec in self.records[sid].items()
                if l == layer
            )
            for layer in range(self.num_layers)
        }

    def by_step_summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for step in range(self.solver_steps):
            obj = self._aggregate(
                rec
                for sid in self.sample_ids
                for (_, s), rec in self.records[sid].items()
                if s == step
            )
            obj["euler_t"] = float(step / self.solver_steps)
            out[str(step)] = obj
        return out

    def by_layer_step_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for layer in range(self.num_layers):
            out[str(layer)] = {}
            for step in range(self.solver_steps):
                obj = self._aggregate(
                    self.records[sid][(layer, step)] for sid in self.sample_ids
                )
                obj["euler_t"] = float(step / self.solver_steps)
                out[str(layer)][str(step)] = obj
        return out

    def collapse_summary(self) -> Dict[str, Any]:
        return {
            "attended_context": {
                "overall": self.collapse_overall.summary(),
                "by_layer": {
                    str(k): v.summary() for k, v in self.collapse_by_layer.items()
                },
                "by_euler_step": {
                    str(k): v.summary() for k, v in self.collapse_by_step.items()
                },
                "by_layer_step": {
                    str(l): {
                        str(s): self.collapse_by_layer_step[(l, s)].summary()
                        for s in range(self.solver_steps)
                    }
                    for l in range(self.num_layers)
                },
            },
            "reasoning_only_context": {
                "overall": self.reasoning_collapse_overall.summary(),
                "by_layer": {
                    str(k): v.summary() for k, v in self.reasoning_collapse_by_layer.items()
                },
                "by_euler_step": {
                    str(k): v.summary() for k, v in self.reasoning_collapse_by_step.items()
                },
            },
        }


# =============================================================================
# EXACT RAW-KV FLOW/DiT ARCHITECTURE
# =============================================================================

@dataclass
class TrajectoryNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)
        if self.mean.shape != self.std.shape:
            raise ValueError("Normalizer mean/std mismatch")
        if torch.any(self.std <= 0):
            raise ValueError("Normalizer std must be > 0")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
        )

    def to(self, device: torch.device | str) -> "TrajectoryNormalizer":
        return TrajectoryNormalizer(self.mean.to(device), self.std.to(device))

    def _stats_for(self, trajectory: torch.Tensor):
        if self.mean.ndim == 1:
            if trajectory.ndim == 2:
                return self.mean.view(1, 3), self.std.view(1, 3)
            return self.mean.view(1, 1, 3), self.std.view(1, 1, 3)
        if trajectory.ndim == 2:
            return self.mean, self.std
        return self.mean.unsqueeze(0), self.std.unsqueeze(0)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        return trajectory * std.to(trajectory.device) + mean.to(trajectory.device)


class RawKVPrefixFusionAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vlm_num_attention_heads,
        vlm_num_kv_heads,
        vlm_head_dim,
        dropout,
        causal_queries,
    ):
        super().__init__()
        self.vlm_num_attention_heads = int(vlm_num_attention_heads)
        self.vlm_num_kv_heads = int(vlm_num_kv_heads)
        self.vlm_head_dim = int(vlm_head_dim)
        if self.vlm_num_attention_heads % self.vlm_num_kv_heads != 0:
            raise ValueError("VLM Hq must be divisible by Hkv")
        self.kv_groups = self.vlm_num_attention_heads // self.vlm_num_kv_heads
        self.q_attn_dim = self.vlm_num_attention_heads * self.vlm_head_dim
        self.kv_attn_dim = self.vlm_num_kv_heads * self.vlm_head_dim
        self.scale = self.vlm_head_dim ** -0.5
        self.causal_queries = bool(causal_queries)
        self.dropout_p = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, self.q_attn_dim, bias=False)
        self.self_k_proj = nn.Linear(hidden_dim, self.kv_attn_dim, bias=False)
        self.self_v_proj = nn.Linear(hidden_dim, self.kv_attn_dim, bias=False)
        self.out_proj = nn.Linear(self.q_attn_dim, hidden_dim, bias=False)

        # Plain Python attributes only; not part of state_dict.
        self.diagnostic_recorder: Optional[AttentionDiagnosticRecorder] = None
        self.diagnostic_layer_index = -1

    def forward(self, query_tokens, vlm_key, vlm_value, memory_mask):
        b, n, _ = query_tokens.shape
        hq = self.vlm_num_attention_heads
        hkv = self.vlm_num_kv_heads
        d = self.vlm_head_dim

        q = self.q_proj(query_tokens).view(b, n, hq, d).transpose(1, 2)
        self_k = self.self_k_proj(query_tokens).view(b, n, hkv, d).transpose(1, 2)
        self_v = self.self_v_proj(query_tokens).view(b, n, hkv, d).transpose(1, 2)
        raw_k = vlm_key.to(device=q.device, dtype=q.dtype)
        raw_v = vlm_value.to(device=q.device, dtype=q.dtype)

        k = torch.cat([raw_k, self_k], dim=2)
        v = torch.cat([raw_v, self_v], dim=2)
        expanded_raw_v = raw_v
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
            expanded_raw_v = raw_v.repeat_interleave(self.kv_groups, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        prefix_len = int(raw_k.shape[2])
        invalid_prefix = (~memory_mask.bool())[:, None, None, :].expand(
            b, 1, n, prefix_len
        )
        if self.causal_queries:
            blocked_self = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=scores.device), diagonal=1
            )[None, None, :, :].expand(b, 1, n, n)
        else:
            blocked_self = torch.zeros(
                (b, 1, n, n), dtype=torch.bool, device=scores.device
            )
        blocked = torch.cat([invalid_prefix, blocked_self], dim=-1)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

        # IMPORTANT: diagnostic observes the same pre-dropout softmax used by AE.
        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        if self.diagnostic_recorder is not None:
            self.diagnostic_recorder.record(
                layer_index=self.diagnostic_layer_index,
                attn=attn,
                expanded_vlm_value=expanded_raw_v,
                memory_mask=memory_mask,
                prefix_len=prefix_len,
            )

        attn_used = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn_used, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.q_attn_dim)
        return self.out_proj(out)


class TrajectoryOutputHeadRawKV(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return self.head(self.norm(x))


class RawKVEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vlm_num_attention_heads,
        vlm_num_kv_heads,
        vlm_head_dim,
        ff_dim,
        dropout,
        causal_queries,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = RawKVPrefixFusionAttention(
            hidden_dim,
            vlm_num_attention_heads,
            vlm_num_kv_heads,
            vlm_head_dim,
            dropout,
            causal_queries,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, vlm_key, vlm_value, memory_mask):
        x = self.norm_attn(tokens)
        tokens = tokens + self.dropout(self.attn(x, vlm_key, vlm_value, memory_mask))
        tokens = tokens + self.dropout(self.ffn(self.norm_ff(tokens)))
        return tokens


class FourierEncoderRawKV(nn.Module):
    def __init__(self, dim=20, max_freq=100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(
            0.0, math.log10(float(max_freq)), steps=half, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x):
        arg = x.float().unsqueeze(-1) * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class FlowActionInputProjectionRawKV(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim,
        time_fourier_dim=20,
        time_mlp_hidden=512,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.time_encoder = FourierEncoderRawKV(time_fourier_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_fourier_dim, time_mlp_hidden),
            nn.LayerNorm(time_mlp_hidden),
            nn.GELU(),
            nn.Linear(time_mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x_t, t):
        b = x_t.shape[0]
        action_hidden = self.action_proj(
            x_t.to(dtype=self.action_proj[0].weight.dtype)
        )
        t_scalar = t.reshape(b, -1)[:, 0]
        time_hidden = self.time_proj(
            self.time_encoder(t_scalar).to(dtype=self.time_proj[0].weight.dtype)
        ).unsqueeze(1)
        return action_hidden + time_hidden


class RawKVFlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_steps,
        num_layers,
        vlm_num_attention_heads,
        vlm_num_kv_heads,
        vlm_head_dim,
        ff_dim,
        dropout,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.action_in = FlowActionInputProjectionRawKV(3, hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )
        self.layers = nn.ModuleList([
            RawKVEncoderBlock(
                hidden_dim,
                vlm_num_attention_heads,
                vlm_num_kv_heads,
                vlm_head_dim,
                ff_dim,
                dropout,
                False,
            )
            for _ in range(num_layers)
        ])
        self.output = TrajectoryOutputHeadRawKV(hidden_dim)

        for i, layer in enumerate(self.layers):
            layer.attn.diagnostic_layer_index = int(i)

    def attach_attention_diagnostic_recorder(
        self,
        recorder: Optional[AttentionDiagnosticRecorder],
    ) -> None:
        for i, layer in enumerate(self.layers):
            layer.attn.diagnostic_layer_index = int(i)
            layer.attn.diagnostic_recorder = recorder

    def forward(self, x_t, t, vlm_key, vlm_value, memory_mask):
        tokens = self.action_in(x_t, t)
        tokens = tokens + self.horizon_embedding.to(
            device=tokens.device, dtype=tokens.dtype
        )
        for layer in self.layers:
            tokens = layer(tokens, vlm_key, vlm_value, memory_mask)
        return self.output(tokens)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model_state_dict",
        "action_config",
        "normalizer",
        "vlm_num_attention_heads",
        "vlm_num_kv_heads",
        "vlm_head_dim",
    ):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key}: {path}")
    return ckpt


def build_model_from_checkpoint(
    ckpt: Dict[str, Any],
    device: torch.device,
) -> Tuple[RawKVFlowMatchingActionExpert, TrajectoryNormalizer, int]:
    cfg = ckpt["action_config"]
    model = RawKVFlowMatchingActionExpert(
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        vlm_num_attention_heads=int(ckpt["vlm_num_attention_heads"]),
        vlm_num_kv_heads=int(ckpt["vlm_num_kv_heads"]),
        vlm_head_dim=int(ckpt["vlm_head_dim"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    normalizer = TrajectoryNormalizer.from_dict(ckpt["normalizer"]).to(device)
    solver_steps = int(cfg.get("solver_steps", 10))
    return model, normalizer, solver_steps


@torch.inference_mode()
def euler_sample_rawkv_controlled(
    model,
    vlm_key,
    vlm_value,
    memory_mask,
    normalizer,
    solver_steps,
    noise_cpu,
    diagnostic_recorder: AttentionDiagnosticRecorder,
    amp_dtype: torch.dtype,
):
    device = vlm_key.device
    x = noise_cpu.to(device=device, dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, solver_steps + 1, device=device)
    for i in range(solver_steps):
        t_now = float(times[i].item())
        dt = float((times[i + 1] - times[i]).item())
        diagnostic_recorder.set_euler_step(i, t_now)
        t = torch.full((1, 1, 1), t_now, device=device, dtype=torch.float32)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            velocity = model(
                x_t=x,
                t=t,
                vlm_key=vlm_key,
                vlm_value=vlm_value,
                memory_mask=memory_mask,
            )
        x = x + dt * velocity.float()
    return normalizer.denormalize(x)


# =============================================================================
# STATISTICS
# =============================================================================

def clean_pair(x: Iterable[Any], y: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray([finite_float(v) for v in x], dtype=np.float64)
    b = np.asarray([finite_float(v) for v in y], dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def pearson_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    a, b = clean_pair(x, y)
    if len(a) < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    a, b = clean_pair(x, y)
    if len(a) < 3:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def corr_pair(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, Any]:
    a, b = clean_pair(x, y)
    return {
        "n": int(len(a)),
        "pearson": pearson_corr(a, b),
        "spearman": spearman_corr(a, b),
    }


def bootstrap_mean_ci(values: Iterable[Any], iterations: int, seed: int) -> Dict[str, Any]:
    v = np.asarray([finite_float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "lo95": float("nan"),
            "hi95": float("nan"),
        }
    if iterations <= 0:
        return {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "lo95": float("nan"),
            "hi95": float("nan"),
        }
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        means[i] = v[rng.integers(0, len(v), size=len(v))].mean()
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "lo95": float(np.percentile(means, 2.5)),
        "hi95": float(np.percentile(means, 97.5)),
    }


# =============================================================================
# SAME-CHECKPOINT ON vs MASKED INFERENCE
# =============================================================================

def evaluate_same_checkpoint_on_vs_masked(
    checkpoint_path: Path,
    rows: Sequence[Dict[str, Any]],
    trace_manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    noise_seed: int,
    solver_steps_override: Optional[int],
    amp_dtype: torch.dtype,
) -> Tuple[
    List[Dict[str, Any]],
    AttentionDiagnosticRecorder,
    AttentionDiagnosticRecorder,
    Dict[str, Any],
]:
    ckpt = load_checkpoint(checkpoint_path)
    model, normalizer, ckpt_steps = build_model_from_checkpoint(ckpt, device)
    solver_steps = int(solver_steps_override) if solver_steps_override else ckpt_steps

    on_recorder = AttentionDiagnosticRecorder(
        role="reasoning_kv_on",
        num_layers=len(model.layers),
        solver_steps=solver_steps,
    )
    masked_recorder = AttentionDiagnosticRecorder(
        role="reasoning_kv_masked",
        num_layers=len(model.layers),
        solver_steps=solver_steps,
    )

    action_cfg = dict(ckpt["action_config"])
    geometry = {
        "vlm_num_attention_heads": int(ckpt["vlm_num_attention_heads"]),
        "vlm_num_kv_heads": int(ckpt["vlm_num_kv_heads"]),
        "vlm_head_dim": int(ckpt["vlm_head_dim"]),
        "num_layers": int(len(model.layers)),
        "solver_steps": int(solver_steps),
        "action_config": action_cfg,
    }

    samples: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for index, (row, item) in enumerate(zip(rows, trace_manifest), 1):
        sid = row["id"]
        if sid != str(item["id"]):
            raise RuntimeError(f"ID mismatch at index={index}: {sid} != {item['id']}")
        cache = torch.load(item["cache_file"], map_location="cpu", weights_only=False)
        if str(cache.get("id", "")) != sid:
            raise RuntimeError(f"Cache ID mismatch id={sid}")

        full_k_cpu, full_v_cpu, direct_tokens, reasoning_tokens = load_full_reasoning_memory(cache)
        full_k = full_k_cpu.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        full_v = full_v_cpu.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        total_tokens = int(full_k.shape[2])
        if total_tokens != direct_tokens + reasoning_tokens:
            raise RuntimeError(f"Token length mismatch id={sid}")

        # Physical K/V tensor is IDENTICAL for both conditions.
        on_mask = torch.ones((1, total_tokens), dtype=torch.bool, device=device)
        masked_mask = on_mask.clone()
        masked_mask[:, direct_tokens:direct_tokens + reasoning_tokens] = False

        # Identical x0 for both conditions.
        noise = make_noise(sid, noise_seed, NUM_STEPS)

        # -----------------------------
        # CONDITION A: reasoning KV ON
        # -----------------------------
        on_recorder.begin_sample(sid, direct_tokens, reasoning_tokens)
        model.attach_attention_diagnostic_recorder(on_recorder)
        sync_cuda(device)
        t0 = time.perf_counter()
        pred_on = euler_sample_rawkv_controlled(
            model,
            full_k,
            full_v,
            on_mask,
            normalizer,
            solver_steps,
            noise,
            on_recorder,
            amp_dtype,
        )
        sync_cuda(device)
        on_ms = (time.perf_counter() - t0) * 1000.0
        on_attn = on_recorder.end_sample()

        # ---------------------------------
        # CONDITION B: reasoning KV MASKED
        # ---------------------------------
        masked_recorder.begin_sample(sid, direct_tokens, reasoning_tokens)
        model.attach_attention_diagnostic_recorder(masked_recorder)
        sync_cuda(device)
        t0 = time.perf_counter()
        pred_masked = euler_sample_rawkv_controlled(
            model,
            full_k,
            full_v,
            masked_mask,
            normalizer,
            solver_steps,
            noise,
            masked_recorder,
            amp_dtype,
        )
        sync_cuda(device)
        masked_ms = (time.perf_counter() - t0) * 1000.0
        masked_attn = masked_recorder.end_sample()

        gt = np.asarray(row["trajectory"], dtype=np.float64)
        on_np = pred_on[0].float().cpu().numpy()
        masked_np = pred_masked[0].float().cpu().numpy()
        m_on = xyz_metrics(on_np, gt)
        m_masked = xyz_metrics(masked_np, gt)

        gain_ade = float(m_masked["ade_m"] - m_on["ade_m"])
        gain_fde = float(m_masked["fde_m"] - m_on["fde_m"])
        gain_heading = float(
            m_masked["heading_mae_rad"] - m_on["heading_mae_rad"]
        )
        traj_delta = np.linalg.norm(
            on_np[:, :2] - masked_np[:, :2], axis=-1
        )

        sample = {
            "index": index - 1,
            "id": sid,
            "clip": str(row.get("clip", "")),
            "mission_command": str(row.get("command", "")),
            "reasoning_text": str(cache.get("reasoning_text", "")),
            "direct_memory_tokens": int(direct_tokens),
            "reasoning_memory_tokens": int(reasoning_tokens),
            "physical_memory_tokens_both_conditions": int(total_tokens),
            "valid_memory_tokens_on": int(total_tokens),
            "valid_memory_tokens_masked": int(direct_tokens),
            "on_ade_m": m_on["ade_m"],
            "on_fde_m": m_on["fde_m"],
            "on_heading_mae_rad": m_on["heading_mae_rad"],
            "masked_ade_m": m_masked["ade_m"],
            "masked_fde_m": m_masked["fde_m"],
            "masked_heading_mae_rad": m_masked["heading_mae_rad"],
            "reasoning_kv_gain_ade_m": gain_ade,
            "reasoning_kv_gain_fde_m": gain_fde,
            "reasoning_kv_gain_heading_rad": gain_heading,
            "reasoning_kv_helped_ade": bool(gain_ade > 0.0),
            "trajectory_on_vs_masked_mean_xy_m": float(traj_delta.mean()),
            "trajectory_on_vs_masked_final_xy_m": float(traj_delta[-1]),
            "on_infer_ms": float(on_ms),
            "masked_infer_ms": float(masked_ms),
            "on_attention": on_attn,
            "masked_attention": masked_attn,
            "on_trajectory": on_np.tolist(),
            "masked_trajectory": masked_np.tolist(),
            "gt_trajectory": row["trajectory"],
        }
        samples.append(sample)

        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(rows) - index)
        if index % 10 == 0 or index == len(rows):
            print(
                f"[SAME-CKPT {index:04d}/{len(rows):04d}] "
                f"ON={m_on['ade_m']:.3f} MASK={m_masked['ade_m']:.3f} "
                f"gain={gain_ade:+.3f} "
                f"RshareMW={fmt(on_attn.get('reasoning_attention_share_within_vlm_mass_weighted'))} "
                f"RenrichMW={fmt(on_attn.get('reasoning_attention_enrichment_mass_weighted'))} "
                f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
                flush=True,
            )

        del cache, full_k_cpu, full_v_cpu, full_k, full_v
        del on_mask, masked_mask, pred_on, pred_masked
        if index % 20 == 0:
            cleanup_cuda()

    model.attach_attention_diagnostic_recorder(None)
    del model, normalizer, ckpt
    cleanup_cuda()
    return samples, on_recorder, masked_recorder, geometry


# =============================================================================
# AGGREGATION
# =============================================================================

def mean_finite(samples: Sequence[Dict[str, Any]], key: str) -> float:
    vals = np.asarray([finite_float(s.get(key)) for s in samples], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def median_finite(samples: Sequence[Dict[str, Any]], key: str) -> float:
    vals = np.asarray([finite_float(s.get(key)) for s in samples], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else float("nan")


def build_attention_gain_correlations(
    samples: Sequence[Dict[str, Any]],
    recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    gain = [s["reasoning_kv_gain_ade_m"] for s in samples]
    metrics = list(ATTENTION_METRICS) + [
        "reasoning_attention_share_within_vlm_mass_weighted",
        "reasoning_attention_enrichment_mass_weighted",
    ]

    overall = {}
    for metric in metrics:
        x = [recorder.sample_summary(s["id"]).get(metric) for s in samples]
        overall[metric] = corr_pair(x, gain)

    by_layer = {}
    for layer in range(recorder.num_layers):
        by_layer[str(layer)] = {}
        for metric in metrics:
            x = [
                recorder.sample_layer_summary(s["id"], layer).get(metric)
                for s in samples
            ]
            by_layer[str(layer)][metric] = corr_pair(x, gain)

    by_step = {}
    for step in range(recorder.solver_steps):
        by_step[str(step)] = {
            "euler_t": float(step / recorder.solver_steps),
            "metrics": {},
        }
        for metric in metrics:
            x = [
                recorder.sample_step_summary(s["id"], step).get(metric)
                for s in samples
            ]
            by_step[str(step)]["metrics"][metric] = corr_pair(x, gain)

    return {
        "gain_definition": "masked_ADE - on_ADE; positive means reasoning KV helps",
        "overall": overall,
        "by_layer": by_layer,
        "by_euler_step": by_step,
    }


def build_summary(
    samples: Sequence[Dict[str, Any]],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    gains = [s["reasoning_kv_gain_ade_m"] for s in samples]
    fde_gains = [s["reasoning_kv_gain_fde_m"] for s in samples]
    heading_gains = [s["reasoning_kv_gain_heading_rad"] for s in samples]

    on_ade = mean_finite(samples, "on_ade_m")
    masked_ade = mean_finite(samples, "masked_ade_m")
    mean_gain = mean_finite(samples, "reasoning_kv_gain_ade_m")

    wins = sum(1 for s in samples if s["reasoning_kv_gain_ade_m"] > 1.0e-9)
    losses = sum(1 for s in samples if s["reasoning_kv_gain_ade_m"] < -1.0e-9)
    ties = len(samples) - wins - losses

    return {
        "experiment": "same_reasoning_checkpoint_reasoning_kv_on_vs_masked",
        "samples": len(samples),
        "gain_definition": "masked - on; positive means reasoning KV improves the same checkpoint",
        "overall": {
            "on_ade_m": on_ade,
            "masked_ade_m": masked_ade,
            "reasoning_kv_gain_ade_m": mean_gain,
            "reasoning_kv_gain_ade_median_m": median_finite(
                samples, "reasoning_kv_gain_ade_m"
            ),
            "relative_ade_improvement_vs_masked": (
                mean_gain / masked_ade if math.isfinite(masked_ade) and masked_ade > 0 else float("nan")
            ),
            "on_fde_m": mean_finite(samples, "on_fde_m"),
            "masked_fde_m": mean_finite(samples, "masked_fde_m"),
            "reasoning_kv_gain_fde_m": mean_finite(
                samples, "reasoning_kv_gain_fde_m"
            ),
            "on_heading_mae_rad": mean_finite(samples, "on_heading_mae_rad"),
            "masked_heading_mae_rad": mean_finite(
                samples, "masked_heading_mae_rad"
            ),
            "reasoning_kv_gain_heading_rad": mean_finite(
                samples, "reasoning_kv_gain_heading_rad"
            ),
            "win_rate": float(wins / len(samples)) if samples else float("nan"),
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
            "mean_on_vs_masked_trajectory_distance_m": mean_finite(
                samples, "trajectory_on_vs_masked_mean_xy_m"
            ),
        },
        "paired_bootstrap_95ci": {
            "ade_gain_m": bootstrap_mean_ci(gains, bootstrap, seed),
            "fde_gain_m": bootstrap_mean_ci(fde_gains, bootstrap, seed + 1),
            "heading_gain_rad": bootstrap_mean_ci(
                heading_gains, bootstrap, seed + 2
            ),
        },
    }


def build_attention_diagnostics(
    samples: Sequence[Dict[str, Any]],
    on_recorder: AttentionDiagnosticRecorder,
    masked_recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    return {
        "version": "same_checkpoint_reasoning_kv_mask_attention_v2",
        "definitions": {
            "reasoning_attention_share_within_vlm": (
                "macro mean of per-head/query reasoning_mass / VLM_mass"
            ),
            "reasoning_attention_enrichment": (
                "macro share divided by physical reasoning-token fraction"
            ),
            "reasoning_attention_share_within_vlm_mass_weighted": (
                "ratio-of-sums: sum(reasoning attention mass) / sum(VLM attention mass)"
            ),
            "reasoning_attention_enrichment_mass_weighted": (
                "sum(reasoning mass) / sum(VLM mass * physical reasoning-token fraction); "
                "1 is the token-count null expectation"
            ),
            "cross_scene_mean_cosine": (
                "mean off-diagonal cosine across scene vectors, accumulated online"
            ),
        },
        "on": {
            "role": "reasoning_kv_on",
            "overall": on_recorder.overall_summary(),
            "by_layer": on_recorder.by_layer_summary(),
            "by_euler_step": on_recorder.by_step_summary(),
            "by_layer_step": on_recorder.by_layer_step_summary(),
            "cross_scene_collapse": on_recorder.collapse_summary(),
        },
        "masked": {
            "role": "reasoning_kv_masked",
            "overall": masked_recorder.overall_summary(),
            "by_layer": masked_recorder.by_layer_summary(),
            "by_euler_step": masked_recorder.by_step_summary(),
            "by_layer_step": masked_recorder.by_layer_step_summary(),
            "cross_scene_collapse": masked_recorder.collapse_summary(),
        },
        "on_attention_metric_vs_same_checkpoint_gain": build_attention_gain_correlations(
            samples, on_recorder
        ),
    }


def build_counterexamples(samples: Sequence[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    keys = [
        "index", "id", "clip", "mission_command", "reasoning_text",
        "reasoning_memory_tokens", "on_ade_m", "masked_ade_m",
        "reasoning_kv_gain_ade_m", "on_fde_m", "masked_fde_m",
        "trajectory_on_vs_masked_mean_xy_m", "on_attention",
    ]

    def compact(s: Dict[str, Any]) -> Dict[str, Any]:
        return {k: s.get(k) for k in keys}

    best = sorted(samples, key=lambda s: s["reasoning_kv_gain_ade_m"], reverse=True)[:topk]
    worst = sorted(samples, key=lambda s: s["reasoning_kv_gain_ade_m"])[:topk]
    return {
        "gain_definition": "masked ADE - on ADE; positive means reasoning KV helps",
        "largest_reasoning_kv_improvements": [compact(s) for s in best],
        "largest_reasoning_kv_degradations": [compact(s) for s in worst],
    }


# =============================================================================
# CSV / PLOTS / REPORT
# =============================================================================

def flatten_for_csv(s: Dict[str, Any]) -> Dict[str, Any]:
    attn = s.get("on_attention", {})
    return {
        "index": s.get("index"),
        "id": s.get("id"),
        "clip": s.get("clip"),
        "mission_command": s.get("mission_command"),
        "direct_memory_tokens": s.get("direct_memory_tokens"),
        "reasoning_memory_tokens": s.get("reasoning_memory_tokens"),
        "physical_memory_tokens_both_conditions": s.get(
            "physical_memory_tokens_both_conditions"
        ),
        "on_ade_m": s.get("on_ade_m"),
        "masked_ade_m": s.get("masked_ade_m"),
        "reasoning_kv_gain_ade_m": s.get("reasoning_kv_gain_ade_m"),
        "on_fde_m": s.get("on_fde_m"),
        "masked_fde_m": s.get("masked_fde_m"),
        "reasoning_kv_gain_fde_m": s.get("reasoning_kv_gain_fde_m"),
        "on_heading_mae_rad": s.get("on_heading_mae_rad"),
        "masked_heading_mae_rad": s.get("masked_heading_mae_rad"),
        "reasoning_kv_gain_heading_rad": s.get("reasoning_kv_gain_heading_rad"),
        "reasoning_kv_helped_ade": s.get("reasoning_kv_helped_ade"),
        "trajectory_on_vs_masked_mean_xy_m": s.get(
            "trajectory_on_vs_masked_mean_xy_m"
        ),
        "reasoning_attention_mass": attn.get("reasoning_attention_mass"),
        "reasoning_attention_share_macro": attn.get(
            "reasoning_attention_share_within_vlm"
        ),
        "reasoning_attention_enrichment_macro": attn.get(
            "reasoning_attention_enrichment"
        ),
        "reasoning_attention_share_mass_weighted": attn.get(
            "reasoning_attention_share_within_vlm_mass_weighted"
        ),
        "reasoning_attention_enrichment_mass_weighted": attn.get(
            "reasoning_attention_enrichment_mass_weighted"
        ),
        "reasoning_effective_token_fraction": attn.get(
            "reasoning_effective_token_fraction"
        ),
        "top1_reasoning_token_share": attn.get("top1_reasoning_token_share"),
        "reasoning_text": s.get("reasoning_text"),
    }


def write_csv(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    rows = [flatten_for_csv(s) for s in samples]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def maybe_make_plots(plot_dir: Path, samples: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; plots skipped: {exc}")
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    # Paired ADE scatter.
    x = np.asarray([s["masked_ade_m"] for s in samples], dtype=np.float64)
    y = np.asarray([s["on_ade_m"] for s in samples], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.6)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Reasoning KV masked ADE (m)")
    ax.set_ylabel("Reasoning KV ON ADE (m)")
    ax.set_title("Same checkpoint: ON vs masked")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = plot_dir / "same_checkpoint_on_vs_masked_ade.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    outputs.append(str(p))

    # Correct mass-weighted reasoning share vs causal gain.
    rx = np.asarray([
        finite_float(s["on_attention"].get(
            "reasoning_attention_share_within_vlm_mass_weighted"
        ))
        for s in samples
    ], dtype=np.float64)
    gy = np.asarray([s["reasoning_kv_gain_ade_m"] for s in samples], dtype=np.float64)
    mask = np.isfinite(rx) & np.isfinite(gy)
    if mask.sum() >= 3:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(rx[mask], gy[mask], alpha=0.6)
        if np.std(rx[mask]) > 1.0e-12:
            coef = np.polyfit(rx[mask], gy[mask], 1)
            xx = np.linspace(float(rx[mask].min()), float(rx[mask].max()), 100)
            ax.plot(xx, coef[0] * xx + coef[1])
        ax.set_xlabel("Reasoning attention share within VLM (mass-weighted)")
        ax.set_ylabel("Same-checkpoint reasoning KV gain (m)")
        ax.set_title(f"Spearman={spearman_corr(rx[mask], gy[mask]):.3f}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = plot_dir / "reasoning_attention_share_vs_same_checkpoint_gain.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        outputs.append(str(p))

    return outputs


def write_report(
    path: Path,
    summary: Dict[str, Any],
    attention: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    o = summary["overall"]
    ci = summary["paired_bootstrap_95ci"]["ade_gain_m"]
    on_a = attention["on"]["overall"]
    masked_a = attention["masked"]["overall"]
    corr = attention["on_attention_metric_vs_same_checkpoint_gain"]["overall"]
    collapse_on = attention["on"]["cross_scene_collapse"]["attended_context"]["overall"]
    collapse_mask = attention["masked"]["cross_scene_collapse"]["attended_context"]["overall"]
    collapse_r = attention["on"]["cross_scene_collapse"]["reasoning_only_context"]["overall"]

    lines = [
        "# Same-Checkpoint Reasoning KV ON vs MASKED",
        "",
        "## 1. Causal intervention",
        "",
        f"- Samples: {summary['samples']}",
        f"- SINGLE checkpoint for both conditions: `{args.reasoning_ckpt}`",
        f"- Trace cache: `{args.trace_cache}`",
        "- Same model instance: **YES**",
        "- Same full K/V tensor: **YES**",
        "- Same physical prefix length: **YES**",
        "- Same per-sample Gaussian x0: **YES**",
        "- Same Euler solver: **YES**",
        "- Only changed variable: generated reasoning-token positions are readable vs masked in `memory_mask`.",
        "",
        "```text",
        "Reasoning KV ON     : [prompt KV][reasoning KV]  mask = [1 ... 1][1 ... 1]",
        "Reasoning KV MASKED : [prompt KV][reasoning KV]  mask = [1 ... 1][0 ... 0]",
        "```",
        "",
        "This removes the separately-trained-checkpoint confound from the previous Direct-vs-Reasoning comparison.",
        "",
        "## 2. Main paired result",
        "",
        "`gain = ADE(masked) - ADE(on)`; positive means reasoning KV helps the same checkpoint.",
        "",
        "| Metric | Reasoning KV ON | Reasoning KV masked | ON benefit |",
        "|---|---:|---:|---:|",
        f"| ADE | {fmt(o['on_ade_m'])} m | {fmt(o['masked_ade_m'])} m | {fmt(o['reasoning_kv_gain_ade_m'])} m |",
        f"| FDE | {fmt(o['on_fde_m'])} m | {fmt(o['masked_fde_m'])} m | {fmt(o['reasoning_kv_gain_fde_m'])} m |",
        f"| Heading MAE | {fmt(o['on_heading_mae_rad'])} rad | {fmt(o['masked_heading_mae_rad'])} rad | {fmt(o['reasoning_kv_gain_heading_rad'])} rad |",
        "",
        f"- ADE win rate: **{pct(o['win_rate'])}** ({o['wins']} wins / {o['losses']} losses / {o['ties']} ties)",
        f"- Mean ADE gain 95% paired-bootstrap CI: **[{fmt(ci['lo95'])}, {fmt(ci['hi95'])}] m**",
        f"- Relative ADE improvement vs masked: **{pct(o['relative_ade_improvement_vs_masked'])}**",
        "",
        "## 3. Attention usage in the ON condition",
        "",
        "Both macro and corrected mass-weighted definitions are reported.",
        "",
        "| Metric | ON | MASKED |",
        "|---|---:|---:|",
        f"| VLM attention mass | {fmt(on_a['vlm_attention_mass'])} | {fmt(masked_a['vlm_attention_mass'])} |",
        f"| Reasoning attention mass | {fmt(on_a['reasoning_attention_mass'])} | {fmt(masked_a['reasoning_attention_mass'])} |",
        f"| Reasoning share (macro mean-of-ratios) | {fmt(on_a['reasoning_attention_share_within_vlm'])} | {fmt(masked_a['reasoning_attention_share_within_vlm'])} |",
        f"| Reasoning enrichment (macro) | {fmt(on_a['reasoning_attention_enrichment'])} | {fmt(masked_a['reasoning_attention_enrichment'])} |",
        f"| **Reasoning share (mass-weighted ratio-of-sums)** | **{fmt(on_a['reasoning_attention_share_within_vlm_mass_weighted'])}** | **{fmt(masked_a['reasoning_attention_share_within_vlm_mass_weighted'])}** |",
        f"| **Reasoning enrichment (mass-weighted)** | **{fmt(on_a['reasoning_attention_enrichment_mass_weighted'])}** | **{fmt(masked_a['reasoning_attention_enrichment_mass_weighted'])}** |",
        f"| Reasoning effective token fraction | {fmt(on_a['reasoning_effective_token_fraction'])} | {fmt(masked_a['reasoning_effective_token_fraction'])} |",
        f"| Top-1 reasoning token share | {fmt(on_a['top1_reasoning_token_share'])} | {fmt(masked_a['top1_reasoning_token_share'])} |",
        "",
        "## 4. Does stronger reasoning attention predict the causal ON-vs-MASKED gain?",
        "",
        "| ON attention metric vs gain | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]

    for label, key in [
        ("Reasoning attention mass", "reasoning_attention_mass"),
        ("Reasoning share macro", "reasoning_attention_share_within_vlm"),
        ("Reasoning enrichment macro", "reasoning_attention_enrichment"),
        ("Reasoning share mass-weighted", "reasoning_attention_share_within_vlm_mass_weighted"),
        ("Reasoning enrichment mass-weighted", "reasoning_attention_enrichment_mass_weighted"),
        ("Reasoning effective token fraction", "reasoning_effective_token_fraction"),
        ("Top-1 reasoning token share", "top1_reasoning_token_share"),
    ]:
        c = corr[key]
        lines.append(
            f"| {label} | {c['n']} | {fmt(c['pearson'])} | {fmt(c['spearman'])} |"
        )

    lines += [
        "",
        "## 5. Cross-scene context collapse (online; no raw vector file)",
        "",
        "| Context | Mean off-diagonal cosine | Centroid norm ratio |",
        "|---|---:|---:|",
        f"| Reasoning KV ON attended context | {fmt(collapse_on['cross_scene_mean_cosine'])} | {fmt(collapse_on['centroid_norm_ratio'])} |",
        f"| Reasoning KV MASKED attended context | {fmt(collapse_mask['cross_scene_mean_cosine'])} | {fmt(collapse_mask['centroid_norm_ratio'])} |",
        f"| ON reasoning-only context | {fmt(collapse_r['cross_scene_mean_cosine'])} | {fmt(collapse_r['centroid_norm_ratio'])} |",
        "",
        "## 6. Interpretation",
        "",
        "- If the paired ADE gain is positive and its 95% CI excludes 0, the same reasoning-trained planner performs better when it can read its generated reasoning KV.",
        "- If the mean gain is near 0 or the CI spans 0, the current evidence does not show a stable causal benefit from exposing reasoning KV at inference.",
        "- If the gain is negative with CI below 0, masking reasoning KV actually improves the same checkpoint, meaning the added reasoning memory is harmful on average despite being used by attention.",
        "- Attention↔gain correlation answers a different question: whether samples that attend more strongly to reasoning are the samples that benefit more from unmasking it.",
        "",
        "### Remaining limitation",
        "",
        "This is an inference-time intervention on a model trained with reasoning memory. Therefore MASKED is intentionally an out-of-training-distribution intervention. It isolates the necessity/usefulness of reasoning memory for this trained checkpoint, but it does not replace a separately trained no-reasoning control model.",
        "",
        "Full layer/Euler-step diagnostics are in `attention_diagnostics.json`.",
        "No `attention_vectors.pt` is generated by this script.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Same reasoning-trained checkpoint: reasoning KV ON vs reasoning-token positions masked."
        )
    )
    p.add_argument("--jsonl", type=Path, default=TEST_JSONL)
    p.add_argument("--trace-cache", type=Path, default=TRACE_CACHE_ROOT)
    p.add_argument("--reasoning-ckpt", type=Path, default=REASONING_CKPT)
    p.add_argument("--output", type=Path, default=RESULT_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--flow-noise-seed", type=int, default=FLOW_NOISE_SEED)
    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--counterexample-topk", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Debug only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("jsonl", "trace_cache", "reasoning_ckpt", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    if not args.trace_cache.is_dir():
        raise FileNotFoundError(args.trace_cache)
    if not args.reasoning_ckpt.is_file():
        raise FileNotFoundError(args.reasoning_ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 selected but this GPU does not support BF16")

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {args.output}\nUse --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(x) for x in read_jsonl(args.jsonl)]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError("No samples")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate sample IDs")

    trace_manifest = validate_trace_cache(args.trace_cache, rows)

    set_seed(SEED)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 132)
    print("SAME-CHECKPOINT REASONING KV CAUSAL ABLATION | ON vs MASKED")
    print("=" * 132)
    print("GPU             :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset         :", args.jsonl)
    print("Trace cache     :", args.trace_cache)
    print("Checkpoint BOTH :", args.reasoning_ckpt)
    print("Samples         :", len(rows))
    print("Same model      : YES")
    print("Same full K/V   : YES")
    print("Same K/V length : YES")
    print("Same Flow x0    : YES")
    print("Only difference : memory_mask over reasoning-token positions")
    print("Output          :", args.output)
    print("=" * 132)

    samples, on_recorder, masked_recorder, geometry = evaluate_same_checkpoint_on_vs_masked(
        checkpoint_path=args.reasoning_ckpt,
        rows=rows,
        trace_manifest=trace_manifest,
        device=device,
        noise_seed=args.flow_noise_seed,
        solver_steps_override=args.solver_steps,
        amp_dtype=amp_dtype,
    )

    summary = build_summary(samples, args.bootstrap, SEED)
    attention = build_attention_diagnostics(samples, on_recorder, masked_recorder)
    counterexamples = build_counterexamples(samples, args.counterexample_topk)

    summary["attention_overview"] = {
        "on": attention["on"]["overall"],
        "masked": attention["masked"]["overall"],
        "on_attention_vs_gain": attention[
            "on_attention_metric_vs_same_checkpoint_gain"
        ]["overall"],
    }
    summary["runtime"] = {
        "dataset": str(args.jsonl),
        "dataset_sha256": sha256_file(args.jsonl),
        "trace_cache": str(args.trace_cache),
        "checkpoint_both_conditions": str(args.reasoning_ckpt),
        "checkpoint_sha256": sha256_file(args.reasoning_ckpt),
        "same_checkpoint": True,
        "same_model_instance": True,
        "same_full_kv_tensor": True,
        "same_physical_prefix_length": True,
        "same_noise_per_sample": True,
        "masked_intervention": "reasoning-token positions set False in memory_mask",
        "flow_noise_seed": int(args.flow_noise_seed),
        "amp_dtype": args.amp_dtype,
        "bootstrap_iterations": int(args.bootstrap),
        "limit": args.limit,
        "model_geometry": geometry,
    }

    write_jsonl(args.output / "samples.jsonl", samples)
    write_csv(args.output / "samples.csv", samples)
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "attention_diagnostics.json", attention)
    save_json(args.output / "counterexamples.json", counterexamples)

    plots = maybe_make_plots(args.output / "plots", samples)
    summary["plots"] = plots
    save_json(args.output / "summary.json", summary)
    write_report(args.output / "report.md", summary, attention, args)

    o = summary["overall"]
    ci = summary["paired_bootstrap_95ci"]["ade_gain_m"]
    on_a = attention["on"]["overall"]

    print("\n" + "=" * 132)
    print("DONE")
    print("=" * 132)
    print("Reasoning KV ON ADE        :", fmt(o["on_ade_m"]), "m")
    print("Reasoning KV MASKED ADE    :", fmt(o["masked_ade_m"]), "m")
    print("Same-ckpt reasoning gain   :", fmt(o["reasoning_kv_gain_ade_m"]), "m")
    print("Paired bootstrap 95% CI    :", f"[{fmt(ci['lo95'])}, {fmt(ci['hi95'])}] m")
    print("ON win rate                :", pct(o["win_rate"]))
    print("Reasoning share macro      :", fmt(
        on_a.get("reasoning_attention_share_within_vlm")
    ))
    print("Reasoning enrichment macro :", fmt(
        on_a.get("reasoning_attention_enrichment")
    ))
    print("Reasoning share mass-weight:", fmt(
        on_a.get("reasoning_attention_share_within_vlm_mass_weighted")
    ))
    print("Reasoning enrich mass-weight:", fmt(
        on_a.get("reasoning_attention_enrichment_mass_weighted")
    ))
    print("Report                     :", args.output / "report.md")
    print("Summary                    :", args.output / "summary.json")
    print("Attention diagnostics      :", args.output / "attention_diagnostics.json")
    print("Samples                    :", args.output / "samples.jsonl")
    print("Counterexamples            :", args.output / "counterexamples.json")
    print("attention_vectors.pt       : NOT GENERATED")
    if plots:
        print("Plots                      :", args.output / "plots")



# =============================================================================
# OVERRIDE: EULER-STAGE REASONING-KV INTERVENTION
# =============================================================================
# This block intentionally reuses every model/cache/diagnostic implementation above.
# It changes only the evaluation protocol and CLI/main entrypoint.

EULER_STAGE_RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/results/reasoning_kv_euler_stage_same_checkpoint"
)


@torch.inference_mode()
def euler_sample_rawkv_scheduled(
    model,
    vlm_key,
    vlm_value,
    direct_tokens: int,
    reasoning_tokens: int,
    reasoning_enabled_steps: Sequence[int],
    normalizer,
    solver_steps: int,
    noise_cpu: torch.Tensor,
    diagnostic_recorder: AttentionDiagnosticRecorder,
    amp_dtype: torch.dtype,
):
    """Euler sample while changing ONLY readability of reasoning-token KV by step.

    Physical K/V is unchanged for every condition and every step.
    Prompt/direct-memory tokens are always readable.
    Generated-reasoning positions are readable only at `reasoning_enabled_steps`.
    """
    device = vlm_key.device
    total_tokens = int(vlm_key.shape[2])
    expected = int(direct_tokens + reasoning_tokens)
    if total_tokens != expected:
        raise RuntimeError(
            f"Physical memory mismatch: total={total_tokens}, expected={expected}"
        )

    enabled = {int(s) for s in reasoning_enabled_steps}
    if any(s < 0 or s >= solver_steps for s in enabled):
        raise ValueError(
            f"Invalid reasoning_enabled_steps={sorted(enabled)}, solver_steps={solver_steps}"
        )

    x = noise_cpu.to(device=device, dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, solver_steps + 1, device=device)

    # Allocate once. We mutate only the reasoning slice before each step.
    memory_mask = torch.ones((1, total_tokens), dtype=torch.bool, device=device)
    r0 = int(direct_tokens)
    r1 = int(direct_tokens + reasoning_tokens)

    for i in range(solver_steps):
        reasoning_on = i in enabled
        memory_mask[:, r0:r1] = bool(reasoning_on)

        t_now = float(times[i].item())
        dt = float((times[i + 1] - times[i]).item())
        diagnostic_recorder.set_euler_step(i, t_now)
        t = torch.full((1, 1, 1), t_now, device=device, dtype=torch.float32)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            velocity = model(
                x_t=x,
                t=t,
                vlm_key=vlm_key,
                vlm_value=vlm_value,
                memory_mask=memory_mask,
            )
        x = x + dt * velocity.float()

    return normalizer.denormalize(x)


def _condition_steps(solver_steps: int, split_step: int) -> Dict[str, List[int]]:
    all_steps = list(range(solver_steps))
    return {
        "all_on": all_steps,
        "all_masked": [],
        "early_on": list(range(0, split_step)),
        "late_on": list(range(split_step, solver_steps)),
    }


def _condition_label(name: str) -> str:
    return {
        "all_on": "ALL_ON",
        "all_masked": "ALL_MASKED",
        "early_on": "EARLY_ON",
        "late_on": "LATE_ON",
    }[name]


def evaluate_same_checkpoint_euler_stage(
    checkpoint_path: Path,
    rows: Sequence[Dict[str, Any]],
    trace_manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    noise_seed: int,
    solver_steps_override: Optional[int],
    split_step_override: Optional[int],
    amp_dtype: torch.dtype,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, AttentionDiagnosticRecorder],
    Dict[str, Any],
    Dict[str, List[int]],
]:
    ckpt = load_checkpoint(checkpoint_path)
    model, normalizer, ckpt_steps = build_model_from_checkpoint(ckpt, device)
    solver_steps = int(solver_steps_override) if solver_steps_override else ckpt_steps

    split_step = (
        int(split_step_override)
        if split_step_override is not None
        else int(solver_steps // 2)
    )
    if not (1 <= split_step < solver_steps):
        raise ValueError(
            f"--split-step must satisfy 1 <= split_step < solver_steps; "
            f"got split_step={split_step}, solver_steps={solver_steps}"
        )

    schedules = _condition_steps(solver_steps, split_step)
    recorders: Dict[str, AttentionDiagnosticRecorder] = {
        name: AttentionDiagnosticRecorder(
            role=f"reasoning_kv_{name}",
            num_layers=len(model.layers),
            solver_steps=solver_steps,
        )
        for name in schedules
    }

    action_cfg = dict(ckpt["action_config"])
    geometry = {
        "vlm_num_attention_heads": int(ckpt["vlm_num_attention_heads"]),
        "vlm_num_kv_heads": int(ckpt["vlm_num_kv_heads"]),
        "vlm_head_dim": int(ckpt["vlm_head_dim"]),
        "num_layers": int(len(model.layers)),
        "solver_steps": int(solver_steps),
        "split_step": int(split_step),
        "action_config": action_cfg,
    }

    samples: List[Dict[str, Any]] = []
    start = time.perf_counter()

    # Fixed condition order; model.eval() means no dropout/random state is consumed.
    condition_order = ["all_on", "all_masked", "early_on", "late_on"]

    for index, (row, item) in enumerate(zip(rows, trace_manifest), 1):
        sid = row["id"]
        if sid != str(item["id"]):
            raise RuntimeError(f"ID mismatch at index={index}: {sid} != {item['id']}")

        cache = torch.load(item["cache_file"], map_location="cpu", weights_only=False)
        if str(cache.get("id", "")) != sid:
            raise RuntimeError(f"Cache ID mismatch id={sid}")

        full_k_cpu, full_v_cpu, direct_tokens, reasoning_tokens = load_full_reasoning_memory(cache)
        full_k = full_k_cpu.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        full_v = full_v_cpu.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)

        total_tokens = int(full_k.shape[2])
        if total_tokens != direct_tokens + reasoning_tokens:
            raise RuntimeError(f"Token length mismatch id={sid}")

        # Exactly the same x0 object is reused for all four conditions.
        noise = make_noise(sid, noise_seed, NUM_STEPS)

        preds: Dict[str, torch.Tensor] = {}
        infer_ms: Dict[str, float] = {}
        attn: Dict[str, Dict[str, float]] = {}

        for condition in condition_order:
            recorder = recorders[condition]
            recorder.begin_sample(sid, direct_tokens, reasoning_tokens)
            model.attach_attention_diagnostic_recorder(recorder)

            sync_cuda(device)
            t0 = time.perf_counter()
            pred = euler_sample_rawkv_scheduled(
                model=model,
                vlm_key=full_k,
                vlm_value=full_v,
                direct_tokens=direct_tokens,
                reasoning_tokens=reasoning_tokens,
                reasoning_enabled_steps=schedules[condition],
                normalizer=normalizer,
                solver_steps=solver_steps,
                noise_cpu=noise,
                diagnostic_recorder=recorder,
                amp_dtype=amp_dtype,
            )
            sync_cuda(device)
            infer_ms[condition] = (time.perf_counter() - t0) * 1000.0
            attn[condition] = recorder.end_sample()
            preds[condition] = pred

        gt = np.asarray(row["trajectory"], dtype=np.float64)
        pred_np = {
            k: v[0].float().cpu().numpy()
            for k, v in preds.items()
        }
        metrics = {k: xyz_metrics(v, gt) for k, v in pred_np.items()}

        # Positive gain => the staged condition is better than ALL_MASKED.
        gain_all = float(metrics["all_masked"]["ade_m"] - metrics["all_on"]["ade_m"])
        gain_early = float(metrics["all_masked"]["ade_m"] - metrics["early_on"]["ade_m"])
        gain_late = float(metrics["all_masked"]["ade_m"] - metrics["late_on"]["ade_m"])
        # Positive => EARLY_ON has lower ADE than LATE_ON.
        early_vs_late = float(metrics["late_on"]["ade_m"] - metrics["early_on"]["ade_m"])

        fde_gain_all = float(metrics["all_masked"]["fde_m"] - metrics["all_on"]["fde_m"])
        fde_gain_early = float(metrics["all_masked"]["fde_m"] - metrics["early_on"]["fde_m"])
        fde_gain_late = float(metrics["all_masked"]["fde_m"] - metrics["late_on"]["fde_m"])
        fde_early_vs_late = float(metrics["late_on"]["fde_m"] - metrics["early_on"]["fde_m"])

        heading_gain_all = float(
            metrics["all_masked"]["heading_mae_rad"] - metrics["all_on"]["heading_mae_rad"]
        )
        heading_gain_early = float(
            metrics["all_masked"]["heading_mae_rad"] - metrics["early_on"]["heading_mae_rad"]
        )
        heading_gain_late = float(
            metrics["all_masked"]["heading_mae_rad"] - metrics["late_on"]["heading_mae_rad"]
        )
        heading_early_vs_late = float(
            metrics["late_on"]["heading_mae_rad"] - metrics["early_on"]["heading_mae_rad"]
        )

        def mean_xy_distance(a: str, b: str) -> float:
            d = np.linalg.norm(pred_np[a][:, :2] - pred_np[b][:, :2], axis=-1)
            return float(d.mean())

        sample: Dict[str, Any] = {
            "index": index - 1,
            "id": sid,
            "clip": str(row.get("clip", "")),
            "mission_command": str(row.get("command", "")),
            "reasoning_text": str(cache.get("reasoning_text", "")),
            "direct_memory_tokens": int(direct_tokens),
            "reasoning_memory_tokens": int(reasoning_tokens),
            "physical_memory_tokens_all_conditions": int(total_tokens),
            "split_step": int(split_step),
            "early_enabled_steps": schedules["early_on"],
            "late_enabled_steps": schedules["late_on"],

            "all_on_ade_m": metrics["all_on"]["ade_m"],
            "all_masked_ade_m": metrics["all_masked"]["ade_m"],
            "early_on_ade_m": metrics["early_on"]["ade_m"],
            "late_on_ade_m": metrics["late_on"]["ade_m"],

            "all_on_fde_m": metrics["all_on"]["fde_m"],
            "all_masked_fde_m": metrics["all_masked"]["fde_m"],
            "early_on_fde_m": metrics["early_on"]["fde_m"],
            "late_on_fde_m": metrics["late_on"]["fde_m"],

            "all_on_heading_mae_rad": metrics["all_on"]["heading_mae_rad"],
            "all_masked_heading_mae_rad": metrics["all_masked"]["heading_mae_rad"],
            "early_on_heading_mae_rad": metrics["early_on"]["heading_mae_rad"],
            "late_on_heading_mae_rad": metrics["late_on"]["heading_mae_rad"],

            "all_on_gain_vs_masked_ade_m": gain_all,
            "early_on_gain_vs_masked_ade_m": gain_early,
            "late_on_gain_vs_masked_ade_m": gain_late,
            "early_vs_late_gain_ade_m": early_vs_late,

            "all_on_gain_vs_masked_fde_m": fde_gain_all,
            "early_on_gain_vs_masked_fde_m": fde_gain_early,
            "late_on_gain_vs_masked_fde_m": fde_gain_late,
            "early_vs_late_gain_fde_m": fde_early_vs_late,

            "all_on_gain_vs_masked_heading_rad": heading_gain_all,
            "early_on_gain_vs_masked_heading_rad": heading_gain_early,
            "late_on_gain_vs_masked_heading_rad": heading_gain_late,
            "early_vs_late_gain_heading_rad": heading_early_vs_late,

            "early_beats_late_ade": bool(early_vs_late > 0.0),
            "early_on_helped_vs_masked_ade": bool(gain_early > 0.0),
            "late_on_helped_vs_masked_ade": bool(gain_late > 0.0),

            "trajectory_all_on_vs_masked_mean_xy_m": mean_xy_distance("all_on", "all_masked"),
            "trajectory_early_on_vs_masked_mean_xy_m": mean_xy_distance("early_on", "all_masked"),
            "trajectory_late_on_vs_masked_mean_xy_m": mean_xy_distance("late_on", "all_masked"),
            "trajectory_early_vs_late_mean_xy_m": mean_xy_distance("early_on", "late_on"),

            "all_on_infer_ms": infer_ms["all_on"],
            "all_masked_infer_ms": infer_ms["all_masked"],
            "early_on_infer_ms": infer_ms["early_on"],
            "late_on_infer_ms": infer_ms["late_on"],

            "all_on_attention": attn["all_on"],
            "all_masked_attention": attn["all_masked"],
            "early_on_attention": attn["early_on"],
            "late_on_attention": attn["late_on"],

            "all_on_trajectory": pred_np["all_on"].tolist(),
            "all_masked_trajectory": pred_np["all_masked"].tolist(),
            "early_on_trajectory": pred_np["early_on"].tolist(),
            "late_on_trajectory": pred_np["late_on"].tolist(),
            "gt_trajectory": row["trajectory"],
        }
        samples.append(sample)

        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(rows) - index)
        if index % 10 == 0 or index == len(rows):
            print(
                f"[EULER-STAGE {index:04d}/{len(rows):04d}] "
                f"MASK={metrics['all_masked']['ade_m']:.3f} "
                f"ALL={metrics['all_on']['ade_m']:.3f} "
                f"EARLY={metrics['early_on']['ade_m']:.3f} "
                f"LATE={metrics['late_on']['ade_m']:.3f} "
                f"E-L={early_vs_late:+.3f} "
                f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
                flush=True,
            )

        del cache, full_k_cpu, full_v_cpu, full_k, full_v, preds
        if index % 20 == 0:
            cleanup_cuda()

    model.attach_attention_diagnostic_recorder(None)
    del model, normalizer, ckpt
    cleanup_cuda()
    return samples, recorders, geometry, schedules


def _paired_result(
    samples: Sequence[Dict[str, Any]],
    condition: str,
    baseline: str,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Return paired condition benefit = baseline error - condition error."""
    ade_gain = [
        finite_float(s[f"{baseline}_ade_m"]) - finite_float(s[f"{condition}_ade_m"])
        for s in samples
    ]
    fde_gain = [
        finite_float(s[f"{baseline}_fde_m"]) - finite_float(s[f"{condition}_fde_m"])
        for s in samples
    ]
    heading_gain = [
        finite_float(s[f"{baseline}_heading_mae_rad"]) - finite_float(s[f"{condition}_heading_mae_rad"])
        for s in samples
    ]
    arr = np.asarray(ade_gain, dtype=np.float64)
    wins = int(np.sum(arr > 1e-9))
    losses = int(np.sum(arr < -1e-9))
    ties = int(len(arr) - wins - losses)
    baseline_mean = mean_finite(samples, f"{baseline}_ade_m")
    gain_mean = float(np.nanmean(arr)) if len(arr) else float("nan")
    return {
        "condition": condition,
        "baseline": baseline,
        "gain_definition": f"{baseline} error - {condition} error; positive means {condition} is better",
        "condition_ade_m": mean_finite(samples, f"{condition}_ade_m"),
        "baseline_ade_m": baseline_mean,
        "ade_gain_m": gain_mean,
        "ade_gain_median_m": float(np.nanmedian(arr)) if len(arr) else float("nan"),
        "relative_ade_improvement_vs_baseline": (
            gain_mean / baseline_mean
            if math.isfinite(baseline_mean) and baseline_mean > 0
            else float("nan")
        ),
        "condition_fde_m": mean_finite(samples, f"{condition}_fde_m"),
        "baseline_fde_m": mean_finite(samples, f"{baseline}_fde_m"),
        "fde_gain_m": float(np.nanmean(np.asarray(fde_gain, dtype=np.float64))),
        "condition_heading_mae_rad": mean_finite(samples, f"{condition}_heading_mae_rad"),
        "baseline_heading_mae_rad": mean_finite(samples, f"{baseline}_heading_mae_rad"),
        "heading_gain_rad": float(np.nanmean(np.asarray(heading_gain, dtype=np.float64))),
        "win_rate": float(wins / len(arr)) if len(arr) else float("nan"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_bootstrap_95ci": {
            "ade_gain_m": bootstrap_mean_ci(ade_gain, bootstrap, seed),
            "fde_gain_m": bootstrap_mean_ci(fde_gain, bootstrap, seed + 1),
            "heading_gain_rad": bootstrap_mean_ci(heading_gain, bootstrap, seed + 2),
        },
    }


def build_euler_stage_summary(
    samples: Sequence[Dict[str, Any]],
    bootstrap: int,
    seed: int,
    schedules: Dict[str, List[int]],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    conditions = ["all_masked", "all_on", "early_on", "late_on"]
    condition_means = {}
    for c in conditions:
        condition_means[c] = {
            "ade_m": mean_finite(samples, f"{c}_ade_m"),
            "fde_m": mean_finite(samples, f"{c}_fde_m"),
            "heading_mae_rad": mean_finite(samples, f"{c}_heading_mae_rad"),
            "infer_ms": mean_finite(samples, f"{c}_infer_ms"),
        }

    comparisons = {
        "all_on_vs_all_masked": _paired_result(
            samples, "all_on", "all_masked", bootstrap, seed
        ),
        "early_on_vs_all_masked": _paired_result(
            samples, "early_on", "all_masked", bootstrap, seed + 10
        ),
        "late_on_vs_all_masked": _paired_result(
            samples, "late_on", "all_masked", bootstrap, seed + 20
        ),
        # Here baseline=late_on, condition=early_on, so positive => early better.
        "early_on_vs_late_on": _paired_result(
            samples, "early_on", "late_on", bootstrap, seed + 30
        ),
    }

    return {
        "experiment": "same_checkpoint_reasoning_kv_euler_stage_intervention",
        "samples": int(len(samples)),
        "solver_steps": int(geometry["solver_steps"]),
        "split_step": int(geometry["split_step"]),
        "schedules": schedules,
        "condition_means": condition_means,
        "paired_comparisons": comparisons,
        "trajectory_change_m": {
            "all_on_vs_masked": mean_finite(samples, "trajectory_all_on_vs_masked_mean_xy_m"),
            "early_on_vs_masked": mean_finite(samples, "trajectory_early_on_vs_masked_mean_xy_m"),
            "late_on_vs_masked": mean_finite(samples, "trajectory_late_on_vs_masked_mean_xy_m"),
            "early_vs_late": mean_finite(samples, "trajectory_early_vs_late_mean_xy_m"),
        },
    }


def build_stage_attention_gain_correlations(
    samples: Sequence[Dict[str, Any]],
    recorder: AttentionDiagnosticRecorder,
    gain_key: str,
) -> Dict[str, Any]:
    gain = [s[gain_key] for s in samples]
    metrics = list(ATTENTION_METRICS) + [
        "reasoning_attention_share_within_vlm_mass_weighted",
        "reasoning_attention_enrichment_mass_weighted",
    ]

    overall: Dict[str, Any] = {}
    for metric in metrics:
        x = [recorder.sample_summary(s["id"]).get(metric) for s in samples]
        overall[metric] = corr_pair(x, gain)

    by_layer: Dict[str, Any] = {}
    for layer in range(recorder.num_layers):
        by_layer[str(layer)] = {}
        for metric in metrics:
            x = [
                recorder.sample_layer_summary(s["id"], layer).get(metric)
                for s in samples
            ]
            by_layer[str(layer)][metric] = corr_pair(x, gain)

    by_step: Dict[str, Any] = {}
    for step in range(recorder.solver_steps):
        by_step[str(step)] = {
            "euler_t": float(step / recorder.solver_steps),
            "metrics": {},
        }
        for metric in metrics:
            x = [
                recorder.sample_step_summary(s["id"], step).get(metric)
                for s in samples
            ]
            by_step[str(step)]["metrics"][metric] = corr_pair(x, gain)

    return {
        "gain_key": gain_key,
        "overall": overall,
        "by_layer": by_layer,
        "by_euler_step": by_step,
    }


def _verify_attention_schedule(
    recorders: Dict[str, AttentionDiagnosticRecorder],
    schedules: Dict[str, List[int]],
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Hard-check that masked reasoning steps recorded zero reasoning attention."""
    checks: Dict[str, Any] = {}
    for condition, recorder in recorders.items():
        by_step = recorder.by_step_summary()
        enabled = set(schedules[condition])
        condition_check = {"passed": True, "steps": {}}
        for step in range(recorder.solver_steps):
            mass = finite_float(by_step[str(step)].get("reasoning_attention_mass"), default=0.0)
            should_be_masked = step not in enabled
            ok = True
            if should_be_masked and abs(mass) > atol:
                ok = False
                condition_check["passed"] = False
            condition_check["steps"][str(step)] = {
                "reasoning_enabled": bool(step in enabled),
                "reasoning_attention_mass": mass,
                "masked_zero_check": bool(ok),
            }
        checks[condition] = condition_check

    if not all(v["passed"] for v in checks.values()):
        raise RuntimeError(
            "Attention schedule verification failed: a masked Euler step had non-zero reasoning attention"
        )
    return checks


def build_euler_stage_attention_diagnostics(
    samples: Sequence[Dict[str, Any]],
    recorders: Dict[str, AttentionDiagnosticRecorder],
    schedules: Dict[str, List[int]],
) -> Dict[str, Any]:
    schedule_checks = _verify_attention_schedule(recorders, schedules)
    payload: Dict[str, Any] = {
        "version": "same_checkpoint_reasoning_kv_euler_stage_attention_v1",
        "definitions": {
            "reasoning_attention_share_within_vlm_mass_weighted": (
                "ratio-of-sums: sum(reasoning attention mass) / sum(VLM attention mass)"
            ),
            "reasoning_attention_enrichment_mass_weighted": (
                "sum(reasoning mass) / sum(VLM mass * physical reasoning-token fraction); "
                "1 is token-count null expectation"
            ),
            "early_on": "reasoning KV readable only before split_step",
            "late_on": "reasoning KV readable only from split_step onward",
        },
        "schedules": schedules,
        "schedule_verification": schedule_checks,
        "conditions": {},
        "attention_vs_gain": {},
    }

    for condition, recorder in recorders.items():
        payload["conditions"][condition] = {
            "role": recorder.role,
            "overall": recorder.overall_summary(),
            "by_layer": recorder.by_layer_summary(),
            "by_euler_step": recorder.by_step_summary(),
            "by_layer_step": recorder.by_layer_step_summary(),
            "cross_scene_collapse": recorder.collapse_summary(),
        }

    payload["attention_vs_gain"]["all_on"] = build_stage_attention_gain_correlations(
        samples, recorders["all_on"], "all_on_gain_vs_masked_ade_m"
    )
    payload["attention_vs_gain"]["early_on"] = build_stage_attention_gain_correlations(
        samples, recorders["early_on"], "early_on_gain_vs_masked_ade_m"
    )
    payload["attention_vs_gain"]["late_on"] = build_stage_attention_gain_correlations(
        samples, recorders["late_on"], "late_on_gain_vs_masked_ade_m"
    )
    return payload


def build_euler_stage_counterexamples(
    samples: Sequence[Dict[str, Any]], topk: int
) -> Dict[str, Any]:
    keys = [
        "index", "id", "clip", "mission_command", "reasoning_text",
        "reasoning_memory_tokens", "all_masked_ade_m", "all_on_ade_m",
        "early_on_ade_m", "late_on_ade_m", "all_on_gain_vs_masked_ade_m",
        "early_on_gain_vs_masked_ade_m", "late_on_gain_vs_masked_ade_m",
        "early_vs_late_gain_ade_m", "trajectory_early_vs_late_mean_xy_m",
    ]

    def compact(s: Dict[str, Any]) -> Dict[str, Any]:
        return {k: s.get(k) for k in keys}

    return {
        "gain_definition": {
            "vs_masked": "masked ADE - staged ADE; positive means staged reasoning helps",
            "early_vs_late": "late ADE - early ADE; positive means EARLY_ON is better",
        },
        "largest_early_better_than_late": [
            compact(s) for s in sorted(
                samples, key=lambda x: x["early_vs_late_gain_ade_m"], reverse=True
            )[:topk]
        ],
        "largest_late_better_than_early": [
            compact(s) for s in sorted(
                samples, key=lambda x: x["early_vs_late_gain_ade_m"]
            )[:topk]
        ],
        "largest_early_benefit_vs_masked": [
            compact(s) for s in sorted(
                samples, key=lambda x: x["early_on_gain_vs_masked_ade_m"], reverse=True
            )[:topk]
        ],
        "largest_late_benefit_vs_masked": [
            compact(s) for s in sorted(
                samples, key=lambda x: x["late_on_gain_vs_masked_ade_m"], reverse=True
            )[:topk]
        ],
    }


def flatten_euler_stage_for_csv(s: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "index": s.get("index"),
        "id": s.get("id"),
        "clip": s.get("clip"),
        "mission_command": s.get("mission_command"),
        "direct_memory_tokens": s.get("direct_memory_tokens"),
        "reasoning_memory_tokens": s.get("reasoning_memory_tokens"),
        "split_step": s.get("split_step"),
    }
    for c in ("all_masked", "all_on", "early_on", "late_on"):
        out[f"{c}_ade_m"] = s.get(f"{c}_ade_m")
        out[f"{c}_fde_m"] = s.get(f"{c}_fde_m")
        out[f"{c}_heading_mae_rad"] = s.get(f"{c}_heading_mae_rad")
        attn = s.get(f"{c}_attention", {})
        out[f"{c}_reasoning_attention_mass"] = attn.get("reasoning_attention_mass")
        out[f"{c}_reasoning_share_mw"] = attn.get(
            "reasoning_attention_share_within_vlm_mass_weighted"
        )
        out[f"{c}_reasoning_enrichment_mw"] = attn.get(
            "reasoning_attention_enrichment_mass_weighted"
        )
    for k in (
        "all_on_gain_vs_masked_ade_m",
        "early_on_gain_vs_masked_ade_m",
        "late_on_gain_vs_masked_ade_m",
        "early_vs_late_gain_ade_m",
        "trajectory_all_on_vs_masked_mean_xy_m",
        "trajectory_early_on_vs_masked_mean_xy_m",
        "trajectory_late_on_vs_masked_mean_xy_m",
        "trajectory_early_vs_late_mean_xy_m",
    ):
        out[k] = s.get(k)
    return out


def write_euler_stage_csv(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    rows = [flatten_euler_stage_for_csv(s) for s in samples]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_euler_stage_plots(
    plot_dir: Path,
    samples: Sequence[Dict[str, Any]],
    attention: Dict[str, Any],
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    # 1) Mean ADE by condition.
    names = ["ALL_MASKED", "ALL_ON", "EARLY_ON", "LATE_ON"]
    keys = ["all_masked_ade_m", "all_on_ade_m", "early_on_ade_m", "late_on_ade_m"]
    vals = [mean_finite(samples, k) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals)
    ax.set_ylabel("ADE (m)")
    ax.set_title("Same-checkpoint Euler-stage reasoning KV intervention")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = plot_dir / "condition_mean_ade.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    # 2) Early gain vs late gain per sample.
    x = np.asarray([s["early_on_gain_vs_masked_ade_m"] for s in samples], dtype=float)
    y = np.asarray([s["late_on_gain_vs_masked_ade_m"] for s in samples], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.6)
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("EARLY_ON benefit vs MASKED (m)")
    ax.set_ylabel("LATE_ON benefit vs MASKED (m)")
    ax.set_title(f"Early vs late per-sample benefit | Spearman={fmt(spearman_corr(x, y), 3)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = plot_dir / "early_gain_vs_late_gain.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    # 3) Step-wise mass-weighted reasoning share.
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in ("all_on", "early_on", "late_on"):
        by_step = attention["conditions"][c]["by_euler_step"]
        steps = sorted(int(k) for k in by_step.keys())
        ys = [
            finite_float(
                by_step[str(i)].get("reasoning_attention_share_within_vlm_mass_weighted"),
                default=0.0,
            )
            for i in steps
        ]
        ax.plot(steps, ys, marker="o", label=_condition_label(c))
    ax.set_xlabel("Euler step")
    ax.set_ylabel("Reasoning attention share within VLM (mass-weighted)")
    ax.set_title("Reasoning KV usage by Euler step")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = plot_dir / "reasoning_share_by_euler_step.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    return paths


def write_euler_stage_report(
    path: Path,
    summary: Dict[str, Any],
    attention: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cm = summary["condition_means"]
    pc = summary["paired_comparisons"]
    split = summary["split_step"]
    steps = summary["solver_steps"]

    def comp_line(name: str) -> str:
        c = pc[name]
        ci = c["paired_bootstrap_95ci"]["ade_gain_m"]
        return (
            f"| {name} | {fmt(c['ade_gain_m'])} | "
            f"[{fmt(ci['lo95'])}, {fmt(ci['hi95'])}] | {pct(c['win_rate'])} |"
        )

    lines = [
        "# Same-Checkpoint Reasoning KV Euler-Stage Intervention",
        "",
        "## 1. Intervention",
        "",
        f"- Samples: {summary['samples']}",
        f"- SINGLE checkpoint: `{args.reasoning_ckpt}`",
        f"- Solver steps: {steps}",
        f"- Split step: {split}",
        f"- EARLY_ON: reasoning readable at steps `{summary['schedules']['early_on']}`",
        f"- LATE_ON: reasoning readable at steps `{summary['schedules']['late_on']}`",
        "- Same model instance: **YES**",
        "- Same full K/V tensor: **YES**",
        "- Same physical prefix length: **YES**",
        "- Same per-sample Gaussian x0: **YES**",
        "- Only intervention: reasoning-token readability as a function of Euler step.",
        "",
        "```text",
        "ALL_ON     : R R R R R | R R R R R",
        "ALL_MASKED : - - - - - | - - - - -",
        "EARLY_ON   : R R R R R | - - - - -",
        "LATE_ON    : - - - - - | R R R R R",
        "```",
        "",
        "## 2. Mean trajectory metrics",
        "",
        "| Condition | ADE (m) | FDE (m) | Heading MAE (rad) |",
        "|---|---:|---:|---:|",
    ]
    for c in ("all_masked", "all_on", "early_on", "late_on"):
        lines.append(
            f"| {_condition_label(c)} | {fmt(cm[c]['ade_m'])} | "
            f"{fmt(cm[c]['fde_m'])} | {fmt(cm[c]['heading_mae_rad'])} |"
        )

    lines += [
        "",
        "## 3. Paired ADE comparisons",
        "",
        "Positive gain means the condition named first has lower ADE than its paired baseline.",
        "",
        "| Comparison | Mean ADE benefit (m) | 95% paired-bootstrap CI | Win rate |",
        "|---|---:|---:|---:|",
        comp_line("all_on_vs_all_masked"),
        comp_line("early_on_vs_all_masked"),
        comp_line("late_on_vs_all_masked"),
        comp_line("early_on_vs_late_on"),
        "",
        "## 4. Interpretation rule",
        "",
        "- **EARLY_ON > MASKED with CI above 0, while LATE_ON ~ MASKED** → evidence that reasoning is useful mainly during coarse/early trajectory formation.",
        "- **LATE_ON > MASKED with CI above 0, while EARLY_ON ~ MASKED** → evidence that reasoning is useful mainly during late geometric refinement.",
        "- **EARLY_ON > LATE_ON with CI above 0** → direct staged evidence favoring early reasoning access.",
        "- **ALL_ON > both EARLY_ON and LATE_ON** → reasoning information may be useful across stages or have sequential interaction effects.",
        "- **All CIs span 0** → previous step-wise correlation trend was likely weak/noisy rather than a robust stage-specific effect.",
        "",
        "## 5. Attention schedule verification",
        "",
        "Masked steps are hard-checked to have zero reasoning attention mass. Failure raises an exception instead of silently writing results.",
        "",
        "Full layer/Euler-step attention diagnostics are in `attention_diagnostics.json`.",
        "No `attention_vectors.pt` is generated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Same reasoning-trained checkpoint: ALL_ON vs ALL_MASKED vs EARLY_ON vs LATE_ON "
            "reasoning-KV intervention across Euler solver steps."
        )
    )
    p.add_argument("--jsonl", type=Path, default=TEST_JSONL)
    p.add_argument("--trace-cache", type=Path, default=TRACE_CACHE_ROOT)
    p.add_argument("--reasoning-ckpt", type=Path, default=REASONING_CKPT)
    p.add_argument("--output", type=Path, default=EULER_STAGE_RESULT_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--flow-noise-seed", type=int, default=FLOW_NOISE_SEED)
    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument(
        "--split-step",
        type=int,
        default=None,
        help=(
            "First Euler step assigned to LATE_ON. Default = solver_steps//2. "
            "For 10 steps: 5 => EARLY 0..4, LATE 5..9."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--counterexample-topk", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Debug only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("jsonl", "trace_cache", "reasoning_ckpt", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    if not args.trace_cache.is_dir():
        raise FileNotFoundError(args.trace_cache)
    if not args.reasoning_ckpt.is_file():
        raise FileNotFoundError(args.reasoning_ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 selected but this GPU does not support BF16")

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {args.output}\nUse --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(x) for x in read_jsonl(args.jsonl)]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError("No samples")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate sample IDs")

    trace_manifest = validate_trace_cache(args.trace_cache, rows)

    set_seed(SEED)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 132)
    print("SAME-CHECKPOINT REASONING KV | EULER-STAGE INTERVENTION")
    print("=" * 132)
    print("GPU             :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset         :", args.jsonl)
    print("Trace cache     :", args.trace_cache)
    print("Checkpoint ALL  :", args.reasoning_ckpt)
    print("Samples         :", len(rows))
    print("Same model      : YES")
    print("Same full K/V   : YES")
    print("Same K/V length : YES")
    print("Same Flow x0    : YES")
    print("Only difference : reasoning memory_mask schedule by Euler step")
    print("Output          :", args.output)
    print("=" * 132)

    samples, recorders, geometry, schedules = evaluate_same_checkpoint_euler_stage(
        checkpoint_path=args.reasoning_ckpt,
        rows=rows,
        trace_manifest=trace_manifest,
        device=device,
        noise_seed=args.flow_noise_seed,
        solver_steps_override=args.solver_steps,
        split_step_override=args.split_step,
        amp_dtype=amp_dtype,
    )

    summary = build_euler_stage_summary(
        samples=samples,
        bootstrap=args.bootstrap,
        seed=SEED,
        schedules=schedules,
        geometry=geometry,
    )
    attention = build_euler_stage_attention_diagnostics(samples, recorders, schedules)
    counterexamples = build_euler_stage_counterexamples(samples, args.counterexample_topk)

    summary["attention_overview"] = {
        c: attention["conditions"][c]["overall"]
        for c in ("all_masked", "all_on", "early_on", "late_on")
    }
    summary["runtime"] = {
        "dataset": str(args.jsonl),
        "dataset_sha256": sha256_file(args.jsonl),
        "trace_cache": str(args.trace_cache),
        "checkpoint_all_conditions": str(args.reasoning_ckpt),
        "checkpoint_sha256": sha256_file(args.reasoning_ckpt),
        "same_checkpoint": True,
        "same_model_instance": True,
        "same_full_kv_tensor": True,
        "same_physical_prefix_length": True,
        "same_noise_per_sample": True,
        "intervention": "reasoning-token positions toggled in memory_mask according to Euler step",
        "schedules": schedules,
        "flow_noise_seed": int(args.flow_noise_seed),
        "amp_dtype": args.amp_dtype,
        "bootstrap_iterations": int(args.bootstrap),
        "limit": args.limit,
        "model_geometry": geometry,
    }

    write_jsonl(args.output / "samples.jsonl", samples)
    write_euler_stage_csv(args.output / "samples.csv", samples)
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "attention_diagnostics.json", attention)
    save_json(args.output / "counterexamples.json", counterexamples)

    plots = maybe_make_euler_stage_plots(args.output / "plots", samples, attention)
    summary["plots"] = plots
    save_json(args.output / "summary.json", summary)
    write_euler_stage_report(args.output / "report.md", summary, attention, args)

    cm = summary["condition_means"]
    pc = summary["paired_comparisons"]
    early_ci = pc["early_on_vs_all_masked"]["paired_bootstrap_95ci"]["ade_gain_m"]
    late_ci = pc["late_on_vs_all_masked"]["paired_bootstrap_95ci"]["ade_gain_m"]
    evl_ci = pc["early_on_vs_late_on"]["paired_bootstrap_95ci"]["ade_gain_m"]

    print("\n" + "=" * 132)
    print("DONE")
    print("=" * 132)
    print("ALL_MASKED ADE             :", fmt(cm["all_masked"]["ade_m"]), "m")
    print("ALL_ON ADE                 :", fmt(cm["all_on"]["ade_m"]), "m")
    print("EARLY_ON ADE               :", fmt(cm["early_on"]["ade_m"]), "m")
    print("LATE_ON ADE                :", fmt(cm["late_on"]["ade_m"]), "m")
    print("EARLY benefit vs MASKED    :", fmt(pc["early_on_vs_all_masked"]["ade_gain_m"]), "m")
    print("EARLY 95% CI               :", f"[{fmt(early_ci['lo95'])}, {fmt(early_ci['hi95'])}] m")
    print("LATE benefit vs MASKED     :", fmt(pc["late_on_vs_all_masked"]["ade_gain_m"]), "m")
    print("LATE 95% CI                :", f"[{fmt(late_ci['lo95'])}, {fmt(late_ci['hi95'])}] m")
    print("EARLY vs LATE benefit      :", fmt(pc["early_on_vs_late_on"]["ade_gain_m"]), "m")
    print("EARLY-vs-LATE 95% CI       :", f"[{fmt(evl_ci['lo95'])}, {fmt(evl_ci['hi95'])}] m")
    print("Euler schedules            :", schedules)
    print("Report                     :", args.output / "report.md")
    print("Summary                    :", args.output / "summary.json")
    print("Attention diagnostics      :", args.output / "attention_diagnostics.json")
    print("Samples                    :", args.output / "samples.jsonl")
    print("Counterexamples            :", args.output / "counterexamples.json")
    print("attention_vectors.pt       : NOT GENERATED")
    if plots:
        print("Plots                      :", args.output / "plots")

if __name__ == "__main__":
    main()
