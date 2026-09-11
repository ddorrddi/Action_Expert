#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired causal analysis for RAW-KV Flow/DiT Action Experts.

This script is intentionally matched to train_rawkv_flow_dit_fixedsplit.py.
It evaluates five controlled conditions on the same fixed-split samples and
the exact same per-sample Gaussian x0:

    DIRECT       : direct_flow_dit + prompt RAW K/V
    ALL_ON       : reasoning_flow_dit + prompt and reasoning RAW K/V
    ALL_MASKED   : reasoning_flow_dit + reasoning RAW K/V always masked
    EARLY_ON     : reasoning RAW K/V visible in the first half of Euler steps
    LATE_ON      : reasoning RAW K/V visible in the second half of Euler steps

The primary causal comparison is ALL_ON versus ALL_MASKED because both use the
same reasoning checkpoint. DIRECT versus ALL_ON is also reported, but it mixes
checkpoint and conditioning differences and must not be interpreted as a pure
KV intervention.

Outputs
-------
    report.txt                    human-readable final report
    report.md                     same results in Markdown
    summary.json                  aggregate metrics and paired statistics
    samples.jsonl                 one complete record per sample
    samples.csv                   flat sample-level table
    samples.txt                   compact sample-level ADE/gain table
    counterexamples.json          largest gains/failures
    attention_diagnostics.json    layer x Euler-step attention diagnostics
    attention_layer_step.txt      compact aggregate attention table
    attention_vectors.pt          ALL_ON reasoning-token attention vectors
    predictions.npz               predictions, GT, IDs, and x0
    plots/*.png                   causal and attention plots

No VLM K/V projection, concatenation into a feature vector, or re-projection
is performed. K and V remain separate [B,Hkv,T,D] tensors.
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
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_PROJECT_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_PROJECT_ROOT / "scripts"
for _path in (SCRIPT_DIR, ACTION_PROJECT_ROOT, ACTION_SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scripts.action_model_flow_dit import (
        TrajectoryNormalizer,
        per_sample_metrics_np,
        trajectory_metrics_np,
    )
except ImportError:
    from action_model_flow_dit import (  # type: ignore
        TrajectoryNormalizer,
        per_sample_metrics_np,
        trajectory_metrics_np,
    )

try:
    from train_rawkv_flow_dit_fixedsplit import build_flow_dit_rawkv
except ImportError as exc:
    raise ImportError(
        "train_rawkv_flow_dit_fixedsplit.py를 이 파일과 같은 디렉터리 또는 "
        "/home/lhh/lab/Action_Expert/scripts에 두어야 합니다."
    ) from exc


DEFAULT_CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/action_kv_cache"
)
DEFAULT_MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit"
)
DEFAULT_RESULT_ROOT = Path(
    "/home/lhh/lab/E2E/Result/rawkv_dit_causality"
)

CONDITIONS = ("DIRECT", "ALL_ON", "ALL_MASKED", "EARLY_ON", "LATE_ON", "TOP25_MASKED", "TOP50_MASKED", "TOP75_MASKED",)
REASONING_CONDITIONS = CONDITIONS[1:]
METRICS = ("ade_m", "fde_m", "heading_mae_rad")
METRIC_LABELS = {
    "ade_m": "ADE (m)",
    "fde_m": "FDE (m)",
    "heading_mae_rad": "Heading MAE (rad)",
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def make_x0(base_seed: int, sample_id: str, num_steps: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_sample_seed(base_seed, sample_id))
    return torch.randn((1, num_steps, 3), generator=generator, dtype=torch.float32)


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    out = [float(x) for x in values if finite_float(x) is not None]
    return float(np.mean(out)) if out else None


def percentile_or_none(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if len(values) else None


# =============================================================================
# CACHE / CHECKPOINT VALIDATION
# =============================================================================

@dataclass
class CacheSample:
    sample_id: str
    clip: str
    direct_key: torch.Tensor
    direct_value: torch.Tensor
    reasoning_key: torch.Tensor
    reasoning_value: torch.Tensor
    trajectory: torch.Tensor
    reasoning_text: str
    reasoning_token_ids: List[int]
    direct_tokens: int
    reasoning_tokens: int
    cache_file: str


def resolve_cache_file(item: Mapping[str, Any], manifest_path: Path) -> Path:
    raw = Path(str(item.get("cache_file", "")))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(manifest_path.parent / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cache file not found: {raw}")


def validate_raw_kv(name: str, tensor: torch.Tensor, sample_id: str) -> None:
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)
        raise RuntimeError(f"{name} must be [H,T,D], id={sample_id}, got={shape}")
    if tensor.shape[1] <= 0:
        raise RuntimeError(f"{name} has zero tokens, id={sample_id}")
    if not tensor.is_floating_point():
        raise RuntimeError(f"{name} must be floating point, id={sample_id}")
    if not torch.isfinite(tensor.float()).all():
        raise RuntimeError(f"{name} contains NaN/Inf, id={sample_id}")


def load_cache_sample(item: Mapping[str, Any], manifest_path: Path) -> CacheSample:
    sample_id = str(item.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Manifest contains an empty sample id")
    cache_file = resolve_cache_file(item, manifest_path)
    cache = torch.load(cache_file, map_location="cpu", weights_only=False)

    required = (
        "direct_key", "direct_value", "reasoning_delta_key",
        "reasoning_delta_value", "trajectory",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"Missing cache fields id={sample_id}: {missing}")

    dk = cache["direct_key"].detach().cpu().contiguous()
    dv = cache["direct_value"].detach().cpu().contiguous()
    rk = cache["reasoning_delta_key"].detach().cpu().contiguous()
    rv = cache["reasoning_delta_value"].detach().cpu().contiguous()
    for name, tensor in (("direct_key", dk), ("direct_value", dv),
                         ("reasoning_delta_key", rk), ("reasoning_delta_value", rv)):
        validate_raw_kv(name, tensor, sample_id)
    if dk.shape != dv.shape or rk.shape != rv.shape:
        raise RuntimeError(f"K/V shape mismatch, id={sample_id}")
    if dk.shape[0] != rk.shape[0] or dk.shape[2] != rk.shape[2]:
        raise RuntimeError(f"Direct/reasoning KV geometry mismatch, id={sample_id}")

    gt = torch.as_tensor(cache["trajectory"], dtype=torch.float32).cpu().contiguous()
    if tuple(gt.shape) != (10, 3):
        raise RuntimeError(f"trajectory must be [10,3], id={sample_id}, got={tuple(gt.shape)}")
    if not torch.isfinite(gt).all():
        raise RuntimeError(f"trajectory contains NaN/Inf, id={sample_id}")

    cache_id = str(cache.get("id", sample_id))
    if cache_id != sample_id:
        raise RuntimeError(f"Manifest/cache id mismatch: {sample_id} != {cache_id}")

    reasoning_text = str(cache.get("reasoning_text", item.get("reasoning_text", ""))).strip()
    token_ids = [int(x) for x in cache.get("reasoning_token_ids", [])]
    return CacheSample(
        sample_id=sample_id,
        clip=str(cache.get("clip", item.get("clip", ""))),
        direct_key=dk,
        direct_value=dv,
        reasoning_key=rk,
        reasoning_value=rv,
        trajectory=gt,
        reasoning_text=reasoning_text,
        reasoning_token_ids=token_ids,
        direct_tokens=int(dk.shape[1]),
        reasoning_tokens=int(rk.shape[1]),
        cache_file=str(cache_file),
    )


def checkpoint_path(model_root: Path, branch: str) -> Path:
    return model_root / "flow" / f"{branch}_flow_dit" / "best.pt"


def load_checkpoint(model_root: Path, branch: str) -> Dict[str, Any]:
    path = checkpoint_path(model_root, branch)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "model_state_dict", "action_config", "normalizer",
        "vlm_num_attention_heads", "vlm_num_kv_heads", "vlm_head_dim",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint missing fields {path}: {missing}")
    stored_branch = str(checkpoint.get("branch", branch))
    if stored_branch != branch:
        raise RuntimeError(f"Checkpoint branch mismatch: expected={branch}, got={stored_branch}")
    checkpoint["_path"] = str(path)
    return checkpoint


def normalizer_arrays(checkpoint: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    payload = checkpoint["normalizer"]
    return (
        np.asarray(payload["mean"], dtype=np.float64),
        np.asarray(payload["std"], dtype=np.float64),
    )


def assert_checkpoint_compatibility(
    direct_ckpt: Mapping[str, Any],
    reasoning_ckpt: Mapping[str, Any],
) -> None:
    geometry_keys = ("vlm_num_attention_heads", "vlm_num_kv_heads", "vlm_head_dim")
    for key in geometry_keys:
        if int(direct_ckpt[key]) != int(reasoning_ckpt[key]):
            raise RuntimeError(f"Checkpoint geometry mismatch for {key}")

    architecture_keys = ("hidden_dim", "num_steps", "num_layers", "ff_dim")
    for key in architecture_keys:
        left = direct_ckpt["action_config"].get(key)
        right = reasoning_ckpt["action_config"].get(key)
        if left != right:
            raise RuntimeError(f"Checkpoint action_config mismatch for {key}: {left} != {right}")

    d_mean, d_std = normalizer_arrays(direct_ckpt)
    r_mean, r_std = normalizer_arrays(reasoning_ckpt)
    if d_mean.shape != r_mean.shape or d_std.shape != r_std.shape:
        raise RuntimeError("Direct/reasoning normalizer shape mismatch")
    if not np.array_equal(d_mean, r_mean) or not np.array_equal(d_std, r_std):
        raise RuntimeError(
            "Direct/reasoning checkpoint normalizers differ. Same x0 no longer yields a "
            "strictly controlled trajectory-space comparison."
        )

    d_signature = direct_ckpt.get("cache_signature")
    r_signature = reasoning_ckpt.get("cache_signature")
    if d_signature and r_signature and d_signature != r_signature:
        raise RuntimeError("Direct/reasoning checkpoints were trained from different KV caches")


def assert_cache_geometry(sample: CacheSample, checkpoint: Mapping[str, Any]) -> None:
    expected = (
        int(checkpoint["vlm_num_kv_heads"]),
        int(checkpoint["vlm_head_dim"]),
    )
    actual = (int(sample.direct_key.shape[0]), int(sample.direct_key.shape[2]))
    if actual != expected:
        raise RuntimeError(
            f"Cache/checkpoint geometry mismatch id={sample.sample_id}: "
            f"cache(Hkv,D)={actual}, checkpoint={expected}"
        )


def build_model(checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint["action_config"]
    model = build_flow_dit_rawkv(
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        vlm_num_attention_heads=int(checkpoint["vlm_num_attention_heads"]),
        vlm_num_kv_heads=int(checkpoint["vlm_num_kv_heads"]),
        vlm_head_dim=int(checkpoint["vlm_head_dim"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state_dict load failed: {incompatible}")
    model.eval()
    return model


# =============================================================================
# ATTENTION DIAGNOSTICS
# =============================================================================

class AttentionDiagnosticRecorder:
    """Recomputes the exact pre-dropout attention probabilities from each layer."""

    def __init__(self, model: torch.nn.Module, solver_steps: int):
        self.model = model
        self.solver_steps = int(solver_steps)
        self.num_layers = len(model.layers)
        self.current_sample_id = ""
        self.current_condition = ""
        self.current_step = -1
        self.direct_len = 0
        self.reasoning_len = 0
        self.rows: List[Dict[str, Any]] = []
        self.vectors: Dict[str, torch.Tensor] = {}
        self._sample_vectors: List[Tuple[int, int, torch.Tensor]] = []
        self._handles = []

        for layer_index, layer in enumerate(model.layers):
            handle = layer.attn.register_forward_pre_hook(
                self._make_hook(layer_index), with_kwargs=False
            )
            self._handles.append(handle)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_condition(
        self,
        sample_id: str,
        condition: str,
        direct_len: int,
        reasoning_len: int,
    ) -> None:
        self.current_sample_id = sample_id
        self.current_condition = condition
        self.current_step = -1
        self.direct_len = int(direct_len)
        self.reasoning_len = int(reasoning_len)
        self._sample_vectors = []

    def begin_step(self, step: int) -> None:
        self.current_step = int(step)

    def end_condition(self) -> None:
        if self.current_condition == "ALL_ON" and self.reasoning_len > 0:
            expected = self.solver_steps * self.num_layers
            if len(self._sample_vectors) != expected:
                raise RuntimeError(
                    f"Attention vector count mismatch id={self.current_sample_id}: "
                    f"{len(self._sample_vectors)} != {expected}"
                )
            ordered = sorted(self._sample_vectors, key=lambda x: (x[0], x[1]))
            vector = torch.stack([x[2] for x in ordered], dim=0)
            vector = vector.view(
                self.solver_steps, self.num_layers, self.reasoning_len
            )
            self.vectors[self.current_sample_id] = vector

    def _make_hook(self, layer_index: int):
        def hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            if len(inputs) != 4:
                raise RuntimeError("Unexpected RawKV attention input signature")
            query_tokens, vlm_key, vlm_value, memory_mask = inputs
            self._record(
                module=module,
                layer_index=layer_index,
                query_tokens=query_tokens,
                vlm_key=vlm_key,
                vlm_value=vlm_value,
                memory_mask=memory_mask,
            )
        return hook

    @torch.no_grad()
    def _record(
        self,
        module: torch.nn.Module,
        layer_index: int,
        query_tokens: torch.Tensor,
        vlm_key: torch.Tensor,
        vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> None:
        if self.current_step < 0:
            raise RuntimeError("Recorder step was not initialized")
        b, n, _ = query_tokens.shape
        if b != 1:
            raise RuntimeError("Attention diagnostics currently require batch size 1")

        hq = int(module.vlm_num_attention_heads)
        hkv = int(module.vlm_num_kv_heads)
        d = int(module.vlm_head_dim)
        q = module.q_proj(query_tokens).view(b, n, hq, d).transpose(1, 2)
        self_k = module.self_k_proj(query_tokens).view(b, n, hkv, d).transpose(1, 2)
        raw_k = vlm_key.to(device=q.device, dtype=q.dtype)
        raw_v = vlm_value.to(device=q.device, dtype=q.dtype)
        k = torch.cat([raw_k, self_k], dim=2)
        v_prefix = raw_v
        if int(module.kv_groups) > 1:
            k = k.repeat_interleave(int(module.kv_groups), dim=1)
            v_prefix = v_prefix.repeat_interleave(int(module.kv_groups), dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(module.scale)
        prefix_len = int(raw_k.shape[2])
        invalid_prefix = (~memory_mask.bool())[:, None, None, :].expand(b, 1, n, prefix_len)
        if bool(module.causal_queries):
            blocked_self = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=scores.device),
                diagonal=1,
            )[None, None, :, :].expand(b, 1, n, n)
        else:
            blocked_self = torch.zeros(
                (b, 1, n, n), dtype=torch.bool, device=scores.device
            )
        blocked = torch.cat([invalid_prefix, blocked_self], dim=-1)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)

        prefix_attn = attn[..., :prefix_len]
        direct_attn = prefix_attn[..., :self.direct_len]
        reason_start = self.direct_len
        reason_end = self.direct_len + self.reasoning_len
        reason_attn = prefix_attn[..., reason_start:reason_end]

        vlm_mass_per_query = prefix_attn.sum(dim=-1)
        direct_mass_per_query = direct_attn.sum(dim=-1)
        reasoning_mass_per_query = reason_attn.sum(dim=-1)
        vlm_mass = float(vlm_mass_per_query.mean().item())
        direct_mass = float(direct_mass_per_query.mean().item())
        reasoning_mass = float(reasoning_mass_per_query.mean().item())

        eps = 1e-12
        if self.reasoning_len > 0:
            share_per_query = reasoning_mass_per_query / vlm_mass_per_query.clamp_min(eps)
            reasoning_share = float(share_per_query.mean().item())
            token_fraction = self.reasoning_len / float(prefix_len)
            enrichment = reasoning_share / token_fraction

            valid = reasoning_mass_per_query > eps
            if bool(valid.any()):
                p = reason_attn / reasoning_mass_per_query.unsqueeze(-1).clamp_min(eps)
                p_safe = p.clamp_min(eps)
                entropy_per_query = -(p * p_safe.log()).sum(dim=-1)
                entropy = float(entropy_per_query[valid].mean().item())
                normalized_entropy = (
                    entropy / math.log(self.reasoning_len)
                    if self.reasoning_len > 1 else 0.0
                )
                effective_fraction = float(
                    (entropy_per_query[valid].exp() / self.reasoning_len).mean().item()
                )
                top1_share = float(p.max(dim=-1).values[valid].mean().item())

                reason_values = v_prefix[..., reason_start:reason_end, :].float()
                attended = torch.einsum("bhqr,bhrd->bhqd", p.float(), reason_values)
                uniform = reason_values.mean(dim=-2, keepdim=True).expand_as(attended)
                cosine = F.cosine_similarity(attended, uniform, dim=-1, eps=1e-8)
                attended_context_cosine = float(cosine[valid].mean().item())
            else:
                p = None
                entropy = None
                normalized_entropy = None
                effective_fraction = None
                top1_share = None
                attended_context_cosine = None
        else:
            reasoning_share = 0.0
            token_fraction = 0.0
            enrichment = None
            p = None
            entropy = None
            normalized_entropy = None
            effective_fraction = None
            top1_share = None
            attended_context_cosine = None

        row = {
            "sample_id": self.current_sample_id,
            "condition": self.current_condition,
            "euler_step": self.current_step,
            "layer": int(layer_index),
            "direct_tokens": self.direct_len,
            "reasoning_tokens": self.reasoning_len,
            "vlm_attention_mass": vlm_mass,
            "direct_attention_mass": direct_mass,
            "reasoning_attention_mass": reasoning_mass,
            "reasoning_attention_share": reasoning_share,
            "reasoning_token_fraction": token_fraction,
            "reasoning_attention_enrichment": enrichment,
            "reasoning_attention_entropy": entropy,
            "reasoning_attention_entropy_normalized": normalized_entropy,
            "effective_reasoning_token_fraction": effective_fraction,
            "top1_reasoning_token_share": top1_share,
            "attended_context_cosine": attended_context_cosine,
        }
        self.rows.append(row)

        if self.current_condition == "ALL_ON" and p is not None:
            vector = p.mean(dim=(0, 1, 2)).detach().float().cpu()
            self._sample_vectors.append((self.current_step, layer_index, vector))


def aggregate_attention_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_keys = (
        "vlm_attention_mass", "direct_attention_mass", "reasoning_attention_mass",
        "reasoning_attention_share", "reasoning_attention_enrichment",
        "reasoning_attention_entropy", "reasoning_attention_entropy_normalized",
        "effective_reasoning_token_fraction", "top1_reasoning_token_share",
        "attended_context_cosine",
    )
    groups: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), int(row["euler_step"]), int(row["layer"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (condition, step, layer), values in sorted(groups.items()):
        agg: Dict[str, Any] = {
            "condition": condition,
            "euler_step": step,
            "layer": layer,
            "samples": len(values),
        }
        for key in metric_keys:
            agg[key] = mean_or_none(row.get(key) for row in values)
        output.append(agg)
    return output


def aggregate_attention_per_sample(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = (
        "vlm_attention_mass", "reasoning_attention_mass", "reasoning_attention_share",
        "reasoning_attention_enrichment", "reasoning_attention_entropy_normalized",
        "effective_reasoning_token_fraction", "top1_reasoning_token_share",
        "attended_context_cosine",
    )
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "ALL_ON":
            groups[str(row["sample_id"])].append(row)
    result: Dict[str, Dict[str, float]] = {}
    for sample_id, values in groups.items():
        result[sample_id] = {
            key: mean_or_none(row.get(key) for row in values) for key in keys
        }
    return result


def verify_masked_attention(
    rows: Sequence[Mapping[str, Any]],
    solver_steps: int,
    tolerance: float = 1e-7,
) -> None:
    split_step = solver_steps // 2
    violations = []
    for row in rows:
        condition = str(row["condition"])
        step = int(row["euler_step"])
        should_be_masked = (
            condition == "ALL_MASKED"
            or (condition == "EARLY_ON" and step >= split_step)
            or (condition == "LATE_ON" and step < split_step)
        )
        if should_be_masked and float(row["reasoning_attention_mass"]) > tolerance:
            violations.append(row)
    if violations:
        first = violations[0]
        raise RuntimeError(
            "Masked reasoning attention is non-zero: "
            f"id={first['sample_id']} condition={first['condition']} "
            f"step={first['euler_step']} layer={first['layer']} "
            f"mass={first['reasoning_attention_mass']}"
        )


# =============================================================================
# CONTROLLED EULER SAMPLING
# =============================================================================

def reasoning_visible(condition: str, step: int, solver_steps: int) -> bool:
    split_step = solver_steps // 2
    if condition == "ALL_ON":
        return True
    if condition == "ALL_MASKED":
        return False
    if condition == "EARLY_ON":
        return step < split_step
    if condition == "LATE_ON":
        return step >= split_step
    raise ValueError(f"Unknown reasoning condition: {condition}")

def compute_reasoning_token_importance(attention_vector: torch.Tensor) -> torch.Tensor:
    """
    attention_vector: [Euler, Layer, Reasoning_Token]
    return: [Reasoning_Token]
    """
    # 모든 Euler step과 Layer에 대해 평균을 내어 토큰별 최종 중요도 산출
    return attention_vector.mean(dim=(0, 1))

def make_memory_mask(
    direct_len: int,
    reasoning_len: int,
    visible: bool,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((1, direct_len + reasoning_len), dtype=torch.bool, device=device)
    if not visible:
        mask[:, direct_len:] = False
    if not bool(mask[:, :direct_len].all()):
        raise RuntimeError("Direct prompt tokens must always remain visible")
    return mask

def make_token_mask(
    direct_len: int,
    reasoning_len: int,
    importance: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((1, direct_len + reasoning_len), dtype=torch.bool, device=device)
    
    if reasoning_len == 0:
        return mask

    # 비율에 맞춰 마스킹할 토큰 개수 산출 (최소 1개, 최대 reasoning_len 개)
    num_mask = max(1, int(round(reasoning_len * mask_ratio)))
    num_mask = min(num_mask, reasoning_len)

    if num_mask > 0:
        # 상위 중요도를 가진 토큰의 인덱스 추출
        top_indices = torch.topk(importance, k=num_mask, largest=True).indices
        # PyTorch 벡터 인덱싱을 활용하여 한 번에 False 처리
        mask[0, direct_len + top_indices] = False

    return mask

@torch.inference_mode()
def euler_sample_from_x0(
    model: torch.nn.Module,
    vlm_key: torch.Tensor,
    vlm_value: torch.Tensor,
    direct_len: int,
    reasoning_len: int,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    x0_cpu: torch.Tensor,
    condition: str,
    recorder: Optional[AttentionDiagnosticRecorder] = None,
    token_importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if condition not in REASONING_CONDITIONS:
        raise ValueError(condition)
    device = vlm_key.device
    x = x0_cpu.to(device=device, dtype=torch.float32).clone()
    if tuple(x.shape) != (1, int(model.num_steps), 3):
        raise ValueError(f"Bad x0 shape: {tuple(x.shape)}")
    dt = 1.0 / float(solver_steps)

    if recorder is not None:
        recorder.begin_condition(
            sample_id=recorder.current_sample_id,
            condition=condition,
            direct_len=direct_len,
            reasoning_len=reasoning_len,
        )

    for step in range(solver_steps):
        if condition in ("TOP25_MASKED", "TOP50_MASKED", "TOP75_MASKED"):
            ratio = {
                "TOP25_MASKED": 0.25,
                "TOP50_MASKED": 0.50,
                "TOP75_MASKED": 0.75,
            }[condition]
            
            if token_importance is None:
                raise ValueError(f"{condition} 조건에서는 token_importance 텐서가 반드시 필요합니다.")
            
            importance_device = token_importance.to(device)
            memory_mask = make_token_mask(
                direct_len,
                reasoning_len,
                importance_device,
                ratio,
                device
            )
        else:
            visible = reasoning_visible(condition, step, solver_steps)
            memory_mask = make_memory_mask(
                direct_len,
                reasoning_len,
                visible,
                device
            )
        t = torch.full(
            (1, 1, 1), float(step) / float(solver_steps),
            device=device, dtype=torch.float32,
        )
        if recorder is not None:
            recorder.begin_step(step)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
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

    if recorder is not None:
        recorder.end_condition()
    return normalizer.to(device).denormalize(x)


@torch.inference_mode()
def euler_sample_direct_from_x0(
    model: torch.nn.Module,
    vlm_key: torch.Tensor,
    vlm_value: torch.Tensor,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    x0_cpu: torch.Tensor,
) -> torch.Tensor:
    device = vlm_key.device
    x = x0_cpu.to(device=device, dtype=torch.float32).clone()
    mask = torch.ones((1, vlm_key.shape[2]), dtype=torch.bool, device=device)
    dt = 1.0 / float(solver_steps)
    for step in range(solver_steps):
        t = torch.full(
            (1, 1, 1), float(step) / float(solver_steps),
            device=device, dtype=torch.float32,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            velocity = model(
                x_t=x, t=t, vlm_key=vlm_key, vlm_value=vlm_value,
                memory_mask=mask,
            )
        x = x + dt * velocity.float()
    return normalizer.to(device).denormalize(x)


def to_device_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        key.unsqueeze(0).to(device=device, dtype=torch.bfloat16, non_blocking=True),
        value.unsqueeze(0).to(device=device, dtype=torch.bfloat16, non_blocking=True),
    )


# =============================================================================
# METRICS / STATISTICS
# =============================================================================

def paired_bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_bootstrap_ci requires a non-empty 1-D array")
    chunk = min(1000, repeats)
    means = []
    remaining = repeats
    while remaining > 0:
        count = min(chunk, remaining)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = abs(float(values.mean()))
    if not np.any(values):
        return 1.0
    extreme = 0
    processed = 0
    while processed < repeats:
        count = min(1000, repeats - processed)
        signs = rng.integers(0, 2, size=(count, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        processed += count
    return float((extreme + 1) / (repeats + 1))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(x: Sequence[float], y: Sequence[float], method: str) -> Dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return {"n": int(x_arr.size), "r": None}
    if method == "spearman":
        x_arr = rankdata_average(x_arr)
        y_arr = rankdata_average(y_arr)
    elif method != "pearson":
        raise ValueError(method)
    return {"n": int(x_arr.size), "r": float(np.corrcoef(x_arr, y_arr)[0, 1])}


def comparison_stats(
    left_metrics: Mapping[str, np.ndarray],
    right_metrics: Mapping[str, np.ndarray],
    name: str,
    seed: int,
    bootstrap_repeats: int,
    permutation_repeats: int,
) -> Dict[str, Any]:
    """Gain is right error - left error, so positive means LEFT is better."""
    output: Dict[str, Any] = {
        "comparison": name,
        "gain_definition": "right_error_minus_left_error; positive means left condition is better",
        "metrics": {},
    }
    for index, metric in enumerate(METRICS):
        left = np.asarray(left_metrics[metric], dtype=np.float64)
        right = np.asarray(right_metrics[metric], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"Paired shape mismatch for {name}/{metric}")
        gain = right - left
        rng_boot = np.random.default_rng(seed + index * 100003 + 11)
        rng_perm = np.random.default_rng(seed + index * 100003 + 29)
        ci_low, ci_high = paired_bootstrap_ci(gain, rng_boot, bootstrap_repeats)
        std = float(np.std(gain, ddof=1)) if gain.size > 1 else 0.0
        output["metrics"][metric] = {
            "n": int(gain.size),
            "left_mean": float(left.mean()),
            "right_mean": float(right.mean()),
            "mean_gain": float(gain.mean()),
            "median_gain": float(np.median(gain)),
            "p25_gain": percentile_or_none(gain, 25),
            "p50_gain": percentile_or_none(gain, 50),
            "p75_gain": percentile_or_none(gain, 75),
            "p90_gain": percentile_or_none(gain, 90),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "win_rate": float(np.mean(gain > 0)),
            "tie_rate": float(np.mean(np.isclose(gain, 0.0, atol=1e-12))),
            "cohens_dz": float(gain.mean() / std) if std > 0 else None,
            "sign_flip_p_two_sided": paired_sign_flip_pvalue(
                gain, rng_perm, permutation_repeats
            ),
        }
    return output


def metric_arrays(predictions: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    result = per_sample_metrics_np(predictions, gt)
    return {key: np.asarray(result[key], dtype=np.float64) for key in METRICS}


def trajectory_intent(trajectory: np.ndarray, threshold_m: float) -> Dict[str, str]:
    final_x = float(trajectory[-1, 0])
    final_y = float(trajectory[-1, 1])
    if final_x > threshold_m:
        longitudinal = "forward"
    elif final_x < -threshold_m:
        longitudinal = "backward"
    else:
        longitudinal = "stationary"
    if final_y > threshold_m:
        lateral = "left"
    elif final_y < -threshold_m:
        lateral = "right"
    else:
        lateral = "straight"
    return {"longitudinal": longitudinal, "lateral": lateral}


# =============================================================================
# OUTPUTS
# =============================================================================

def write_samples_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def flatten_sample_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "sample_id": row["sample_id"],
        "clip": row["clip"],
        "direct_tokens": row["direct_tokens"],
        "reasoning_tokens": row["reasoning_tokens"],
        "reasoning_text": row["reasoning_text"],
    }
    for condition, metrics in row["conditions"].items():
        for metric, value in metrics.items():
            flat[f"{condition.lower()}_{metric}"] = value
    for name, gains in row["gains"].items():
        for metric, value in gains.items():
            flat[f"{name}_{metric}_gain"] = value
    for key, value in row.get("attention_all_on", {}).items():
        flat[f"attention_{key}"] = value
    for prefix, intent in (("gt", row["gt_intent"]), ("all_on", row["all_on_intent"])):
        flat[f"{prefix}_longitudinal"] = intent["longitudinal"]
        flat[f"{prefix}_lateral"] = intent["lateral"]
    return flat


def write_samples_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows = [flatten_sample_for_csv(row) for row in rows]
    fieldnames: List[str] = []
    seen = set()
    for row in flat_rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_samples_txt(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    header = (
        f"{'index':>5} {'sample_id':<32} {'dT':>5} {'rT':>5} "
        f"{'DIRECT':>10} {'ALL_ON':>10} {'MASKED':>10} "
        f"{'EARLY':>10} {'LATE':>10} {'GAIN':>10}"
    )
    lines = [
        "Per-sample ADE and causal gain",
        "GAIN = ALL_MASKED ADE - ALL_ON ADE; positive means reasoning KV helps",
        header,
        "-" * len(header),
    ]
    for index, row in enumerate(rows, 1):
        conditions = row["conditions"]
        gain = row["gains"]["all_on_vs_all_masked"]["ade_m"]
        lines.append(
            f"{index:5d} {str(row['sample_id'])[:32]:<32} "
            f"{int(row['direct_tokens']):5d} {int(row['reasoning_tokens']):5d} "
            f"{conditions['DIRECT']['ade_m']:10.4f} "
            f"{conditions['ALL_ON']['ade_m']:10.4f} "
            f"{conditions['ALL_MASKED']['ade_m']:10.4f} "
            f"{conditions['EARLY_ON']['ade_m']:10.4f} "
            f"{conditions['LATE_ON']['ade_m']:10.4f} {gain:10.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attention_txt(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    header = (
        f"{'condition':<12} {'step':>4} {'layer':>5} {'vlm_mass':>10} "
        f"{'reason_mass':>11} {'share':>9} {'enrich':>9} "
        f"{'eff_frac':>9} {'top1':>9} {'ctx_cos':>9}"
    )
    lines = [
        "Aggregate RAW-KV attention diagnostics",
        "Each row is averaged across all evaluated samples, trajectory queries, and heads.",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{str(row['condition']):<12} {int(row['euler_step']):4d} {int(row['layer']):5d} "
            f"{f4(row.get('vlm_attention_mass')):>10} "
            f"{f4(row.get('reasoning_attention_mass')):>11} "
            f"{f4(row.get('reasoning_attention_share')):>9} "
            f"{f4(row.get('reasoning_attention_enrichment')):>9} "
            f"{f4(row.get('effective_reasoning_token_fraction')):>9} "
            f"{f4(row.get('top1_reasoning_token_share')):>9} "
            f"{f4(row.get('attended_context_cosine')):>9}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_counterexamples(
    sample_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    key = "all_on_vs_all_masked"
    ranked = sorted(
        sample_rows,
        key=lambda row: float(row["gains"][key]["ade_m"]),
    )
    failures = ranked[:top_k]
    improvements = list(reversed(ranked[-top_k:]))

    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "sample_id": row["sample_id"],
            "clip": row["clip"],
            "reasoning_text": row["reasoning_text"],
            "direct_tokens": row["direct_tokens"],
            "reasoning_tokens": row["reasoning_tokens"],
            "conditions": row["conditions"],
            "reasoning_ae_gain_ade_m": row["gains"][key]["ade_m"],
            "reasoning_ae_gain_fde_m": row["gains"][key]["fde_m"],
            "reasoning_ae_gain_heading_rad": row["gains"][key]["heading_mae_rad"],
            "gt_intent": row["gt_intent"],
            "all_on_intent": row["all_on_intent"],
            "all_masked_intent": row["all_masked_intent"],
            "attention_all_on": row.get("attention_all_on", {}),
        }

    return {
        "gain_definition": "ALL_MASKED error - ALL_ON error; positive means reasoning KV helps",
        "largest_failures": [compact(row) for row in failures],
        "largest_improvements": [compact(row) for row in improvements],
    }


def create_plots(
    plot_dir: Path,
    predictions: Mapping[str, np.ndarray],
    gt: np.ndarray,
    sample_metrics: Mapping[str, Mapping[str, np.ndarray]],
    attention_aggregate: Sequence[Mapping[str, Any]],
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    on = sample_metrics["ALL_ON"]["ade_m"]
    masked = sample_metrics["ALL_MASKED"]["ade_m"]
    limit = float(max(on.max(), masked.max()) * 1.03 + 1e-6)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(masked, on, s=22, alpha=0.75)
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(xlabel="ALL_MASKED ADE (m)", ylabel="ALL_ON ADE (m)",
           title="Paired causal ADE: reasoning KV ON vs MASKED", xlim=(0, limit), ylim=(0, limit))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "paired_ade_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path))

    gain = masked - on
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(gain, bins=min(30, max(8, int(math.sqrt(len(gain)) * 2))), alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(float(gain.mean()), color="tab:red", linewidth=1.5, label=f"mean={gain.mean():.3f} m")
    ax.set(xlabel="ADE gain = MASKED - ON (m)", ylabel="Samples",
           title="Per-sample causal ADE gain")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = plot_dir / "ade_gain_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path))

    on_rows = [row for row in attention_aggregate if row["condition"] == "ALL_ON"]
    if on_rows:
        steps = sorted({int(row["euler_step"]) for row in on_rows})
        step_mass = [
            np.mean([row["reasoning_attention_mass"] for row in on_rows if int(row["euler_step"]) == step])
            for step in steps
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(steps, step_mass, marker="o")
        ax.set(xlabel="Euler step", ylabel="Reasoning attention mass",
               title="Reasoning attention across denoising steps", xticks=steps)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "attention_by_euler_step.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(str(path))

        layers = sorted({int(row["layer"]) for row in on_rows})
        matrix = np.full((len(layers), len(steps)), np.nan, dtype=np.float64)
        for row in on_rows:
            matrix[layers.index(int(row["layer"])), steps.index(int(row["euler_step"]))] = float(
                row["reasoning_attention_share"]
            )
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set(xlabel="Euler step", ylabel="Layer", title="Reasoning share within VLM-prefix attention")
        ax.set_xticks(range(len(steps)), labels=steps)
        ax.set_yticks(range(len(layers)), labels=layers)
        fig.colorbar(image, ax=ax, label="Reasoning attention share")
        fig.tight_layout()
        path = plot_dir / "attention_layer_step_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(str(path))

    return created


def f4(value: Any) -> str:
    number = finite_float(value)
    return "N/A" if number is None else f"{number:.4f}"


def pct(value: Any) -> str:
    number = finite_float(value)
    return "N/A" if number is None else f"{number * 100:.2f}%"


def build_text_report(summary: Mapping[str, Any], counterexamples: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.extend([
        "RAW-KV Flow/DiT Reasoning Causality Analysis",
        "=" * 78,
        f"Split: {summary['run']['split']}",
        f"Samples: {summary['run']['samples']}",
        f"Seed / Euler steps: {summary['run']['seed']} / {summary['run']['solver_steps']}",
        f"Direct checkpoint epoch: {summary['checkpoints']['direct']['epoch']}",
        f"Reasoning checkpoint epoch: {summary['checkpoints']['reasoning']['epoch']}",
        "",
        "해석 기준",
        "- 핵심 인과 비교: ALL_ON vs ALL_MASKED (동일 reasoning checkpoint)",
        "- gain = 비교 오른쪽 조건의 error - 왼쪽 조건의 error",
        "- gain > 0: 왼쪽 조건이 더 우수함",
        "- DIRECT vs ALL_ON은 checkpoint 차이까지 포함하므로 순수 KV 효과가 아님",
        "- EARLY_ON: Euler 전반부만 reasoning KV 사용",
        "- LATE_ON: Euler 후반부만 reasoning KV 사용",
        "",
        "1. 조건별 전체 성능",
        "-" * 78,
        f"{'Condition':<14} {'ADE(m)':>12} {'FDE(m)':>12} {'Heading(rad)':>15} {'ms/sample':>13}",
    ])
    for condition in CONDITIONS:
        metrics = summary["condition_metrics"][condition]
        latency = summary["latency_ms"][condition]
        lines.append(
            f"{condition:<14} {metrics['ade_m']:>12.4f} {metrics['fde_m']:>12.4f} "
            f"{metrics['heading_mae_rad']:>15.4f} {latency:>13.2f}"
        )

    lines.extend(["", "2. Paired causal statistics", "-" * 78])
    for name, comparison in summary["comparisons"].items():
        lines.append(f"[{name}]")
        for metric in METRICS:
            stat = comparison["metrics"][metric]
            lines.append(
                f"  {METRIC_LABELS[metric]}: mean gain={f4(stat['mean_gain'])}, "
                f"95% CI=[{f4(stat['ci95_low'])}, {f4(stat['ci95_high'])}], "
                f"win={pct(stat['win_rate'])}, p={f4(stat['sign_flip_p_two_sided'])}, "
                f"P25/P50/P75/P90={f4(stat['p25_gain'])}/{f4(stat['p50_gain'])}/"
                f"{f4(stat['p75_gain'])}/{f4(stat['p90_gain'])}"
            )
        lines.append("")

    causal_ade = summary["comparisons"]["all_on_vs_all_masked"]["metrics"]["ade_m"]
    ci_excludes_zero = causal_ade["ci95_low"] > 0 or causal_ade["ci95_high"] < 0
    direction = "개선" if causal_ade["mean_gain"] > 0 else "악화"
    lines.extend([
        "3. 핵심 결론",
        "-" * 78,
        f"- Reasoning KV ON은 MASKED 대비 평균 ADE를 {abs(causal_ade['mean_gain']):.4f} m {direction}.",
        f"- 95% paired-bootstrap CI는 [{causal_ade['ci95_low']:.4f}, {causal_ade['ci95_high']:.4f}] m.",
        f"- 95% CI의 0 제외 여부: {'YES' if ci_excludes_zero else 'NO'}.",
        f"- 샘플 단위 ADE 승률: {causal_ade['win_rate'] * 100:.2f}%.",
        "",
        "4. ALL_ON attention diagnostics (전체 layer/step 평균)",
        "-" * 78,
    ])
    attn = summary.get("attention_all_on_global", {})
    for key in (
        "vlm_attention_mass", "reasoning_attention_mass", "reasoning_attention_share",
        "reasoning_attention_enrichment", "reasoning_attention_entropy_normalized",
        "effective_reasoning_token_fraction", "top1_reasoning_token_share",
        "attended_context_cosine",
    ):
        lines.append(f"- {key}: {f4(attn.get(key))}")

    lines.extend(["", "5. Attention ↔ causal ADE gain correlation", "-" * 78])
    for key, methods in summary.get("attention_gain_correlations", {}).items():
        lines.append(
            f"- {key}: Pearson r={f4(methods['pearson']['r'])}, "
            f"Spearman r={f4(methods['spearman']['r'])}, n={methods['pearson']['n']}"
        )

    lines.extend(["", "6. 가장 큰 반례 (Reasoning KV가 ADE를 악화)", "-" * 78])
    for row in counterexamples["largest_failures"]:
        lines.append(
            f"- id={row['sample_id']} | gain={row['reasoning_ae_gain_ade_m']:.4f} m | "
            f"ON={row['conditions']['ALL_ON']['ade_m']:.4f} | "
            f"MASKED={row['conditions']['ALL_MASKED']['ade_m']:.4f} | "
            f"reasoning={row['reasoning_text']}"
        )

    lines.extend(["", "7. 가장 큰 개선 사례", "-" * 78])
    for row in counterexamples["largest_improvements"]:
        lines.append(
            f"- id={row['sample_id']} | gain={row['reasoning_ae_gain_ade_m']:.4f} m | "
            f"ON={row['conditions']['ALL_ON']['ade_m']:.4f} | "
            f"MASKED={row['conditions']['ALL_MASKED']['ade_m']:.4f} | "
            f"reasoning={row['reasoning_text']}"
        )

    lines.extend([
        "",
        "8. 주의사항",
        "-" * 78,
        "- attention은 정보의 존재/참조 패턴이며, 인과 효과는 ON↔MASKED trajectory 차이로 판단.",
        "- intent는 ego frame의 x=longitudinal, y=left-positive 가정으로 파생.",
        "- attention diagnostics 활성화 시 ms/sample에는 진단용 attention 재계산 비용이 포함됨.",
        "",
    ])
    return "\n".join(lines)


def build_markdown_report(text_report: str) -> str:
    lines = text_report.splitlines()
    output = ["# RAW-KV Flow/DiT Reasoning Causality Analysis", ""]
    section_numbers = tuple(f"{i}. " for i in range(1, 9))
    for line in lines[2:]:
        if line.startswith(section_numbers):
            output.extend([f"## {line}", ""])
        elif set(line) == {"-"}:
            continue
        elif line.startswith("[") and line.endswith("]"):
            output.extend([f"### {line[1:-1]}", ""])
        else:
            output.append(line)
    return "\n".join(output) + "\n"


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Result directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def evaluate_direct(
    manifest: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    seed: int,
    solver_steps: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    model = build_model(checkpoint, device)
    normalizer = TrajectoryNormalizer.from_dict(checkpoint["normalizer"]).to(device)
    predictions, gts, ids, latencies = [], [], [], []
    start = time.perf_counter()
    for index, item in enumerate(manifest, 1):
        sample = load_cache_sample(item, manifest_path)
        assert_cache_geometry(sample, checkpoint)
        x0 = make_x0(seed, sample.sample_id, int(model.num_steps))
        key, value = to_device_kv(sample.direct_key, sample.direct_value, device)
        sync_cuda(device)
        t0 = time.perf_counter()
        pred = euler_sample_direct_from_x0(
            model, key, value, normalizer, solver_steps, x0
        )
        sync_cuda(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        predictions.append(pred[0].float().cpu().numpy())
        gts.append(sample.trajectory.numpy())
        ids.append(sample.sample_id)
        if index == 1 or index % 10 == 0 or index == len(manifest):
            elapsed = time.perf_counter() - start
            eta = elapsed / index * (len(manifest) - index)
            print(
                f"[DIRECT] {index:04d}/{len(manifest):04d} | "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )
        del key, value, pred
    del model, normalizer
    cleanup_cuda()
    return np.stack(predictions), np.stack(gts), ids, latencies


def evaluate_reasoning(
    manifest: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    seed: int,
    solver_steps: int,
    attention_diagnostics: bool,
) -> Tuple[
    Dict[str, np.ndarray], List[CacheSample], Dict[str, List[float]],
    List[Dict[str, Any]], Dict[str, torch.Tensor],
]:
    model = build_model(checkpoint, device)
    normalizer = TrajectoryNormalizer.from_dict(checkpoint["normalizer"]).to(device)
    recorder = AttentionDiagnosticRecorder(model, solver_steps) if attention_diagnostics else None
    predictions: Dict[str, List[np.ndarray]] = {condition: [] for condition in REASONING_CONDITIONS}
    latencies: Dict[str, List[float]] = {condition: [] for condition in REASONING_CONDITIONS}
    samples: List[CacheSample] = []
    start = time.perf_counter()

    try:
        for index, item in enumerate(manifest, 1):
            sample = load_cache_sample(item, manifest_path)
            assert_cache_geometry(sample, checkpoint)
            x0 = make_x0(seed, sample.sample_id, int(model.num_steps))
            full_key = torch.cat([sample.direct_key, sample.reasoning_key], dim=1)
            full_value = torch.cat([sample.direct_value, sample.reasoning_value], dim=1)
            key, value = to_device_kv(full_key, full_value, device)

            for condition in REASONING_CONDITIONS:
                if recorder is not None:
                    recorder.current_sample_id = sample.sample_id
                
                # 핵심: ALL_ON 조건이 선행된 후 TOP_MASKED 조건이 실행되므로, 루프 내부에서 중요도를 계산
                token_importance = None
                if condition in ("TOP25_MASKED", "TOP50_MASKED", "TOP75_MASKED"):
                    if recorder is None or sample.sample_id not in recorder.vectors:
                        raise RuntimeError(f"{condition}을 실행하려면 ALL_ON Attention 진단 결과가 선행되어야 합니다.")
                    attn_vec = recorder.vectors[sample.sample_id]
                    token_importance = compute_reasoning_token_importance(attn_vec)

                sync_cuda(device)
                t0 = time.perf_counter()
                pred = euler_sample_from_x0(
                    model=model,
                    vlm_key=key,
                    vlm_value=value,
                    direct_len=sample.direct_tokens,
                    reasoning_len=sample.reasoning_tokens,
                    normalizer=normalizer,
                    solver_steps=solver_steps,
                    x0_cpu=x0,
                    condition=condition,
                    recorder=recorder,
                    token_importance=token_importance, # 계산된 중요도 전달
                )
                sync_cuda(device)
                latencies[condition].append((time.perf_counter() - t0) * 1000.0)
                predictions[condition].append(pred[0].float().cpu().numpy())
                del pred

            samples.append(sample)
            if index == 1 or index % 10 == 0 or index == len(manifest):
                elapsed = time.perf_counter() - start
                eta = elapsed / index * (len(manifest) - index)
                print(
                    f"[REASONING x4] {index:04d}/{len(manifest):04d} | "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                    flush=True,
                )
            del key, value, full_key, full_value
    finally:
        if recorder is not None:
            recorder.close()

    attention_rows = recorder.rows if recorder is not None else []
    attention_vectors = recorder.vectors if recorder is not None else {}
    del model, normalizer, recorder
    cleanup_cuda()
    return (
        {condition: np.stack(values) for condition, values in predictions.items()},
        samples,
        latencies,
        attention_rows,
        attention_vectors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired causal RAW-KV analysis for direct/reasoning Flow-DiT Action Experts"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260841)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-attention-diagnostics", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=20000)
    parser.add_argument("--counterexample-top-k", type=int, default=10)
    parser.add_argument("--intent-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.solver_steps < 2 or args.solver_steps % 2 != 0:
        raise ValueError("--solver-steps must be a positive even integer >= 2")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.bootstrap_repeats < 1000 or args.permutation_repeats < 1000:
        raise ValueError("bootstrap/permutation repeats must each be >= 1000")
    if args.counterexample_top_k < 1:
        raise ValueError("--counterexample-top-k must be >= 1")
    if args.intent_threshold_m < 0:
        raise ValueError("--intent-threshold-m must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")

    manifest_path = args.cache_root / args.split / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    ids = [str(row.get("id", "")) for row in manifest]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Manifest contains duplicate sample IDs")
    if args.limit is not None:
        manifest = manifest[:args.limit]

    output_dir = args.result_root / args.split
    prepare_output_dir(output_dir, args.overwrite)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    direct_ckpt = load_checkpoint(args.model_root, "direct")
    reasoning_ckpt = load_checkpoint(args.model_root, "reasoning")
    assert_checkpoint_compatibility(direct_ckpt, reasoning_ckpt)
    first_sample = load_cache_sample(manifest[0], manifest_path)
    assert_cache_geometry(first_sample, direct_ckpt)
    expected_steps = int(direct_ckpt["action_config"]["num_steps"])
    if expected_steps != 10:
        raise RuntimeError(f"Expected 10 trajectory waypoints, checkpoint has {expected_steps}")

    print("=" * 100)
    print("RAW-KV FLOW/DiT CAUSAL ANALYSIS")
    print("=" * 100)
    print("GPU              :", torch.cuda.get_device_name(args.gpu_id))
    print("Split / samples  :", args.split, "/", len(manifest))
    print("Euler steps      :", args.solver_steps)
    print("Early / Late     :", f"0-{args.solver_steps//2-1} / {args.solver_steps//2}-{args.solver_steps-1}")
    print("Attention diag   :", not args.no_attention_diagnostics)
    print("Output           :", output_dir)
    print("Primary contrast : ALL_ON vs ALL_MASKED (same checkpoint and x0)")
    print("=" * 100)

    total_start = time.perf_counter()
    direct_pred, gt, direct_ids, direct_latency = evaluate_direct(
        manifest, manifest_path, direct_ckpt, device,
        args.seed, args.solver_steps,
    )
    reasoning_pred, samples, reasoning_latency, attention_rows, attention_vectors = evaluate_reasoning(
        manifest, manifest_path, reasoning_ckpt, device,
        args.seed, args.solver_steps, not args.no_attention_diagnostics,
    )
    sample_ids = [sample.sample_id for sample in samples]
    if direct_ids != sample_ids:
        raise RuntimeError("Direct/reasoning sample order mismatch")
    reasoning_gt = np.stack([sample.trajectory.numpy() for sample in samples])
    if not np.array_equal(gt, reasoning_gt):
        raise RuntimeError("Direct/reasoning GT mismatch")

    predictions: Dict[str, np.ndarray] = {"DIRECT": direct_pred, **reasoning_pred}
    for condition, array in predictions.items():
        if array.shape != gt.shape or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid predictions for {condition}: {array.shape}")

    condition_metrics = {
        condition: trajectory_metrics_np(pred, gt)
        for condition, pred in predictions.items()
    }
    per_condition = {
        condition: metric_arrays(pred, gt)
        for condition, pred in predictions.items()
    }
    latency_ms = {
        "DIRECT": float(np.mean(direct_latency)),
        **{condition: float(np.mean(values)) for condition, values in reasoning_latency.items()},
    }

    comparisons_spec = {
        "all_on_vs_all_masked": ("ALL_ON", "ALL_MASKED"),
        "early_on_vs_all_masked": ("EARLY_ON", "ALL_MASKED"),
        "late_on_vs_all_masked": ("LATE_ON", "ALL_MASKED"),
        "early_on_vs_late_on": ("EARLY_ON", "LATE_ON"),
        "all_on_vs_direct": ("ALL_ON", "DIRECT"),
        # 새로 추가되는 TOP 마스킹 비교군
        "all_on_vs_top25": ("ALL_ON", "TOP25_MASKED"),
        "all_on_vs_top50": ("ALL_ON", "TOP50_MASKED"),
        "all_on_vs_top75": ("ALL_ON", "TOP75_MASKED"),
    }
    comparisons = {}
    for index, (name, (left, right)) in enumerate(comparisons_spec.items()):
        comparisons[name] = comparison_stats(
            per_condition[left], per_condition[right], name,
            seed=args.seed + index * 1000003,
            bootstrap_repeats=args.bootstrap_repeats,
            permutation_repeats=args.permutation_repeats,
        )

    if attention_rows:
        verify_masked_attention(attention_rows, args.solver_steps)
    attention_aggregate = aggregate_attention_rows(attention_rows)
    attention_per_sample = aggregate_attention_per_sample(attention_rows)
    attention_keys = (
        "vlm_attention_mass", "reasoning_attention_mass", "reasoning_attention_share",
        "reasoning_attention_enrichment", "reasoning_attention_entropy_normalized",
        "effective_reasoning_token_fraction", "top1_reasoning_token_share",
        "attended_context_cosine",
    )
    all_on_attention_rows = [row for row in attention_rows if row["condition"] == "ALL_ON"]
    attention_global = {
        key: mean_or_none(row.get(key) for row in all_on_attention_rows)
        for key in attention_keys
    }

    causal_ade_gain = (
        per_condition["ALL_MASKED"]["ade_m"] - per_condition["ALL_ON"]["ade_m"]
    )
    attention_gain_correlations: Dict[str, Any] = {}
    for key in attention_keys:
        x, y = [], []
        for index, sample_id in enumerate(sample_ids):
            value = attention_per_sample.get(sample_id, {}).get(key)
            if finite_float(value) is not None:
                x.append(float(value))
                y.append(float(causal_ade_gain[index]))
        attention_gain_correlations[key] = {
            "pearson": correlation(x, y, "pearson"),
            "spearman": correlation(x, y, "spearman"),
        }

    sample_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        conditions = {
            condition: {
                metric: float(per_condition[condition][metric][index])
                for metric in METRICS
            }
            for condition in CONDITIONS
        }
        gains = {}
        for name, (left, right) in comparisons_spec.items():
            gains[name] = {
                metric: float(
                    per_condition[right][metric][index] - per_condition[left][metric][index]
                )
                for metric in METRICS
            }
        sample_rows.append({
            "sample_id": sample.sample_id,
            "clip": sample.clip,
            "cache_file": sample.cache_file,
            "direct_tokens": sample.direct_tokens,
            "reasoning_tokens": sample.reasoning_tokens,
            "reasoning_token_ids": sample.reasoning_token_ids,
            "reasoning_text": sample.reasoning_text,
            "conditions": conditions,
            "gains": gains,
            "gt_final_xyz": gt[index, -1].tolist(),
            "gt_intent": trajectory_intent(gt[index], args.intent_threshold_m),
            "all_on_intent": trajectory_intent(predictions["ALL_ON"][index], args.intent_threshold_m),
            "all_masked_intent": trajectory_intent(predictions["ALL_MASKED"][index], args.intent_threshold_m),
            "attention_all_on": attention_per_sample.get(sample.sample_id, {}),
        })

    counterexamples = build_counterexamples(
        sample_rows, min(args.counterexample_top_k, len(sample_rows))
    )
    elapsed = time.perf_counter() - total_start
    summary = {
        "run": {
            "split": args.split,
            "samples": len(samples),
            "seed": args.seed,
            "solver_steps": args.solver_steps,
            "early_steps": list(range(0, args.solver_steps // 2)),
            "late_steps": list(range(args.solver_steps // 2, args.solver_steps)),
            "same_x0_per_sample_across_all_conditions": True,
            "attention_diagnostics": not args.no_attention_diagnostics,
            "elapsed_seconds": elapsed,
            "gpu": torch.cuda.get_device_name(args.gpu_id),
        },
        "paths": {
            "manifest": str(manifest_path),
            "model_root": str(args.model_root),
            "result_dir": str(output_dir),
        },
        "checkpoints": {
            "direct": {
                "path": direct_ckpt["_path"],
                "epoch": int(direct_ckpt.get("epoch", -1)),
                "branch": direct_ckpt.get("branch", "direct"),
                "val_metrics": direct_ckpt.get("val_metrics", {}),
            },
            "reasoning": {
                "path": reasoning_ckpt["_path"],
                "epoch": int(reasoning_ckpt.get("epoch", -1)),
                "branch": reasoning_ckpt.get("branch", "reasoning"),
                "val_metrics": reasoning_ckpt.get("val_metrics", {}),
            },
            "action_config": reasoning_ckpt["action_config"],
            "vlm_geometry": {
                "num_attention_heads": int(reasoning_ckpt["vlm_num_attention_heads"]),
                "num_kv_heads": int(reasoning_ckpt["vlm_num_kv_heads"]),
                "head_dim": int(reasoning_ckpt["vlm_head_dim"]),
            },
        },
        "condition_definitions": {
            "DIRECT": "direct checkpoint + direct prompt RAW K/V",
            "ALL_ON": "reasoning checkpoint + prompt/reasoning RAW K/V at all Euler steps",
            "ALL_MASKED": "reasoning checkpoint + reasoning positions masked at all Euler steps",
            "EARLY_ON": "reasoning positions visible only in the first half of Euler steps",
            "LATE_ON": "reasoning positions visible only in the second half of Euler steps",
        },
        "condition_metrics": condition_metrics,
        "latency_ms": latency_ms,
        "comparisons": comparisons,
        "attention_all_on_global": attention_global,
        "attention_gain_correlations": attention_gain_correlations,
    }

    write_json(output_dir / "summary.json", json_safe(summary))
    write_samples_jsonl(output_dir / "samples.jsonl", sample_rows)
    write_samples_csv(output_dir / "samples.csv", sample_rows)
    write_samples_txt(output_dir / "samples.txt", sample_rows)
    write_json(output_dir / "counterexamples.json", json_safe(counterexamples))
    write_json(
        output_dir / "attention_diagnostics.json",
        json_safe({
            "definitions": {
                "vlm_attention_mass": "total attention probability assigned to all VLM prefix tokens",
                "reasoning_attention_mass": "total attention probability assigned to reasoning prefix tokens",
                "reasoning_attention_share": "reasoning mass divided by total VLM-prefix mass",
                "reasoning_attention_enrichment": "reasoning attention share divided by reasoning token fraction",
                "reasoning_attention_entropy_normalized": "entropy within reasoning tokens divided by log(R)",
                "effective_reasoning_token_fraction": "exp(entropy) / number of reasoning tokens",
                "top1_reasoning_token_share": "largest conditional share among reasoning tokens",
                "attended_context_cosine": "cosine between attention-weighted and uniformly averaged reasoning V context",
            },
            "masked_attention_hard_check": "PASS" if attention_rows else "SKIPPED",
            "global_all_on": attention_global,
            "per_sample_all_on": attention_per_sample,
            "aggregate_by_condition_step_layer": attention_aggregate,
            "raw_rows": attention_rows,
        }),
    )
    torch.save(
        {
            "sample_ids": sample_ids,
            "description": "ALL_ON conditional attention over reasoning tokens; [Euler, layer, R] per sample",
            "vectors": attention_vectors,
        },
        output_dir / "attention_vectors.pt",
    )
    write_attention_txt(output_dir / "attention_layer_step.txt", attention_aggregate)
    x0_array = np.concatenate(
        [make_x0(args.seed, sample_id, expected_steps).numpy() for sample_id in sample_ids],
        axis=0,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_ids=np.asarray(sample_ids),
        gt=gt,
        x0=x0_array,
        **{condition.lower(): pred for condition, pred in predictions.items()},
    )

    plot_paths = create_plots(
        output_dir / "plots", predictions, gt, per_condition, attention_aggregate
    )
    summary["paths"]["plots"] = plot_paths
    write_json(output_dir / "summary.json", json_safe(summary))

    report = build_text_report(summary, counterexamples)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "report.md").write_text(build_markdown_report(report), encoding="utf-8")

    print("\n" + report)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
