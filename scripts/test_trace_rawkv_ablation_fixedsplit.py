#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the controlled 2x2 RAW-KV ablation using the EXISTING two Action Experts.

Action Experts
--------------
DIRECT checkpoint:
/home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/direct_flow_dit/best.pt

REUSE checkpoint (trained with prompt + generated reasoning RAW-KV):
/home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt

2x2 matrix
----------
                    trajectory task                    reasoning task
DIRECT model   trajectory prompt K/V only         reasoning prompt K/V only
REUSE model    trajectory prompt + trajectory     reasoning prompt + reasoning
               generated-trace K/V                generated-trace K/V

Fairness controls
-----------------
- same test IDs and GT
- same VLM backbone was used to extract all caches
- same images / mission / speed / acceleration / heading within each sample
- same RAW last-layer K/V representation
- same Action Expert architecture family
- exact same per-sample Gaussian x0 across ALL FOUR cells
- same Euler solver-step count unless explicitly overridden

Interpretation
--------------
The cleanest within-checkpoint contrast for the semantic question is:
    reuse_reasoning vs reuse_trajectory
because the Action Expert weights are identical in that comparison.

direct_reasoning vs direct_trajectory quantifies how much the prompt/task
instruction alone changes prompt-boundary VLM K/V before any generated trace is
reused.

NOTE: direct vs reuse uses two separately trained Action Expert checkpoints, so
that row-wise contrast contains both conditioning-memory and trained-weight effects.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================================
# DEFAULT PATHS
# =============================================================================

CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit"
)
DIRECT_CKPT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/direct_flow_dit/best.pt"
)
REUSE_CKPT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt"
)
RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/results/trace_rawkv_ablation_fixedsplit"
)

NUM_STEPS = 10
TEST_NOISE_SEED = 20260841
CACHE_DTYPE = torch.bfloat16

CONDITIONS = (
    "direct_trajectory",
    "direct_reasoning",
    "reuse_trajectory",
    "reuse_reasoning",
)


# =============================================================================
# METRICS
# =============================================================================

def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_metrics_np(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Bad metric shapes pred={pred.shape} gt={gt.shape}")
    xy_error = np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1)
    heading = np.abs(wrap_angle_np(pred[..., 2] - gt[..., 2]))
    return {
        "ade_m": float(xy_error.mean()),
        "fde_m": float(xy_error[:, -1].mean()),
        "heading_mae_rad": float(heading.mean()),
    }


def per_sample_metrics_np(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    xy_error = np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1)
    heading = np.abs(wrap_angle_np(pred[..., 2] - gt[..., 2]))
    return {
        "ade_m": xy_error.mean(axis=1),
        "fde_m": xy_error[:, -1],
        "heading_mae_rad": heading.mean(axis=1),
    }


# =============================================================================
# NORMALIZER - same schema as training helper
# =============================================================================

@dataclass
class TrajectoryNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)
        if self.mean.shape != self.std.shape:
            raise ValueError("Normalizer mean/std shape mismatch")
        if self.mean.ndim == 1:
            if tuple(self.mean.shape) != (3,):
                raise ValueError(f"Global normalizer must be [3], got {self.mean.shape}")
        elif self.mean.ndim == 2:
            if self.mean.shape[-1] != 3:
                raise ValueError(f"Per-waypoint normalizer must be [N,3], got {self.mean.shape}")
        else:
            raise ValueError(f"Normalizer stats must be [3] or [N,3], got {self.mean.shape}")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Non-finite normalizer")
        if torch.any(self.std <= 0):
            raise ValueError("Normalizer std must be >0")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
        )

    def to(self, device: torch.device | str) -> "TrajectoryNormalizer":
        return TrajectoryNormalizer(
            mean=self.mean.to(device=device, dtype=torch.float32),
            std=self.std.to(device=device, dtype=torch.float32),
        )

    @property
    def mode(self) -> str:
        return "global_xyz" if self.mean.ndim == 1 else "per_waypoint_xyz"

    def _stats_for(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if trajectory.ndim not in (2, 3) or trajectory.shape[-1] != 3:
            raise ValueError(f"trajectory must be [N,3] or [B,N,3], got {trajectory.shape}")

        if self.mean.ndim == 1:
            if trajectory.ndim == 2:
                return self.mean.view(1, 3), self.std.view(1, 3)
            return self.mean.view(1, 1, 3), self.std.view(1, 1, 3)

        n = int(trajectory.shape[-2])
        if int(self.mean.shape[0]) != n:
            raise ValueError(
                f"Normalizer waypoint mismatch stats={self.mean.shape[0]} traj={n}"
            )
        if trajectory.ndim == 2:
            return self.mean, self.std
        return self.mean.unsqueeze(0), self.std.unsqueeze(0)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        return (
            trajectory
            - mean.to(device=trajectory.device, dtype=trajectory.dtype)
        ) / std.to(device=trajectory.device, dtype=trajectory.dtype)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        return (
            trajectory * std.to(device=trajectory.device, dtype=trajectory.dtype)
            + mean.to(device=trajectory.device, dtype=trajectory.dtype)
        )


# =============================================================================
# RAW-KV FLOW/DiT ARCHITECTURE - exact structure from training script
# =============================================================================

class RawKVPrefixFusionAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vlm_num_attention_heads: int,
        vlm_num_kv_heads: int,
        vlm_head_dim: int,
        dropout: float,
        causal_queries: bool,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
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

    def forward(
        self,
        query_tokens: torch.Tensor,
        vlm_key: torch.Tensor,
        vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.ndim != 3:
            raise ValueError("query_tokens must be [B,N,H]")
        if vlm_key.ndim != 4 or vlm_value.ndim != 4 or vlm_key.shape != vlm_value.shape:
            raise ValueError("raw VLM K/V must be equal-shape [B,H,T,D]")
        if int(vlm_key.shape[1]) != self.vlm_num_kv_heads:
            raise ValueError("VLM KV head count mismatch")
        if int(vlm_key.shape[-1]) != self.vlm_head_dim:
            raise ValueError("VLM head_dim mismatch")
        if memory_mask.shape != (vlm_key.shape[0], vlm_key.shape[2]):
            raise ValueError("memory_mask/raw-KV shape mismatch")

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
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

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

        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.q_attn_dim)
        return self.out_proj(out)


class TrajectoryOutputHeadRawKV(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


class RawKVEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vlm_num_attention_heads: int,
        vlm_num_kv_heads: int,
        vlm_head_dim: int,
        ff_dim: int,
        dropout: float,
        causal_queries: bool,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = RawKVPrefixFusionAttention(
            hidden_dim=hidden_dim,
            vlm_num_attention_heads=vlm_num_attention_heads,
            vlm_num_kv_heads=vlm_num_kv_heads,
            vlm_head_dim=vlm_head_dim,
            dropout=dropout,
            causal_queries=causal_queries,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        vlm_key: torch.Tensor,
        vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_attn(tokens)
        tokens = tokens + self.dropout(self.attn(x, vlm_key, vlm_value, memory_mask))
        tokens = tokens + self.dropout(self.ffn(self.norm_ff(tokens)))
        return tokens


class FourierEncoderRawKV(nn.Module):
    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("Fourier dim must be even and >=2")
        half = dim // 2
        freqs = torch.logspace(
            0.0, math.log10(float(max_freq)), steps=half, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x.float().unsqueeze(-1) * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class FlowActionInputProjectionRawKV(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        time_fourier_dim: int = 20,
        time_mlp_hidden: int = 512,
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

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != self.action_dim:
            raise ValueError(f"x_t must be [B,N,{self.action_dim}]")
        b = x_t.shape[0]
        action_hidden = self.action_proj(x_t.to(dtype=self.action_proj[0].weight.dtype))
        if t.shape[0] != b:
            raise ValueError("t batch mismatch")
        t_scalar = t.reshape(b, -1)[:, 0]
        time_feat = self.time_encoder(t_scalar)
        time_hidden = self.time_proj(
            time_feat.to(dtype=self.time_proj[0].weight.dtype)
        ).unsqueeze(1)
        return action_hidden + time_hidden


class RawKVFlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_steps: int,
        num_layers: int,
        vlm_num_attention_heads: int,
        vlm_num_kv_heads: int,
        vlm_head_dim: int,
        ff_dim: int,
        dropout: float,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.action_in = FlowActionInputProjectionRawKV(3, hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )
        self.layers = nn.ModuleList([
            RawKVEncoderBlock(
                hidden_dim=hidden_dim,
                vlm_num_attention_heads=vlm_num_attention_heads,
                vlm_num_kv_heads=vlm_num_kv_heads,
                vlm_head_dim=vlm_head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                causal_queries=False,
            )
            for _ in range(num_layers)
        ])
        self.output = TrajectoryOutputHeadRawKV(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        vlm_key: torch.Tensor,
        vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or tuple(x_t.shape[1:]) != (self.num_steps, 3):
            raise ValueError(
                f"x_t must be [B,{self.num_steps},3], got={tuple(x_t.shape)}"
            )
        tokens = self.action_in(x_t, t)
        tokens = tokens + self.horizon_embedding.to(
            device=tokens.device, dtype=tokens.dtype
        )
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                tokens = checkpoint(
                    layer, tokens, vlm_key, vlm_value, memory_mask,
                    use_reentrant=False,
                )
            else:
                tokens = layer(tokens, vlm_key, vlm_value, memory_mask)
        return self.output(tokens)


def build_flow_dit_rawkv(
    hidden_dim: int,
    num_steps: int,
    num_layers: int,
    vlm_num_attention_heads: int,
    vlm_num_kv_heads: int,
    vlm_head_dim: int,
    ff_dim: int,
    dropout: float,
    gradient_checkpointing: bool,
) -> nn.Module:
    return RawKVFlowMatchingActionExpert(
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        vlm_num_attention_heads=vlm_num_attention_heads,
        vlm_num_kv_heads=vlm_num_kv_heads,
        vlm_head_dim=vlm_head_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        gradient_checkpointing=gradient_checkpointing,
    )


# =============================================================================
# IO HELPERS
# =============================================================================

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSONL parse error {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((value + int(base_seed)) % (2**63 - 1))


def make_noise(sample_id: str, base_seed: int, num_steps: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_sample_seed(sample_id, base_seed))
    return torch.randn((1, num_steps, 3), generator=g, dtype=torch.float32)


# =============================================================================
# CACHE ACCESS
# =============================================================================

def condition_spec(condition: str) -> Tuple[str, bool, str]:
    """Return (task, reuse_generated_trace, checkpoint_role)."""
    if condition == "direct_trajectory":
        return "trajectory", False, "direct"
    if condition == "direct_reasoning":
        return "reasoning", False, "direct"
    if condition == "reuse_trajectory":
        return "trajectory", True, "reuse"
    if condition == "reuse_reasoning":
        return "reasoning", True, "reuse"
    raise ValueError(condition)


def load_condition_memory(
    cache: Dict[str, Any],
    condition: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task, reuse, _ = condition_spec(condition)
    key = cache[f"{task}_direct_key"]
    value = cache[f"{task}_direct_value"]

    if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
        raise RuntimeError(f"Bad direct raw K/V for {condition}")

    if reuse:
        delta_key = cache[f"{task}_delta_key"]
        delta_value = cache[f"{task}_delta_value"]
        if (
            delta_key.ndim != 3
            or delta_value.ndim != 3
            or delta_key.shape != delta_value.shape
        ):
            raise RuntimeError(f"Bad delta raw K/V for {condition}")
        if key.shape[0] != delta_key.shape[0] or key.shape[2] != delta_key.shape[2]:
            raise RuntimeError(f"Direct/delta K/V geometry mismatch for {condition}")
        key = torch.cat([key, delta_key], dim=1)
        value = torch.cat([value, delta_value], dim=1)

    mask = torch.ones((1, int(key.shape[1])), dtype=torch.bool)
    return key.unsqueeze(0), value.unsqueeze(0), mask


def load_manifest(cache_root: Path, limit: int | None) -> List[Dict[str, Any]]:
    manifest_path = cache_root / "test" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = read_jsonl(manifest_path)
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("Empty cache manifest")

    seen = set()
    for row in rows:
        sid = str(row["id"])
        if sid in seen:
            raise RuntimeError(f"Duplicate cache ID: {sid}")
        seen.add(sid)
        p = Path(row["cache_file"])
        if not p.is_file():
            raise FileNotFoundError(p)
    return rows


# =============================================================================
# CHECKPOINT LOADING / VALIDATION
# =============================================================================

def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "model_state_dict",
        "action_config",
        "normalizer",
        "vlm_num_attention_heads",
        "vlm_num_kv_heads",
        "vlm_head_dim",
    )
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key}: {path}")
    return ckpt


def build_model_from_checkpoint(
    ckpt: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, TrajectoryNormalizer, int]:
    cfg = ckpt["action_config"]
    model = build_flow_dit_rawkv(
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        vlm_num_attention_heads=int(ckpt["vlm_num_attention_heads"]),
        vlm_num_kv_heads=int(ckpt["vlm_num_kv_heads"]),
        vlm_head_dim=int(ckpt["vlm_head_dim"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    normalizer = TrajectoryNormalizer.from_dict(ckpt["normalizer"]).to(device)
    solver_steps = int(cfg.get("solver_steps", 10))
    return model, normalizer, solver_steps


def validate_checkpoint_pair(
    direct_ckpt: Dict[str, Any],
    reuse_ckpt: Dict[str, Any],
) -> None:
    # Architecture/geometry must match; branch and learned weights may differ.
    keys = ("vlm_num_attention_heads", "vlm_num_kv_heads", "vlm_head_dim")
    for key in keys:
        if int(direct_ckpt[key]) != int(reuse_ckpt[key]):
            raise RuntimeError(f"Checkpoint geometry mismatch: {key}")

    dc = direct_ckpt["action_config"]
    rc = reuse_ckpt["action_config"]
    for key in ("hidden_dim", "num_steps", "num_layers", "ff_dim", "dropout", "solver_steps"):
        if dc.get(key) != rc.get(key):
            raise RuntimeError(
                f"Direct/reuse Action Expert config mismatch: {key}: "
                f"{dc.get(key)} vs {rc.get(key)}"
            )

    dn = TrajectoryNormalizer.from_dict(direct_ckpt["normalizer"])
    rn = TrajectoryNormalizer.from_dict(reuse_ckpt["normalizer"])
    if not torch.equal(dn.mean, rn.mean) or not torch.equal(dn.std, rn.std):
        max_diff = max(
            float((dn.mean - rn.mean).abs().max().item()),
            float((dn.std - rn.std).abs().max().item()),
        )
        raise RuntimeError(
            f"Direct/reuse normalizer mismatch; controlled comparison invalid. max_diff={max_diff}"
        )


# =============================================================================
# SAMPLING
# =============================================================================

@torch.inference_mode()
def euler_sample_rawkv_controlled(
    model: nn.Module,
    vlm_key: torch.Tensor,
    vlm_value: torch.Tensor,
    memory_mask: torch.Tensor,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    noise_cpu: torch.Tensor,
) -> torch.Tensor:
    if solver_steps < 1:
        raise ValueError("solver_steps must be >=1")
    device = vlm_key.device
    x = noise_cpu.to(device=device, dtype=torch.float32)
    if tuple(x.shape) != (1, int(model.num_steps), 3):
        raise ValueError(f"Bad initial noise shape: {x.shape}")

    times = torch.linspace(0.0, 1.0, solver_steps + 1, device=device)
    for i in range(solver_steps):
        t_now = float(times[i].item())
        dt = float((times[i + 1] - times[i]).item())
        t = torch.full((1, 1, 1), t_now, device=device, dtype=torch.float32)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
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
# CONDITION EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate_condition(
    condition: str,
    manifest: Sequence[Dict[str, Any]],
    checkpoint_path: Path,
    device: torch.device,
    noise_seed: int,
    solver_steps_override: int | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ckpt = load_checkpoint(checkpoint_path)
    model, normalizer, ckpt_solver_steps = build_model_from_checkpoint(ckpt, device)
    solver_steps = (
        int(solver_steps_override)
        if solver_steps_override is not None
        else int(ckpt_solver_steps)
    )

    expected_geom = (
        int(ckpt["vlm_num_attention_heads"]),
        int(ckpt["vlm_num_kv_heads"]),
        int(ckpt["vlm_head_dim"]),
    )

    all_pred: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []
    per_rows: List[Dict[str, Any]] = []

    total_infer_sec = 0.0
    start = time.perf_counter()

    for idx, item in enumerate(manifest, 1):
        cache = torch.load(item["cache_file"], map_location="cpu", weights_only=False)
        sid = str(cache["id"])
        if sid != str(item["id"]):
            raise RuntimeError(f"Manifest/cache ID mismatch: {item['id']} vs {sid}")

        cache_geom = (
            int(cache["vlm_num_attention_heads"]),
            int(cache["vlm_num_kv_heads"]),
            int(cache["vlm_head_dim"]),
        )
        if cache_geom != expected_geom:
            raise RuntimeError(
                f"Cache/checkpoint KV geometry mismatch id={sid}: "
                f"cache={cache_geom} ckpt={expected_geom}"
            )

        vlm_key, vlm_value, memory_mask = load_condition_memory(cache, condition)
        gt = torch.as_tensor(cache["trajectory_gt"], dtype=torch.float32)
        if tuple(gt.shape) != (NUM_STEPS, 3):
            raise RuntimeError(f"Bad GT shape id={sid}: {gt.shape}")

        noise = make_noise(sid, noise_seed, NUM_STEPS)

        vlm_key = vlm_key.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        vlm_value = vlm_value.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        memory_mask = memory_mask.to(device=device, non_blocking=True)

        sync_cuda(device)
        t0 = time.perf_counter()
        pred = euler_sample_rawkv_controlled(
            model=model,
            vlm_key=vlm_key,
            vlm_value=vlm_value,
            memory_mask=memory_mask,
            normalizer=normalizer,
            solver_steps=solver_steps,
            noise_cpu=noise,
        )
        sync_cuda(device)
        infer_sec = time.perf_counter() - t0
        total_infer_sec += infer_sec

        pred_np = pred[0].float().cpu().numpy()
        gt_np = gt.numpy()
        m = per_sample_metrics_np(pred_np[None, ...], gt_np[None, ...])

        per_rows.append({
            "id": sid,
            "condition": condition,
            "ade_m": float(m["ade_m"][0]),
            "fde_m": float(m["fde_m"][0]),
            "heading_mae_rad": float(m["heading_mae_rad"][0]),
            "infer_ms": float(infer_sec * 1000.0),
            "memory_tokens": int(vlm_key.shape[2]),
        })
        all_pred.append(pred_np)
        all_gt.append(gt_np)

        elapsed = time.perf_counter() - start
        eta = elapsed / idx * (len(manifest) - idx)
        if idx % 10 == 0 or idx == len(manifest):
            print(
                f"[{condition}] {idx:04d}/{len(manifest):04d} | "
                f"ADE={per_rows[-1]['ade_m']:.4f}m | "
                f"memT={per_rows[-1]['memory_tokens']} | "
                f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
                flush=True,
            )

        del cache, vlm_key, vlm_value, memory_mask, pred

    pred_np = np.stack(all_pred, axis=0)
    gt_np = np.stack(all_gt, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)
    summary = {
        "condition": condition,
        "checkpoint": str(checkpoint_path),
        "checkpoint_branch": str(ckpt.get("branch", "")),
        "samples": len(manifest),
        "solver_steps": solver_steps,
        "normalizer_mode": normalizer.mode,
        "ade_m": metrics["ade_m"],
        "fde_m": metrics["fde_m"],
        "heading_mae_rad": metrics["heading_mae_rad"],
        "mean_infer_ms": float(total_infer_sec / len(manifest) * 1000.0),
    }

    del model, ckpt, normalizer
    cleanup_cuda()
    return summary, per_rows


# =============================================================================
# COMPARISON
# =============================================================================

def metric_delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    """Return a - b. Negative error delta means A is better."""
    return {
        "ade_m": float(a["ade_m"] - b["ade_m"]),
        "fde_m": float(a["fde_m"] - b["fde_m"]),
        "heading_mae_rad": float(a["heading_mae_rad"] - b["heading_mae_rad"]),
    }


def pct_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return float("nan")
    return (new - old) / old * 100.0


def paired_win_stats(
    per_by_condition: Dict[str, List[Dict[str, Any]]],
    a: str,
    b: str,
) -> Dict[str, Any]:
    """Compare A vs B; lower error wins."""
    ra = per_by_condition[a]
    rb = per_by_condition[b]
    if [x["id"] for x in ra] != [x["id"] for x in rb]:
        raise RuntimeError(f"Paired ID order mismatch: {a} vs {b}")

    out: Dict[str, Any] = {"a": a, "b": b, "samples": len(ra)}
    for metric in ("ade_m", "fde_m", "heading_mae_rad"):
        va = np.asarray([x[metric] for x in ra], dtype=np.float64)
        vb = np.asarray([x[metric] for x in rb], dtype=np.float64)
        d = va - vb
        out[metric] = {
            "mean_delta_a_minus_b": float(d.mean()),
            "median_delta_a_minus_b": float(np.median(d)),
            "a_win_rate_pct": float((d < 0).mean() * 100.0),
            "tie_rate_pct": float((np.isclose(d, 0.0, atol=1e-12)).mean() * 100.0),
        }
    return out


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "condition", "ade_m", "fde_m", "heading_mae_rad", "infer_ms", "memory_tokens"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test controlled 2x2 trajectory/reasoning RAW-KV ablation.")
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--direct-ckpt", type=Path, default=DIRECT_CKPT)
    p.add_argument("--reuse-ckpt", type=Path, default=REUSE_CKPT)
    p.add_argument("--output", type=Path, default=RESULT_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=TEST_NOISE_SEED)
    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Debug only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_root = args.cache_root.expanduser().resolve()
    args.direct_ckpt = args.direct_ckpt.expanduser().resolve()
    args.reuse_ckpt = args.reuse_ckpt.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU is required")
    if args.output.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.cache_root, args.limit)
    direct_ckpt_obj = load_checkpoint(args.direct_ckpt)
    reuse_ckpt_obj = load_checkpoint(args.reuse_ckpt)
    validate_checkpoint_pair(direct_ckpt_obj, reuse_ckpt_obj)

    # Branch sanity check. Keep tolerant of absent/legacy labels, strict if present.
    if direct_ckpt_obj.get("branch") not in (None, "direct"):
        raise RuntimeError(f"Direct checkpoint branch is {direct_ckpt_obj.get('branch')}")
    if reuse_ckpt_obj.get("branch") not in (None, "reasoning"):
        raise RuntimeError(f"Reuse checkpoint branch is {reuse_ckpt_obj.get('branch')}")

    del direct_ckpt_obj, reuse_ckpt_obj

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 120)
    print("TRACE RAW-KV 2x2 ABLATION TEST")
    print("=" * 120)
    print("GPU          :", torch.cuda.get_device_name(args.gpu_id))
    print("Cache        :", args.cache_root)
    print("Direct ckpt  :", args.direct_ckpt)
    print("Reuse ckpt   :", args.reuse_ckpt)
    print("Samples      :", len(manifest))
    print("Noise seed   :", args.noise_seed)
    print("Solver steps :", args.solver_steps if args.solver_steps is not None else "checkpoint default")
    print("Noise control: exact same x0 per sample ID across all 4 cells")
    print()
    print("2x2 cells:")
    print("  direct_trajectory = DIRECT model + trajectory-prompt boundary K/V")
    print("  direct_reasoning  = DIRECT model + reasoning-prompt boundary K/V")
    print("  reuse_trajectory  = REUSE model  + trajectory prompt + generated trajectory K/V")
    print("  reuse_reasoning   = REUSE model  + reasoning prompt + generated reasoning K/V")

    summaries: Dict[str, Dict[str, Any]] = {}
    per_by_condition: Dict[str, List[Dict[str, Any]]] = {}
    all_per_rows: List[Dict[str, Any]] = []

    for condition in CONDITIONS:
        _, _, ckpt_role = condition_spec(condition)
        ckpt_path = args.direct_ckpt if ckpt_role == "direct" else args.reuse_ckpt
        print("\n" + "-" * 120)
        print("EVALUATE:", condition)
        print("-" * 120)
        summary, per_rows = evaluate_condition(
            condition=condition,
            manifest=manifest,
            checkpoint_path=ckpt_path,
            device=device,
            noise_seed=args.noise_seed,
            solver_steps_override=args.solver_steps,
        )
        summaries[condition] = summary
        per_by_condition[condition] = per_rows
        all_per_rows.extend(per_rows)
        print(
            f"[{condition}] ADE={summary['ade_m']:.4f}m "
            f"FDE={summary['fde_m']:.4f}m "
            f"Heading={summary['heading_mae_rad']:.4f}rad "
            f"Infer={summary['mean_infer_ms']:.2f}ms/sample"
        )

    # Primary comparisons.
    comparisons = {
        # Same DIRECT checkpoint; only task prompt changes.
        "direct_prompt_effect_reasoning_vs_trajectory": {
            "delta_reasoning_minus_trajectory": metric_delta(
                summaries["direct_reasoning"], summaries["direct_trajectory"]
            ),
            "ade_pct": pct_change(
                summaries["direct_reasoning"]["ade_m"],
                summaries["direct_trajectory"]["ade_m"],
            ),
            "paired": paired_win_stats(
                per_by_condition, "direct_reasoning", "direct_trajectory"
            ),
        },
        # Same REUSE checkpoint; core semantic-trace comparison.
        "reuse_trace_effect_reasoning_vs_trajectory": {
            "delta_reasoning_minus_trajectory": metric_delta(
                summaries["reuse_reasoning"], summaries["reuse_trajectory"]
            ),
            "ade_pct": pct_change(
                summaries["reuse_reasoning"]["ade_m"],
                summaries["reuse_trajectory"]["ade_m"],
            ),
            "paired": paired_win_stats(
                per_by_condition, "reuse_reasoning", "reuse_trajectory"
            ),
        },
        # These mix checkpoint weights + memory regime; report, but do not call pure cache effects.
        "trajectory_direct_vs_reuse_mixed_model_and_memory_effect": {
            "delta_reuse_minus_direct": metric_delta(
                summaries["reuse_trajectory"], summaries["direct_trajectory"]
            ),
            "ade_pct": pct_change(
                summaries["reuse_trajectory"]["ade_m"],
                summaries["direct_trajectory"]["ade_m"],
            ),
            "paired": paired_win_stats(
                per_by_condition, "reuse_trajectory", "direct_trajectory"
            ),
        },
        "reasoning_direct_vs_reuse_mixed_model_and_memory_effect": {
            "delta_reuse_minus_direct": metric_delta(
                summaries["reuse_reasoning"], summaries["direct_reasoning"]
            ),
            "ade_pct": pct_change(
                summaries["reuse_reasoning"]["ade_m"],
                summaries["direct_reasoning"]["ade_m"],
            ),
            "paired": paired_win_stats(
                per_by_condition, "reuse_reasoning", "direct_reasoning"
            ),
        },
    }

    payload = {
        "experiment": "fixedsplit_trace_rawkv_2x2_ablation",
        "cache_root": str(args.cache_root),
        "direct_checkpoint": str(args.direct_ckpt),
        "reuse_checkpoint": str(args.reuse_ckpt),
        "samples": len(manifest),
        "noise_seed": int(args.noise_seed),
        "solver_steps_override": args.solver_steps,
        "fairness": {
            "same_test_ids": True,
            "same_gt": True,
            "same_per_sample_gaussian_x0_all_cells": True,
            "same_input_state_between_trajectory_reasoning_cache_extraction": True,
            "same_vlm_backbone_for_cache_extraction": True,
        },
        "interpretation_note": (
            "reuse_reasoning vs reuse_trajectory is the main within-checkpoint trace-content contrast; "
            "direct_reasoning vs direct_trajectory measures task-prompt effect before trace reuse; "
            "direct vs reuse uses separately trained checkpoints and is therefore not a pure inference-time cache ablation."
        ),
        "results": summaries,
        "comparisons": comparisons,
    }

    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(args.output / "per_sample.jsonl", all_per_rows)
    save_csv(args.output / "per_sample.csv", all_per_rows)

    # Human-readable summary.
    lines = [
        "=" * 120,
        "FIXEDSPLIT TRACE RAW-KV 2x2 ABLATION",
        "=" * 120,
        f"Samples: {len(manifest)}",
        f"Noise seed: {args.noise_seed}",
        "",
        f"{'CONDITION':<24}{'ADE(m)':>12}{'FDE(m)':>12}{'HEAD(rad)':>14}{'INFER(ms)':>14}",
        "-" * 120,
    ]
    for condition in CONDITIONS:
        r = summaries[condition]
        lines.append(
            f"{condition:<24}{r['ade_m']:>12.4f}{r['fde_m']:>12.4f}"
            f"{r['heading_mae_rad']:>14.4f}{r['mean_infer_ms']:>14.2f}"
        )

    dr = comparisons["direct_prompt_effect_reasoning_vs_trajectory"]
    rr = comparisons["reuse_trace_effect_reasoning_vs_trajectory"]
    lines.extend([
        "",
        "KEY CONTRASTS (negative delta = reasoning condition has lower error)",
        "-" * 120,
        "DIRECT checkpoint | reasoning prompt - trajectory prompt:",
        f"  ADE delta={dr['delta_reasoning_minus_trajectory']['ade_m']:+.6f} m "
        f"({dr['ade_pct']:+.3f}%)",
        f"  FDE delta={dr['delta_reasoning_minus_trajectory']['fde_m']:+.6f} m",
        f"  Heading delta={dr['delta_reasoning_minus_trajectory']['heading_mae_rad']:+.6f} rad",
        "",
        "REUSE checkpoint | reasoning trace - trajectory trace  <-- PRIMARY:",
        f"  ADE delta={rr['delta_reasoning_minus_trajectory']['ade_m']:+.6f} m "
        f"({rr['ade_pct']:+.3f}%)",
        f"  FDE delta={rr['delta_reasoning_minus_trajectory']['fde_m']:+.6f} m",
        f"  Heading delta={rr['delta_reasoning_minus_trajectory']['heading_mae_rad']:+.6f} rad",
        f"  Reasoning ADE win rate={rr['paired']['ade_m']['a_win_rate_pct']:.2f}%",
        "",
        "Interpretation:",
        "  1) direct_reasoning vs direct_trajectory: prompt/task-instruction representation effect.",
        "  2) reuse_reasoning vs reuse_trajectory: generated trace representation/content effect under SAME reuse AE.",
        "  3) direct vs reuse comparisons are reported but mix separate AE weights with memory differences.",
    ])

    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 120)
    print("FINAL 2x2")
    print("=" * 120)
    print(f"{'CONDITION':<24}{'ADE(m)':>12}{'FDE(m)':>12}{'HEAD(rad)':>14}")
    print("-" * 70)
    for condition in CONDITIONS:
        r = summaries[condition]
        print(
            f"{condition:<24}{r['ade_m']:>12.4f}{r['fde_m']:>12.4f}"
            f"{r['heading_mae_rad']:>14.4f}"
        )

    print("\nPRIMARY: reuse_reasoning - reuse_trajectory")
    print(
        f"  ADE     : {rr['delta_reasoning_minus_trajectory']['ade_m']:+.6f} m "
        f"({rr['ade_pct']:+.3f}%)"
    )
    print(f"  FDE     : {rr['delta_reasoning_minus_trajectory']['fde_m']:+.6f} m")
    print(f"  Heading : {rr['delta_reasoning_minus_trajectory']['heading_mae_rad']:+.6f} rad")
    print(f"  ADE win : {rr['paired']['ade_m']['a_win_rate_pct']:.2f}%")
    print("\nSaved:")
    print("  ", args.output / "summary.txt")
    print("  ", args.output / "summary.json")
    print("  ", args.output / "per_sample.jsonl")
    print("  ", args.output / "per_sample.csv")


if __name__ == "__main__":
    main()
