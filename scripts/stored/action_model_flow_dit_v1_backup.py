#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled ~0.5B Flow-Matching / DiT-style Action Expert.

Goal
----
Compare fairly against the current 0.499B direct-regression Transformer.

Kept matched to the Transformer baseline:
  - same cached-memory interface: learned KVMemoryProjector + segment embedding
  - same decoder-cross backbone geometry: H1536 / 13L / 12 heads / FF6144
  - same cross-attention to projected VLM memory
  - same output-head parameterization
  - same 10 x (x, y, yaw) trajectory representation

Changed for Flow Matching:
  - learnable trajectory queries -> noisy trajectory x_t + timestep embedding
  - trajectory-token self-attention is non-causal (joint denoising)
  - direct regression loss -> velocity-field MSE
  - inference uses Euler integration from noise to trajectory

IMPORTANT:
This intentionally does NOT split the saved final-layer [K||V] tensor and inject
it as raw prefix K/V into every Action Expert layer. Current caches do not contain
the all-layer VLM past_key_values / RoPE state required for faithful Alpamayo-style
raw KV reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================================
# METRICS
# =============================================================================

def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_metrics_np(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
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
# NORMALIZATION
# =============================================================================

@dataclass
class TrajectoryNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_dict(cls, payload: dict) -> "TrajectoryNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu().float().tolist(),
            "std": self.std.detach().cpu().float().tolist(),
        }

    def to(self, device: torch.device) -> "TrajectoryNormalizer":
        return TrajectoryNormalizer(
            mean=self.mean.to(device=device, dtype=torch.float32),
            std=self.std.to(device=device, dtype=torch.float32),
        )

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return (trajectory - self.mean.view(1, 1, 3)) / self.std.view(1, 1, 3)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory * self.std.view(1, 1, 3) + self.mean.view(1, 1, 3)


# =============================================================================
# SAME MEMORY CONDITIONING AS THE DIRECT TRANSFORMER
# =============================================================================

class KVMemoryProjector(nn.Module):
    """Intentionally identical to the current 0.499B Transformer baseline."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.segment_embedding = nn.Embedding(2, self.hidden_dim)

    def forward(self, memory: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"memory must be [B,T,D], got {tuple(memory.shape)}")
        if memory.shape[-1] != self.input_dim:
            raise ValueError(
                f"KV dim mismatch: got={memory.shape[-1]}, expected={self.input_dim}"
            )
        projected = self.net(memory)
        return projected + self.segment_embedding(segment_ids.clamp(0, 1))


# =============================================================================
# FLOW ACTION/TIME INPUT
# =============================================================================

class FourierEncoder(nn.Module):
    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Fourier dim must be even")
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freqs", freqs[None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None].float() * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class FlowActionInputProjection(nn.Module):
    """
    Per-waypoint noisy action + scalar diffusion time -> Transformer hidden token.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        fourier_dim: int = 20,
        mlp_hidden: int = 512,
        max_freq: float = 100.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_encoders = nn.ModuleList(
            [FourierEncoder(fourier_dim, max_freq=max_freq) for _ in range(action_dim)]
        )
        self.time_encoder = FourierEncoder(fourier_dim, max_freq=max_freq)

        in_dim = (action_dim + 1) * fourier_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"x_t must be [B,N,{self.action_dim}], got {tuple(x_t.shape)}"
            )

        b, n, _ = x_t.shape
        action_feats = torch.cat(
            [enc(x_t[:, :, i]) for i, enc in enumerate(self.action_encoders)],
            dim=-1,
        )

        if t.ndim == 1:
            t_scalar = t
        else:
            t_scalar = t.reshape(b, -1)[:, 0]
        time_feats = self.time_encoder(t_scalar).unsqueeze(1).expand(-1, n, -1)

        feats = torch.cat([action_feats, time_feats], dim=-1)
        return self.net(feats.to(dtype=self.net[0].weight.dtype))


# =============================================================================
# SAME DECODER-CROSS BLOCK GEOMETRY, NON-CAUSAL FOR DENOISING
# =============================================================================

class FlowDecoderCrossBlock(nn.Module):
    """
    Same parameterization as Transformer DecoderCrossBlock.

    Difference: self-attention among noisy trajectory tokens is non-causal so all
    10 waypoints are denoised jointly. Cross-attention conditioning is unchanged.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.norm_q1 = nn.LayerNorm(hidden_dim)
        self.query_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q2 = nn.LayerNorm(hidden_dim)
        self.norm_mem = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
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
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        qn = self.norm_q1(queries)
        self_out, _ = self.query_self_attn(
            qn,
            qn,
            qn,
            need_weights=False,
        )
        queries = queries + self.dropout(self_out)

        qn = self.norm_q2(queries)
        mn = self.norm_mem(memory)
        cross_out, _ = self.cross_attn(
            qn,
            mn,
            mn,
            key_padding_mask=~memory_mask.bool(),
            need_weights=False,
        )
        queries = queries + self.dropout(cross_out)
        queries = queries + self.dropout(self.ffn(self.norm_ff(queries)))
        return queries


class VectorFieldOutputHead(nn.Module):
    """Same parameterization as the direct Transformer's trajectory output head."""

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


class FlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1536,
        num_steps: int = 10,
        num_layers: int = 13,
        num_heads: int = 12,
        ff_dim: int = 6144,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.memory_projector = KVMemoryProjector(input_dim, hidden_dim)
        self.action_in = FlowActionInputProjection(action_dim=3, hidden_dim=hidden_dim)

        # Keeps explicit waypoint identity after noisy-action projection.
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                FlowDecoderCrossBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = VectorFieldOutputHead(hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(x_t.shape[1:]) != (self.num_steps, 3):
            raise ValueError(
                f"x_t must be [B,{self.num_steps},3], got {tuple(x_t.shape)}"
            )

        projected_memory = self.memory_projector(memory, segment_ids)
        queries = self.action_in(x_t, t)
        queries = queries + self.horizon_embedding.to(dtype=queries.dtype)

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                queries = checkpoint(
                    layer,
                    queries,
                    projected_memory,
                    memory_mask,
                    use_reentrant=False,
                )
            else:
                queries = layer(queries, projected_memory, memory_mask)

        return self.output(queries)


def build_flow_dit(
    input_dim: int,
    hidden_dim: int = 1536,
    num_steps: int = 10,
    num_layers: int = 13,
    num_heads: int = 12,
    ff_dim: int = 6144,
    dropout: float = 0.1,
    gradient_checkpointing: bool = True,
) -> FlowMatchingActionExpert:
    return FlowMatchingActionExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        gradient_checkpointing=gradient_checkpointing,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# FLOW MATCHING
# =============================================================================

def _sample_timesteps(
    batch_size: int,
    rng: torch.Generator,
    sampler: str,
) -> torch.Tensor:
    if sampler == "uniform":
        return torch.rand(
            (batch_size, 1, 1),
            generator=rng,
            device="cpu",
            dtype=torch.float32,
        )

    if sampler == "beta":
        # Beta(alpha=1.5, beta=1.0) using its inverse CDF, then map to [0, 0.999].
        # Retained from the previous Flow-Matching prototype for continuity.
        u = torch.rand(
            (batch_size, 1, 1),
            generator=rng,
            device="cpu",
            dtype=torch.float32,
        )
        beta_sample = u.pow(1.0 / 1.5)
        return 0.999 - 0.999 * beta_sample

    raise ValueError(f"Unknown timestep sampler: {sampler}")


def flow_matching_batch(
    gt_normalized: torch.Tensor,
    rng: torch.Generator,
    timestep_sampler: str = "beta",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Linear flow from Gaussian noise x0 to normalized GT x1:
      x_t = (1-t) * x0 + t * x1
      v*  = x1 - x0
    """
    x0 = torch.randn(
        gt_normalized.shape,
        generator=rng,
        dtype=torch.float32,
        device="cpu",
    ).to(gt_normalized.device)

    t = _sample_timesteps(
        gt_normalized.shape[0],
        rng=rng,
        sampler=timestep_sampler,
    ).to(gt_normalized.device)

    x_t = (1.0 - t) * x0 + t * gt_normalized
    v_target = gt_normalized - x0
    return x_t, t, v_target, x0


@torch.inference_mode()
def euler_sample(
    model: nn.Module,
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    normalizer: TrajectoryNormalizer,
    solver_steps: int = 10,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    if solver_steps < 1:
        raise ValueError("solver_steps must be >= 1")

    b = int(memory.shape[0])
    device = memory.device

    if noise is None:
        if rng is None:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(0)
        noise = torch.randn(
            (b, model.num_steps, 3),
            generator=rng,
            dtype=torch.float32,
            device="cpu",
        ).to(device)
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    x = noise
    times = torch.linspace(0.0, 1.0, solver_steps + 1, device=device)

    for i in range(solver_steps):
        t_now = float(times[i].item())
        dt = float((times[i + 1] - times[i]).item())
        t = torch.full((b, 1, 1), t_now, device=device, dtype=torch.float32)

        # The caller controls autocast during training. In inference this model may
        # have FP32 master weights, so enable BF16 autocast here for matched runtime.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            v = model(
                x_t=x,
                t=t,
                memory=memory,
                memory_mask=memory_mask,
                segment_ids=segment_ids,
            )
        x = x + dt * v.float()

    return normalizer.denormalize(x.float())
