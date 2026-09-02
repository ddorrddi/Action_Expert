#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled ~0.5B Flow-Matching / DiT-style Action Expert (v2).

Goal
----
Keep the direct Transformer comparison controlled while fixing the most likely
Flow-specific optimization problems found in the first implementation.

Matched to the direct Transformer baseline
------------------------------------------
  - same cached-memory interface: learned KVMemoryProjector + segment embedding
  - same decoder-cross backbone scale: H1536 / 13L / 12 heads / FF6144
  - same cross-attention to projected VLM memory
  - same output-head parameterization
  - same 10 x (x, y, yaw) trajectory representation

Flow-specific design
--------------------
  - normalized noisy trajectory x_t is projected DIRECTLY with Linear(3 -> H)
  - only scalar flow time t uses Fourier features
  - explicit learned waypoint/horizon embedding for all 10 action tokens
  - non-causal self-attention among trajectory tokens
  - linear conditional flow matching:
        x_t = (1-t) * x0 + t * x1
        v*  = x1 - x0
  - Euler integration from t=0 noise to t=1 trajectory

Compatibility
-------------
TrajectoryNormalizer supports both:
  - legacy global stats: mean/std shape [3]
  - new per-waypoint stats: mean/std shape [N, 3]
so test utilities can keep using the same API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# =============================================================================
# METRICS
# =============================================================================

def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_metrics_np(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"expected [B,N,3], got {pred.shape}")

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

    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"expected [B,N,3], got {pred.shape}")

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
    """
    Normalize trajectory coordinates.

    Supported statistics:
      legacy global:       mean/std [3]
      v2 per-waypoint:     mean/std [N,3]

    Per-waypoint statistics are preferred for Flow Matching because waypoint
    distributions change strongly across the prediction horizon.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)

        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"normalizer mean/std shape mismatch: "
                f"{tuple(self.mean.shape)} vs {tuple(self.std.shape)}"
            )

        if self.mean.ndim == 1:
            if tuple(self.mean.shape) != (3,):
                raise ValueError(
                    f"global normalizer must have shape [3], got {tuple(self.mean.shape)}"
                )
        elif self.mean.ndim == 2:
            if self.mean.shape[-1] != 3:
                raise ValueError(
                    f"per-waypoint normalizer must have shape [N,3], "
                    f"got {tuple(self.mean.shape)}"
                )
        else:
            raise ValueError(
                f"normalizer stats must be [3] or [N,3], got {tuple(self.mean.shape)}"
            )

        if not torch.isfinite(self.mean).all():
            raise ValueError("normalizer mean contains non-finite values")
        if not torch.isfinite(self.std).all():
            raise ValueError("normalizer std contains non-finite values")
        if torch.any(self.std <= 0):
            raise ValueError("normalizer std must be > 0")

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

    def to(self, device: torch.device | str) -> "TrajectoryNormalizer":
        return TrajectoryNormalizer(
            mean=self.mean.to(device=device, dtype=torch.float32),
            std=self.std.to(device=device, dtype=torch.float32),
        )

    @property
    def mode(self) -> str:
        return "global_xyz" if self.mean.ndim == 1 else "per_waypoint_xyz"

    def _stats_for(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.ndim not in (2, 3):
            raise ValueError(
                f"trajectory must be [N,3] or [B,N,3], got {tuple(trajectory.shape)}"
            )
        if trajectory.shape[-1] != 3:
            raise ValueError(
                f"trajectory last dim must be 3, got {tuple(trajectory.shape)}"
            )

        if self.mean.ndim == 1:
            if trajectory.ndim == 2:
                mean = self.mean.view(1, 3)
                std = self.std.view(1, 3)
            else:
                mean = self.mean.view(1, 1, 3)
                std = self.std.view(1, 1, 3)
            return mean, std

        # Per-waypoint stats [N,3].
        n = int(trajectory.shape[-2])
        if int(self.mean.shape[0]) != n:
            raise ValueError(
                f"normalizer waypoint count mismatch: stats={self.mean.shape[0]}, traj={n}"
            )

        if trajectory.ndim == 2:
            return self.mean, self.std

        return self.mean.unsqueeze(0), self.std.unsqueeze(0)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        mean = mean.to(device=trajectory.device, dtype=trajectory.dtype)
        std = std.to(device=trajectory.device, dtype=trajectory.dtype)
        return (trajectory - mean) / std

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        mean = mean.to(device=trajectory.device, dtype=trajectory.dtype)
        std = std.to(device=trajectory.device, dtype=trajectory.dtype)
        return trajectory * std + mean


# =============================================================================
# SAME MEMORY CONDITIONING AS THE DIRECT TRANSFORMER
# =============================================================================

class KVMemoryProjector(nn.Module):
    """Intentionally matched to the current ~0.499B Transformer baseline."""

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

    def forward(
        self,
        memory: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"memory must be [B,T,D], got {tuple(memory.shape)}")
        if memory.shape[-1] != self.input_dim:
            raise ValueError(
                f"KV dim mismatch: got={memory.shape[-1]}, expected={self.input_dim}"
            )
        if segment_ids.shape != memory.shape[:2]:
            raise ValueError(
                f"segment_ids must be [B,T]={tuple(memory.shape[:2])}, "
                f"got {tuple(segment_ids.shape)}"
            )

        projected = self.net(memory)
        seg = self.segment_embedding(segment_ids.long().clamp(0, 1))
        return projected + seg


# =============================================================================
# FLOW ACTION/TIME INPUT
# =============================================================================

class FourierEncoder(nn.Module):
    """Fourier features used ONLY for scalar flow time t."""

    def __init__(self, dim: int = 20, max_freq: float = 100.0):
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("Fourier dim must be a positive even integer >= 2")
        if max_freq <= 1.0:
            raise ValueError("max_freq must be > 1")

        half = dim // 2
        freqs = torch.logspace(
            0.0,
            math.log10(float(max_freq)),
            steps=half,
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: arbitrary shape -> [..., fourier_dim]
        arg = x.float().unsqueeze(-1) * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class FlowActionInputProjection(nn.Module):
    """
    v2 action input projection.

    IMPORTANT CHANGE FROM v1:
      v1: Fourier(x) || Fourier(y) || Fourier(yaw) || Fourier(t) -> MLP
      v2: raw normalized (x_t, y_t, yaw_t) -> Linear(3,H)
          Fourier(t) -> small MLP -> H
          then ADD the two hidden representations.

    This preserves the actual continuous geometry of x_t instead of making the
    model recover it from high-frequency periodic features.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        time_fourier_dim: int = 20,
        time_mlp_hidden: int = 512,
        max_time_freq: float = 100.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.time_encoder = FourierEncoder(
            dim=time_fourier_dim,
            max_freq=max_time_freq,
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_fourier_dim, time_mlp_hidden),
            nn.LayerNorm(time_mlp_hidden),
            nn.GELU(),
            nn.Linear(time_mlp_hidden, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != self.action_dim:
            raise ValueError(
                f"x_t must be [B,N,{self.action_dim}], got {tuple(x_t.shape)}"
            )

        b, _, _ = x_t.shape

        action_dtype = self.action_proj[0].weight.dtype
        action_hidden = self.action_proj(x_t.to(dtype=action_dtype))

        if t.ndim == 1:
            if t.shape[0] != b:
                raise ValueError(f"t batch mismatch: {tuple(t.shape)} vs B={b}")
            t_scalar = t
        else:
            if t.shape[0] != b:
                raise ValueError(f"t batch mismatch: {tuple(t.shape)} vs B={b}")
            t_scalar = t.reshape(b, -1)[:, 0]

        time_feat = self.time_encoder(t_scalar)
        time_dtype = self.time_proj[0].weight.dtype
        time_hidden = self.time_proj(time_feat.to(dtype=time_dtype))
        time_hidden = time_hidden.unsqueeze(1)

        return action_hidden + time_hidden


# =============================================================================
# SAME DECODER-CROSS BLOCK GEOMETRY, NON-CAUSAL FOR DENOISING
# =============================================================================

class FlowDecoderCrossBlock(nn.Module):
    """
    Same parameterization as the direct Transformer's DecoderCrossBlock.

    Difference:
      trajectory-token self-attention is non-causal because all 10 noisy
      waypoints are denoised jointly.
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
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError(
                f"memory_mask must be {tuple(memory.shape[:2])}, "
                f"got {tuple(memory_mask.shape)}"
            )

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

        queries = queries + self.dropout(
            self.ffn(self.norm_ff(queries))
        )
        return queries


class VectorFieldOutputHead(nn.Module):
    """Same output-head parameterization as the direct Transformer baseline."""

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

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.memory_projector = KVMemoryProjector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )

        self.action_in = FlowActionInputProjection(
            action_dim=3,
            hidden_dim=self.hidden_dim,
        )

        # Explicit waypoint identity: waypoint 0 ... waypoint 9.
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, self.num_steps, self.hidden_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                FlowDecoderCrossBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output = VectorFieldOutputHead(self.hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or tuple(x_t.shape[1:]) != (self.num_steps, 3):
            raise ValueError(
                f"x_t must be [B,{self.num_steps},3], got {tuple(x_t.shape)}"
            )
        if memory.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"batch mismatch x_t={x_t.shape[0]} memory={memory.shape[0]}"
            )

        projected_memory = self.memory_projector(memory, segment_ids)

        queries = self.action_in(x_t, t)
        queries = queries + self.horizon_embedding.to(
            device=queries.device,
            dtype=queries.dtype,
        )

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
                queries = layer(
                    queries,
                    projected_memory,
                    memory_mask,
                )

        return self.output(queries)


# =============================================================================
# BUILD / PARAM COUNT
# =============================================================================

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
    """
    Sample flow time t on CPU for deterministic branch-matched RNG.

    uniform (recommended/default):
        t ~ U[0,1)

    beta:
        t ~ Beta(alpha=1.5, beta=1.0), implemented via inverse CDF
        t = U^(1/1.5), which correctly biases samples toward t=1.

    NOTE:
    The old implementation returned 0.999 - 0.999 * U^(1/1.5), which reversed
    the intended distribution and biased training toward t=0.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    u = torch.rand(
        (batch_size, 1, 1),
        generator=rng,
        device="cpu",
        dtype=torch.float32,
    )

    if sampler == "uniform":
        return u

    if sampler == "beta":
        return u.pow(1.0 / 1.5)

    raise ValueError(f"Unknown timestep sampler: {sampler}")


def flow_matching_batch(
    gt_normalized: torch.Tensor,
    rng: torch.Generator,
    timestep_sampler: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Linear conditional flow from Gaussian x0 to normalized GT x1.

      x0 ~ N(0, I)
      x1 = normalized GT trajectory
      x_t = (1-t) * x0 + t * x1
      v*  = x1 - x0

    Returns:
      x_t, t, v_target, x0
    """
    if gt_normalized.ndim != 3 or gt_normalized.shape[-1] != 3:
        raise ValueError(
            f"gt_normalized must be [B,N,3], got {tuple(gt_normalized.shape)}"
        )
    if not torch.isfinite(gt_normalized).all():
        raise ValueError("gt_normalized contains non-finite values")

    # Keep RNG on CPU so both branches can use exactly the same random stream.
    x0 = torch.randn(
        gt_normalized.shape,
        generator=rng,
        dtype=torch.float32,
        device="cpu",
    ).to(
        device=gt_normalized.device,
        dtype=torch.float32,
    )

    t = _sample_timesteps(
        batch_size=int(gt_normalized.shape[0]),
        rng=rng,
        sampler=timestep_sampler,
    ).to(
        device=gt_normalized.device,
        dtype=torch.float32,
    )

    x1 = gt_normalized.float()
    x_t = (1.0 - t) * x0 + t * x1
    v_target = x1 - x0

    return x_t, t, v_target, x0


# =============================================================================
# SAMPLING
# =============================================================================

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
    """
    Integrate learned vector field from Gaussian noise at t=0 to trajectory at t=1.

    x_{k+1} = x_k + dt * v_theta(x_k, t_k, condition)
    """
    if solver_steps < 1:
        raise ValueError("solver_steps must be >= 1")

    b = int(memory.shape[0])
    device = memory.device

    if noise is None:
        if rng is None:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(0)

        noise = torch.randn(
            (b, int(model.num_steps), 3),
            generator=rng,
            dtype=torch.float32,
            device="cpu",
        ).to(device=device, dtype=torch.float32)
    else:
        expected = (b, int(model.num_steps), 3)
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"noise must be {expected}, got {tuple(noise.shape)}"
            )
        noise = noise.to(device=device, dtype=torch.float32)

    x = noise.clone()
    dt = 1.0 / float(solver_steps)

    for i in range(solver_steps):
        t_now = float(i) / float(solver_steps)
        t = torch.full(
            (b, 1, 1),
            t_now,
            device=device,
            dtype=torch.float32,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            v = model(
                x_t=x,
                t=t,
                memory=memory,
                memory_mask=memory_mask,
                segment_ids=segment_ids,
            )

        if not torch.isfinite(v).all():
            raise RuntimeError(
                f"non-finite vector field during Euler sampling at step {i}/{solver_steps}"
            )

        x = x + dt * v.float()

    return normalizer.denormalize(x.float())


# =============================================================================
# OPTIONAL SANITY CHECK
# =============================================================================

def linear_flow_oracle_sanity_check(
    seed: int = 1234,
    solver_steps: int = 10,
) -> float:
    """
    Verify the sign/direction convention independently of the neural network.

    Because the target vector for the linear path is constant v=x1-x0, Euler
    integration should recover x1 up to floating-point error.
    """
    if solver_steps < 1:
        raise ValueError("solver_steps must be >= 1")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x0 = torch.randn((8, 10, 3), generator=g, dtype=torch.float32)
    x1 = torch.randn((8, 10, 3), generator=g, dtype=torch.float32)
    v = x1 - x0

    x = x0.clone()
    dt = 1.0 / float(solver_steps)
    for _ in range(solver_steps):
        x = x + dt * v

    return float((x - x1).abs().max().item())
