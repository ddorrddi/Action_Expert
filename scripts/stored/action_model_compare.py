#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed ~0.5B Action Expert for the Alpamayo-style two-way ablation.

Purpose
-------
This is a CAPACITY-ONLY re-run of the previous small Transformer ablation.

Preserved exactly:
  - decoder + cross-attention interface
  - causal trajectory-query self-attention
  - learned KV memory projection
  - direct 10 x (x, y, yaw) regression
  - SmoothL1 XY + cosine heading loss
  - same model class for traj_only / coc_reasoning

Scaled only:
  old : H512  / 3 layers  / 8 heads  / FF2048
  new : H1536 / 13 layers / 12 heads / FF6144

With input_dim=2048, the new model has:
  499,051,011 trainable parameters ~= 0.499B

Inputs:
  memory      : [B,T,D] cached VLM K/V representation
  memory_mask : [B,T]
  segment_ids : 0 = observation/prompt, 1 = generated reasoning

Output:
  10 x (x, y, yaw)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================================
# LOSS / METRICS
# =============================================================================

def trajectory_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    heading_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.shape != gt.shape:
        raise ValueError(
            f"trajectory shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}"
        )

    # Keep the loss numerically identical/stable under BF16 autocast by
    # evaluating the final regression loss in FP32.
    pred_f = pred.float()
    gt_f = gt.float()

    xy_loss = F.smooth_l1_loss(pred_f[..., :2], gt_f[..., :2])
    heading_loss = (1.0 - torch.cos(pred_f[..., 2] - gt_f[..., 2])).mean()
    total = xy_loss + float(heading_weight) * heading_loss
    return total, xy_loss.detach(), heading_loss.detach()


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
# MODEL
# =============================================================================

class KVMemoryProjector(nn.Module):
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

        # Same parameterization for both branches.
        # TRAJ_ONLY uses only id 0; COC_REASONING uses ids 0 and 1.
        self.segment_embedding = nn.Embedding(2, self.hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(
                f"memory must be [B,T,D], got {tuple(memory.shape)}"
            )
        if memory.shape[-1] != self.input_dim:
            raise ValueError(
                f"KV dim mismatch: got={memory.shape[-1]}, expected={self.input_dim}"
            )

        projected = self.net(memory)
        return projected + self.segment_embedding(segment_ids.clamp(0, 1))


class TrajectoryOutputHead(nn.Module):
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


class DecoderCrossBlock(nn.Module):
    """
    SAME block as the original small ablation:
      causal trajectory-query self-attention
      + query->VLM cross-attention
      + FFN
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

    @staticmethod
    def _causal_mask(
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        qn = self.norm_q1(queries)
        q_mask = self._causal_mask(qn.shape[1], qn.device)

        self_out, _ = self.query_self_attn(
            qn,
            qn,
            qn,
            attn_mask=q_mask,
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


class AlpamayoAblationActionExpert(nn.Module):
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

        self.memory_projector = KVMemoryProjector(
            input_dim,
            hidden_dim,
        )

        self.trajectory_queries = nn.Parameter(
            torch.randn(
                1,
                self.num_steps,
                hidden_dim,
            )
            * 0.02
        )

        self.layers = nn.ModuleList(
            [
                DecoderCrossBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output = TrajectoryOutputHead(hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.memory_projector(
            memory,
            segment_ids,
        )

        queries = self.trajectory_queries.expand(
            memory.shape[0],
            -1,
            -1,
        )

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                queries = checkpoint(
                    layer,
                    queries,
                    projected,
                    memory_mask,
                    use_reentrant=False,
                )
            else:
                queries = layer(
                    queries,
                    projected,
                    memory_mask,
                )

        return self.output(queries)


def build_action_expert(
    input_dim: int,
    hidden_dim: int = 1536,
    num_steps: int = 10,
    num_layers: int = 13,
    num_heads: int = 12,
    ff_dim: int = 6144,
    dropout: float = 0.1,
    gradient_checkpointing: bool = True,
) -> AlpamayoAblationActionExpert:
    return AlpamayoAblationActionExpert(
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
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
