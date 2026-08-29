#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Action Expert definitions for the capacity-scaled 2 x 2 x 2 ablation.

Factors
-------
Input (2)
  - traj_only      : trajectory-only observation KV cache
  - coc_reasoning  : direct/prompt KV + generated reasoning delta KV

Transformer family (2)
  - encoder : non-causal trajectory-query processing
  - decoder : causal trajectory-query processing

Attention interface (2)
  - self  : projected VLM memory + 10 trajectory queries are concatenated and
            processed by self-attention.
  - cross : VLM memory stays external; trajectory queries attend to it through
            cross-attention.

IMPORTANT CAPACITY RULE
-----------------------
The common hyperparameters are fixed for every architecture:
  H=1536 / L=13 / heads=12 / FF=6144 by default.

Parameter counts are NOT artificially matched across attention interfaces.
Self-attention models naturally have fewer parameters because they do not have
an extra cross-attention module in every layer.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ARCHITECTURES = (
    "encoder_self",
    "encoder_cross",
    "decoder_self",
    "decoder_cross",
)


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

    # Match the current capacity-control code: compute the final regression
    # loss in FP32 even when the forward pass uses BF16 autocast.
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
# COMMON
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

        # Keep the same parameterization as the current decoder+cross code.
        # traj_only uses only id 0; coc_reasoning uses ids 0 and 1.
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
                "segment_ids must be [B,T] matching memory, "
                f"got segment_ids={tuple(segment_ids.shape)}, "
                f"memory={tuple(memory.shape)}"
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


# =============================================================================
# SELF-ATTENTION INTERFACE
# =============================================================================

class SelfAttentionTrajectoryExpert(nn.Module):
    """
    Sequence:
        [projected VLM memory] + [10 learned trajectory queries]

    encoder_self: fully bidirectional self-attention.
    decoder_self: causal self-attention over the concatenated sequence.

    This intentionally follows the original 8-way definition. No extra layers
    are added to compensate for the missing cross-attention parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1536,
        num_steps: int = 10,
        num_layers: int = 13,
        num_heads: int = 12,
        ff_dim: int = 6144,
        dropout: float = 0.1,
        causal: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.causal = bool(causal)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.memory_projector = KVMemoryProjector(input_dim, hidden_dim)
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )

        # Equivalent block type to the original nn.TransformerEncoder stack,
        # but held explicitly so each layer can be gradient-checkpointed.
        base_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_layer) for _ in range(num_layers)]
        )
        self.output = TrajectoryOutputHead(hidden_dim)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    @staticmethod
    def _run_layer(
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return layer(
            x,
            src_mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.memory_projector(memory, segment_ids)
        batch = memory.shape[0]
        queries = self.trajectory_queries.expand(batch, -1, -1)

        tokens = torch.cat([projected, queries], dim=1)

        query_valid = torch.ones(
            (batch, self.num_steps),
            dtype=torch.bool,
            device=memory_mask.device,
        )
        valid_mask = torch.cat([memory_mask.bool(), query_valid], dim=1)
        padding_mask = ~valid_mask

        # checkpoint() requires Tensor arguments. For encoder_self there is no
        # logical causal mask, so use an all-False tensor rather than None.
        if self.causal:
            attn_mask = self._causal_mask(tokens.shape[1], tokens.device)
        else:
            attn_mask = torch.zeros(
                (tokens.shape[1], tokens.shape[1]),
                dtype=torch.bool,
                device=tokens.device,
            )

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                tokens = checkpoint(
                    self._run_layer,
                    layer,
                    tokens,
                    attn_mask,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                tokens = layer(
                    tokens,
                    src_mask=attn_mask,
                    src_key_padding_mask=padding_mask,
                )

        trajectory_tokens = tokens[:, -self.num_steps :, :]
        return self.output(trajectory_tokens)


# =============================================================================
# CROSS-ATTENTION INTERFACE
# =============================================================================

class QueryCrossBlock(nn.Module):
    """
    1) trajectory-query self-attention
       - encoder_cross: bidirectional
       - decoder_cross: causal
    2) query -> projected VLM memory cross-attention
    3) FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        causal_queries: bool,
    ):
        super().__init__()
        self.causal_queries = bool(causal_queries)

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
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        qn = self.norm_q1(queries)
        q_mask = None
        if self.causal_queries:
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
        queries = queries + self.dropout(self.ffn(self.norm_ff(queries)))
        return queries


class CrossAttentionTrajectoryExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1536,
        num_steps: int = 10,
        num_layers: int = 13,
        num_heads: int = 12,
        ff_dim: int = 6144,
        dropout: float = 0.1,
        causal_queries: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.causal_queries = bool(causal_queries)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.memory_projector = KVMemoryProjector(input_dim, hidden_dim)
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                QueryCrossBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    causal_queries=causal_queries,
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
        projected = self.memory_projector(memory, segment_ids)
        queries = self.trajectory_queries.expand(memory.shape[0], -1, -1)

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
                queries = layer(queries, projected, memory_mask)

        return self.output(queries)


# =============================================================================
# FACTORY
# =============================================================================

def parse_architecture(architecture: str) -> Tuple[str, str]:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture={architecture!r}; choices={ARCHITECTURES}"
        )
    family, attention = architecture.split("_", 1)
    return family, attention


def build_action_expert(
    architecture: str,
    input_dim: int,
    hidden_dim: int = 1536,
    num_steps: int = 10,
    num_layers: int = 13,
    num_heads: int = 12,
    ff_dim: int = 6144,
    dropout: float = 0.1,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    family, attention = parse_architecture(architecture)
    causal = family == "decoder"

    common = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        gradient_checkpointing=gradient_checkpointing,
    )

    if attention == "self":
        return SelfAttentionTrajectoryExpert(causal=causal, **common)

    return CrossAttentionTrajectoryExpert(causal_queries=causal, **common)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
