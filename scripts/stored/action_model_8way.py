#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the 2 x 2 x 2 Action Expert ablation.

Factors
-------
1) Cache stage
   - direct    : Qwen last-layer KV immediately before Reasoning generation
   - reasoning : Qwen last-layer KV after Reasoning generation

2) Transformer family
   - encoder: non-causal trajectory-query processing
   - decoder: causal trajectory-query processing

3) Context attention interface
   - self : VLM KV tokens and trajectory queries are concatenated and processed
            with self-attention.
   - cross: VLM KV remains external memory and trajectory queries attend to it
            using cross-attention. Encoder-cross uses bidirectional query
            self-attention; decoder-cross uses causal query self-attention.

All models predict 10 future (x, y, yaw) waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_STAGES = ("direct", "reasoning")
TRANSFORMER_FAMILIES = ("encoder", "decoder")
ATTENTION_INTERFACES = ("self", "cross")

MODEL_NAMES = tuple(
    f"{stage}_{family}_{attention}"
    for stage in CACHE_STAGES
    for family in TRANSFORMER_FAMILIES
    for attention in ATTENTION_INTERFACES
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

    xy_loss = F.smooth_l1_loss(pred[..., :2], gt[..., :2])
    heading_loss = (1.0 - torch.cos(pred[..., 2] - gt[..., 2])).mean()
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


# =============================================================================
# COMMON EMBEDDING / OUTPUT
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
        # 0 = prompt/direct tokens, 1 = generated reasoning tokens
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
    Context is consumed through self-attention over:
        [projected VLM KV memory] + [10 learned trajectory queries]

    encoder mode: fully bidirectional self-attention
    decoder mode: causal self-attention over the concatenated sequence
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_steps: int = 10,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.causal = bool(causal)

        self.memory_projector = KVMemoryProjector(input_dim, hidden_dim)
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, self.num_steps, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = TrajectoryOutputHead(hidden_dim)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
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
            (batch, self.num_steps), dtype=torch.bool, device=memory_mask.device
        )
        valid_mask = torch.cat([memory_mask.bool(), query_valid], dim=1)
        padding_mask = ~valid_mask

        attn_mask = None
        if self.causal:
            attn_mask = self._causal_mask(tokens.shape[1], tokens.device)

        encoded = self.transformer(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )
        trajectory_tokens = encoded[:, -self.num_steps :, :]
        return self.output(trajectory_tokens)


# =============================================================================
# CROSS-ATTENTION INTERFACE
# =============================================================================

class QueryCrossBlock(nn.Module):
    """
    Transformer-style trajectory-query block.

    1) query self-attention
       - encoder family: bidirectional
       - decoder family: causal
    2) query -> VLM KV cross-attention
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
        hidden_dim: int = 512,
        num_steps: int = 10,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        causal_queries: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.causal_queries = bool(causal_queries)

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
            queries = layer(queries, projected, memory_mask)
        return self.output(queries)


# =============================================================================
# FACTORY
# =============================================================================

def parse_model_name(model_name: str) -> Tuple[str, str, str]:
    parts = model_name.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Bad model name {model_name!r}. Expected <direct|reasoning>_<encoder|decoder>_<self|cross>."
        )
    stage, family, attention = parts
    if stage not in CACHE_STAGES:
        raise ValueError(f"Unknown cache stage: {stage}")
    if family not in TRANSFORMER_FAMILIES:
        raise ValueError(f"Unknown transformer family: {family}")
    if attention not in ATTENTION_INTERFACES:
        raise ValueError(f"Unknown attention interface: {attention}")
    return stage, family, attention


def build_action_expert(
    model_name: str,
    input_dim: int,
    hidden_dim: int = 512,
    num_steps: int = 10,
    num_layers: int = 3,
    num_heads: int = 8,
    ff_dim: int = 2048,
    dropout: float = 0.1,
) -> nn.Module:
    _, family, attention = parse_model_name(model_name)
    causal = family == "decoder"

    common = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )

    if attention == "self":
        return SelfAttentionTrajectoryExpert(causal=causal, **common)

    return CrossAttentionTrajectoryExpert(causal_queries=causal, **common)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
