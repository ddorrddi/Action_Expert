#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Expert definitions for the final 2 x 2 x 2 ablation.

Ablation axes
-------------
1) Input (handled by the trainer)
   - direct
   - reasoning

2) Structure
   - encoder : encoder-style fusion
               [projected VLM memory + trajectory queries] are processed in
               one self-attention stack.
   - decoder : decoder-style fusion
               trajectory queries are a separate stream and cross-attend to
               projected VLM memory.

3) Attention direction
   - bidirectional : trajectory queries can attend to all trajectory queries.
   - causal        : trajectory query t can attend only to trajectory queries
                     <= t.

Model names produced by the trainer
-----------------------------------
direct_encoder_bidirectional
direct_encoder_causal
direct_decoder_bidirectional
direct_decoder_causal
reasoning_encoder_bidirectional
reasoning_encoder_causal
reasoning_decoder_bidirectional
reasoning_decoder_causal

Capacity
--------
The common capacity is intentionally kept from the existing ~0.5B experiment:
    H=1536 / L=13 / heads=12 / FF=6144 / dropout=0.1

The decoder-style model contains an additional cross-attention module per
layer, so exact trainable parameter counts are naturally different from the
encoder-style model. The hidden size/depth/FF width are not changed.
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
    "encoder_bidirectional",
    "encoder_causal",
    "decoder_bidirectional",
    "decoder_causal",
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
            f"trajectory shape mismatch: pred={tuple(pred.shape)}, "
            f"gt={tuple(gt.shape)}"
        )

    pred_f = pred.float()
    gt_f = gt.float()

    xy_loss = F.smooth_l1_loss(
        pred_f[..., :2],
        gt_f[..., :2],
    )
    heading_loss = (
        1.0
        - torch.cos(
            pred_f[..., 2]
            - gt_f[..., 2]
        )
    ).mean()

    total = (
        xy_loss
        + float(heading_weight) * heading_loss
    )
    return total, xy_loss.detach(), heading_loss.detach()


def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (
        x + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def trajectory_metrics_np(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    pred = np.asarray(
        pred,
        dtype=np.float64,
    )
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    xy_error = np.linalg.norm(
        pred[..., :2]
        - gt[..., :2],
        axis=-1,
    )
    heading = np.abs(
        wrap_angle_np(
            pred[..., 2]
            - gt[..., 2]
        )
    )

    return {
        "ade_m": float(
            xy_error.mean()
        ),
        "fde_m": float(
            xy_error[:, -1].mean()
        ),
        "heading_mae_rad": float(
            heading.mean()
        ),
    }


def per_sample_metrics_np(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    pred = np.asarray(
        pred,
        dtype=np.float64,
    )
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    xy_error = np.linalg.norm(
        pred[..., :2]
        - gt[..., :2],
        axis=-1,
    )
    heading = np.abs(
        wrap_angle_np(
            pred[..., 2]
            - gt[..., 2]
        )
    )

    return {
        "ade_m": xy_error.mean(axis=1),
        "fde_m": xy_error[:, -1],
        "heading_mae_rad": heading.mean(axis=1),
    }


# =============================================================================
# COMMON
# =============================================================================

class KVMemoryProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.input_dim = int(
            input_dim
        )
        self.hidden_dim = int(
            hidden_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.LayerNorm(
                self.hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        # direct uses segment 0 only.
        # reasoning uses prompt/direct=0 and reasoning-delta=1.
        self.segment_embedding = nn.Embedding(
            2,
            self.hidden_dim,
        )

    def forward(
        self,
        memory: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(
                "memory must be [B,T,D], "
                f"got {tuple(memory.shape)}"
            )

        if memory.shape[-1] != self.input_dim:
            raise ValueError(
                f"KV dim mismatch: "
                f"got={memory.shape[-1]}, "
                f"expected={self.input_dim}"
            )

        if segment_ids.shape != memory.shape[:2]:
            raise ValueError(
                "segment_ids must match [B,T]: "
                f"segment_ids={tuple(segment_ids.shape)} "
                f"memory={tuple(memory.shape)}"
            )

        projected = self.net(
            memory
        )

        return (
            projected
            + self.segment_embedding(
                segment_ids.clamp(0, 1)
            )
        )


class TrajectoryOutputHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                3,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.norm(x)
        )


# =============================================================================
# ENCODER-STYLE FUSION
# =============================================================================

class EncoderStyleTrajectoryExpert(nn.Module):
    """
    Encoder-style Action Expert.

    Input sequence:
        [projected VLM memory] + [10 learned trajectory queries]

    The VLM memory and trajectory queries live in one Transformer self-attention
    stack. This is the structural "encoder" branch of the ablation.

    Attention direction is a separate factor:
      - bidirectional: trajectory queries see all trajectory queries.
      - causal: query i cannot see future query j > i.

    For the causal variant, VLM memory tokens remain mutually bidirectional.
    Query tokens can always attend to all VLM memory. This isolates the causal
    ablation to trajectory-query direction instead of accidentally making the
    already-computed VLM memory itself causal again.
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
        causal_queries: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.input_dim = int(
            input_dim
        )
        self.hidden_dim = int(
            hidden_dim
        )
        self.num_steps = int(
            num_steps
        )
        self.causal_queries = bool(
            causal_queries
        )
        self.gradient_checkpointing = bool(
            gradient_checkpointing
        )

        self.memory_projector = KVMemoryProjector(
            input_dim,
            hidden_dim,
        )

        self.trajectory_queries = nn.Parameter(
            torch.randn(
                1,
                self.num_steps,
                hidden_dim,
            ) * 0.02
        )

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
            [
                copy.deepcopy(
                    base_layer
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.output = TrajectoryOutputHead(
            hidden_dim
        )

    def _attention_mask(
        self,
        memory_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Boolean mask for [memory | queries].
        True = blocked.

        Bidirectional:
            no logical blocking.

        Causal:
            memory -> memory : allowed
            memory -> query  : blocked
            query  -> memory : allowed
            query i -> query j : allowed only when j <= i

        This keeps "structure" and "direction" as independent ablation axes.
        """
        total = (
            memory_length
            + self.num_steps
        )

        mask = torch.zeros(
            (total, total),
            dtype=torch.bool,
            device=device,
        )

        if not self.causal_queries:
            return mask

        # VLM memory is conditioning context. Do not let memory tokens absorb
        # trajectory-query content in the causal branch.
        mask[
            :memory_length,
            memory_length:
        ] = True

        # Causal direction only among trajectory queries.
        q_causal = torch.triu(
            torch.ones(
                (
                    self.num_steps,
                    self.num_steps,
                ),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

        mask[
            memory_length:,
            memory_length:
        ] = q_causal

        return mask

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
        projected = self.memory_projector(
            memory,
            segment_ids,
        )

        batch = int(
            memory.shape[0]
        )
        memory_length = int(
            memory.shape[1]
        )

        queries = self.trajectory_queries.expand(
            batch,
            -1,
            -1,
        )

        tokens = torch.cat(
            [
                projected,
                queries,
            ],
            dim=1,
        )

        query_valid = torch.ones(
            (
                batch,
                self.num_steps,
            ),
            dtype=torch.bool,
            device=memory_mask.device,
        )

        valid_mask = torch.cat(
            [
                memory_mask.bool(),
                query_valid,
            ],
            dim=1,
        )

        padding_mask = (
            ~valid_mask
        )

        attn_mask = self._attention_mask(
            memory_length=memory_length,
            device=tokens.device,
        )

        for layer in self.layers:
            if (
                self.training
                and self.gradient_checkpointing
            ):
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

        trajectory_tokens = tokens[
            :,
            -self.num_steps:,
            :,
        ]

        return self.output(
            trajectory_tokens
        )


# =============================================================================
# DECODER-STYLE FUSION
# =============================================================================

class DecoderStyleBlock(nn.Module):
    """
    Transformer decoder-style block:

      1) trajectory-query self-attention
         - bidirectional or causal
      2) trajectory-query -> VLM-memory cross-attention
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

        self.causal_queries = bool(
            causal_queries
        )

        self.norm_q1 = nn.LayerNorm(
            hidden_dim
        )

        self.query_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q2 = nn.LayerNorm(
            hidden_dim
        )
        self.norm_mem = nn.LayerNorm(
            hidden_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(
            hidden_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                ff_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                ff_dim,
                hidden_dim,
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    @staticmethod
    def _causal_mask(
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                (
                    length,
                    length,
                ),
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
        qn = self.norm_q1(
            queries
        )

        q_mask = None
        if self.causal_queries:
            q_mask = self._causal_mask(
                qn.shape[1],
                qn.device,
            )

        self_out, _ = self.query_self_attn(
            qn,
            qn,
            qn,
            attn_mask=q_mask,
            need_weights=False,
        )

        queries = (
            queries
            + self.dropout(
                self_out
            )
        )

        qn = self.norm_q2(
            queries
        )
        mn = self.norm_mem(
            memory
        )

        cross_out, _ = self.cross_attn(
            qn,
            mn,
            mn,
            key_padding_mask=(
                ~memory_mask.bool()
            ),
            need_weights=False,
        )

        queries = (
            queries
            + self.dropout(
                cross_out
            )
        )

        queries = (
            queries
            + self.dropout(
                self.ffn(
                    self.norm_ff(
                        queries
                    )
                )
            )
        )

        return queries


class DecoderStyleTrajectoryExpert(nn.Module):
    """
    Decoder-style Action Expert.

    VLM memory stays external.
    10 trajectory queries are processed by a decoder-style stack and
    cross-attend to the VLM memory at every layer.

    Attention direction is independently controlled only on trajectory-query
    self-attention.
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
        causal_queries: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.input_dim = int(
            input_dim
        )
        self.hidden_dim = int(
            hidden_dim
        )
        self.num_steps = int(
            num_steps
        )
        self.causal_queries = bool(
            causal_queries
        )
        self.gradient_checkpointing = bool(
            gradient_checkpointing
        )

        self.memory_projector = KVMemoryProjector(
            input_dim,
            hidden_dim,
        )

        self.trajectory_queries = nn.Parameter(
            torch.randn(
                1,
                self.num_steps,
                hidden_dim,
            ) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                DecoderStyleBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    causal_queries=causal_queries,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.output = TrajectoryOutputHead(
            hidden_dim
        )

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
            if (
                self.training
                and self.gradient_checkpointing
            ):
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

        return self.output(
            queries
        )


# =============================================================================
# FACTORY
# =============================================================================

def parse_architecture(
    architecture: str,
) -> Tuple[str, str]:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture={architecture!r}; "
            f"choices={ARCHITECTURES}"
        )

    structure, direction = architecture.split(
        "_",
        1,
    )

    if structure not in (
        "encoder",
        "decoder",
    ):
        raise ValueError(
            structure
        )

    if direction not in (
        "bidirectional",
        "causal",
    ):
        raise ValueError(
            direction
        )

    return (
        structure,
        direction,
    )


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
    structure, direction = parse_architecture(
        architecture
    )

    causal_queries = (
        direction == "causal"
    )

    common = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        causal_queries=causal_queries,
        gradient_checkpointing=gradient_checkpointing,
    )

    if structure == "encoder":
        return EncoderStyleTrajectoryExpert(
            **common
        )

    return DecoderStyleTrajectoryExpert(
        **common
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
