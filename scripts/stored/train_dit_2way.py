#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the controlled Flow-Matching / DiT v2 Action Experts using ONE
Reasoning_VLM generation-time KV cache.

PURPOSE
-------
Fix the previous Flow traj_only / coc_reasoning conditioning mismatch.

OLD Flow comparison:
    traj_only:
        VLM_Baseline observation_kv

    coc_reasoning:
        Reasoning_VLM direct_kv + reasoning_delta_kv

NEW controlled Flow comparison:
    traj_only:
        Reasoning_VLM direct_kv

    coc_reasoning:
        Reasoning_VLM direct_kv + reasoning_delta_kv

Therefore the only branch-level conditioning difference is:
    reasoning_delta_kv absent vs present.

COMMON INPUT CONDITION
----------------------
Both branches come from the SAME Reasoning_VLM cache generated with:
    - 3 cameras
    - speed_mps included in the VLM prompt
    - route command included
    - same action8 reasoning prompt
    - BF16
    - SDPA

IMPORTANT:
    speed_mps is NOT injected into the Flow Action Expert as a separate scalar.
    It is already represented inside the VLM prompt-boundary KV.

FLOW v2
-------
    - ~0.5B decoder-cross conditioning backbone
    - normalized noisy trajectory x_t -> Linear(3 -> H)
    - scalar t -> Fourier features
    - non-causal joint trajectory-token self-attention
    - linear conditional Flow Matching
        x_t = (1-t) * x0 + t * x1
        v*  = x1 - x0
    - uniform timestep sampling
    - TRAIN-only per-waypoint [10,3] normalizer
    - Euler integration, default 10 solver steps

Required local module:
    action_model_flow_dit.py

The module must be the v2 implementation containing:
    TrajectoryNormalizer
    build_flow_dit
    count_trainable_parameters
    euler_sample
    flow_matching_batch
    linear_flow_oracle_sanity_check
    trajectory_metrics_np
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from scripts.stored.action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    count_trainable_parameters,
    euler_sample,
    flow_matching_batch,
    linear_flow_oracle_sanity_check,
    trajectory_metrics_np,
)


# =============================================================================
# PATHS
# =============================================================================

REASONING_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache"
)

MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/"
    "alpamayo_flow_dit_05b_reasoning_vlm_controlled_v2"
)

EXPECTED_VLM = "/home/lhh/lab/models/vlm/Reasoning_VLM"
EXPECTED_CACHE_VERSION = "action8_generation_kv_v1"


# =============================================================================
# CONFIG
# =============================================================================

BRANCHES = (
    "traj_only",
    "coc_reasoning",
)

SEED = 20260823
VAL_NOISE_SEED = 20260824
VAL_FLOW_SEED = 20260840

ACTION_HIDDEN_DIM = 1536
ACTION_NUM_LAYERS = 13
ACTION_NUM_HEADS = 12
ACTION_FF_DIM = 6144
ACTION_DROPOUT = 0.1
ACTION_NUM_STEPS = 10

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-2
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_PATIENCE = 10

DEFAULT_SOLVER_STEPS = 10
DEFAULT_TIMESTEP_SAMPLER = "uniform"

GRAD_CLIP_NORM = 1.0
MIN_DELTA_ADE = 1.0e-4
NORMALIZER_STD_FLOOR = 1.0e-3

TRANSFORMER_REFERENCE_PARAMS_2048 = 499_051_011


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"JSON parse failed: {path}:{line_no}"
                ) from exc

    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


# =============================================================================
# CACHE VALIDATION
# =============================================================================

def validate_reasoning_cache_root(root: Path) -> Dict[str, Any]:
    """
    Refuse to train if the cache is not the intended Reasoning_VLM cache.
    """

    meta_path = root / "meta.json"

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    meta = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    cache_version = str(
        meta.get("cache_version", "")
    )

    model_path = str(
        meta.get("model_path", "")
    )

    if cache_version != EXPECTED_CACHE_VERSION:
        raise RuntimeError(
            "Unexpected cache version.\n"
            f"Expected: {EXPECTED_CACHE_VERSION}\n"
            f"Found   : {cache_version}"
        )

    if model_path != EXPECTED_VLM:
        raise RuntimeError(
            "This is not the expected Reasoning_VLM cache.\n"
            f"Expected: {EXPECTED_VLM}\n"
            f"Found   : {model_path}"
        )

    representation = meta.get(
        "representation",
        {},
    )

    direct_description = str(
        representation.get("direct", "")
    )

    if (
        direct_description
        and "prompt boundary" not in direct_description.lower()
    ):
        raise RuntimeError(
            "Cache metadata does not describe direct_kv "
            "as prompt-boundary KV."
        )

    return meta


# =============================================================================
# MANIFEST
# =============================================================================

def load_manifest(
    root: Path,
    split: str,
) -> List[Dict[str, Any]]:

    path = root / split / "manifest.jsonl"

    rows = read_jsonl(path)

    if not rows:
        raise RuntimeError(
            f"Empty manifest: {path}"
        )

    ids = [
        str(row["id"])
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"Duplicate IDs in {path}"
        )

    fixed_rows: List[Dict[str, Any]] = []

    for row in rows:
        row = dict(row)

        old_path = Path(
            str(row["cache_file"])
        )

        if old_path.is_file():
            resolved = old_path
        else:
            candidate = (
                root
                / split
                / old_path.name
            )

            if candidate.is_file():
                resolved = candidate
            else:
                raise FileNotFoundError(
                    "Could not resolve cache file.\n"
                    f"Manifest path : {old_path}\n"
                    f"Fallback path : {candidate}"
                )

        row["cache_file"] = str(
            resolved
        )

        fixed_rows.append(
            row
        )

    return fixed_rows


def validate_split_leakage(
    train_manifest: Sequence[Dict[str, Any]],
    val_manifest: Sequence[Dict[str, Any]],
) -> None:

    train_ids = {
        str(x["id"])
        for x in train_manifest
    }

    val_ids = {
        str(x["id"])
        for x in val_manifest
    }

    id_overlap = (
        train_ids
        & val_ids
    )

    if id_overlap:
        raise RuntimeError(
            "Train/Val ID leakage: "
            f"{sorted(id_overlap)[:5]}"
        )

    train_clips = {
        str(x.get("clip", ""))
        for x in train_manifest
        if x.get("clip")
    }

    val_clips = {
        str(x.get("clip", ""))
        for x in val_manifest
        if x.get("clip")
    }

    clip_overlap = (
        train_clips
        & val_clips
    )

    if clip_overlap:
        raise RuntimeError(
            "Train/Val clip leakage: "
            f"{sorted(clip_overlap)[:5]}"
        )


def validate_cache_samples(
    manifest: Sequence[Dict[str, Any]],
    max_samples: int = 32,
) -> None:
    """
    Verify that cached records contain the fields needed for the controlled
    comparison, including the input-condition metadata.
    """

    take = min(
        len(manifest),
        max_samples,
    )

    if take == 0:
        raise RuntimeError(
            "Empty manifest"
        )

    for entry in manifest[:take]:

        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        required = (
            "direct_kv",
            "reasoning_delta_kv",
            "trajectory",
            "speed_mps",
            "command",
        )

        missing = [
            key
            for key in required
            if key not in cache
        ]

        if missing:
            raise RuntimeError(
                f"Cache id={entry['id']} "
                f"is missing required fields: {missing}"
            )

        direct = cache[
            "direct_kv"
        ]

        delta = cache[
            "reasoning_delta_kv"
        ]

        trajectory = cache[
            "trajectory"
        ]

        if direct.ndim != 2:
            raise RuntimeError(
                f"direct_kv must be [T,D], "
                f"id={entry['id']}, "
                f"got={tuple(direct.shape)}"
            )

        if delta.ndim != 2:
            raise RuntimeError(
                f"reasoning_delta_kv must be [R,D], "
                f"id={entry['id']}, "
                f"got={tuple(delta.shape)}"
            )

        if (
            direct.shape[-1]
            != delta.shape[-1]
        ):
            raise RuntimeError(
                f"direct/delta KV dim mismatch "
                f"id={entry['id']}: "
                f"{direct.shape[-1]} vs "
                f"{delta.shape[-1]}"
            )

        if tuple(
            trajectory.shape
        ) != (ACTION_NUM_STEPS, 3):
            raise RuntimeError(
                f"trajectory must be "
                f"[{ACTION_NUM_STEPS},3], "
                f"id={entry['id']}, "
                f"got={tuple(trajectory.shape)}"
            )

        speed = float(
            cache["speed_mps"]
        )

        if not math.isfinite(
            speed
        ):
            raise RuntimeError(
                f"Non-finite speed_mps "
                f"id={entry['id']}: "
                f"{speed}"
            )

        command = str(
            cache["command"]
        ).strip()

        if not command:
            raise RuntimeError(
                f"Empty route command "
                f"id={entry['id']}"
            )


# =============================================================================
# DATASET
# =============================================================================

class UnifiedReasoningKVFlowDataset(Dataset):
    """
    Both branches use the exact same Reasoning_VLM cache.

    traj_only:
        direct_kv

    coc_reasoning:
        concat(direct_kv, reasoning_delta_kv)
    """

    def __init__(
        self,
        manifest: Sequence[Dict[str, Any]],
        branch: str,
    ):
        if branch not in BRANCHES:
            raise ValueError(
                branch
            )

        self.manifest = list(
            manifest
        )

        self.branch = branch

    def __len__(self) -> int:
        return len(
            self.manifest
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        entry = self.manifest[
            index
        ]

        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        if "direct_kv" not in cache:
            raise RuntimeError(
                f"direct_kv missing "
                f"id={entry['id']}"
            )

        if "trajectory" not in cache:
            raise RuntimeError(
                f"trajectory missing "
                f"id={entry['id']}"
            )

        direct = cache[
            "direct_kv"
        ]

        trajectory = torch.as_tensor(
            cache["trajectory"],
            dtype=torch.float32,
        )

        if direct.ndim != 2:
            raise RuntimeError(
                f"direct_kv must be [T,D], "
                f"id={entry['id']}, "
                f"got={tuple(direct.shape)}"
            )

        if tuple(
            trajectory.shape
        ) != (ACTION_NUM_STEPS, 3):
            raise RuntimeError(
                f"trajectory must be "
                f"[{ACTION_NUM_STEPS},3], "
                f"id={entry['id']}, "
                f"got={tuple(trajectory.shape)}"
            )

        if self.branch == "traj_only":

            memory = direct

            segment_ids = torch.zeros(
                direct.shape[0],
                dtype=torch.long,
            )

        else:

            if (
                "reasoning_delta_kv"
                not in cache
            ):
                raise RuntimeError(
                    f"reasoning_delta_kv missing "
                    f"id={entry['id']}"
                )

            delta = cache[
                "reasoning_delta_kv"
            ]

            if delta.ndim != 2:
                raise RuntimeError(
                    f"reasoning_delta_kv "
                    f"must be [R,D], "
                    f"id={entry['id']}, "
                    f"got={tuple(delta.shape)}"
                )

            if (
                direct.shape[-1]
                != delta.shape[-1]
            ):
                raise RuntimeError(
                    f"direct/delta KV dim mismatch "
                    f"id={entry['id']}"
                )

            if delta.shape[0] <= 0:
                raise RuntimeError(
                    f"Empty reasoning_delta_kv "
                    f"id={entry['id']}"
                )

            memory = torch.cat(
                [
                    direct,
                    delta,
                ],
                dim=0,
            )

            segment_ids = torch.cat(
                [
                    torch.zeros(
                        direct.shape[0],
                        dtype=torch.long,
                    ),
                    torch.ones(
                        delta.shape[0],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )

        return {
            "id": str(
                entry["id"]
            ),
            "memory": memory,
            "segment_ids": segment_ids,
            "trajectory": trajectory,
        }


def collate_kv(
    items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    if not items:
        raise ValueError(
            "Empty batch"
        )

    batch_size = len(
        items
    )

    max_len = max(
        int(
            item["memory"].shape[0]
        )
        for item in items
    )

    dim = int(
        items[0]["memory"].shape[1]
    )

    dtype = items[0][
        "memory"
    ].dtype

    memory = torch.zeros(
        (
            batch_size,
            max_len,
            dim,
        ),
        dtype=dtype,
    )

    memory_mask = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.bool,
    )

    segment_ids = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.long,
    )

    trajectories = []
    ids = []

    for i, item in enumerate(
        items
    ):

        x = item[
            "memory"
        ]

        if int(
            x.shape[1]
        ) != dim:
            raise RuntimeError(
                "KV dim mismatch "
                f"{x.shape[1]} vs "
                f"{dim}"
            )

        length = int(
            x.shape[0]
        )

        memory[
            i,
            :length,
        ] = x

        memory_mask[
            i,
            :length,
        ] = True

        segment_ids[
            i,
            :length,
        ] = item[
            "segment_ids"
        ]

        trajectories.append(
            item[
                "trajectory"
            ]
        )

        ids.append(
            item["id"]
        )

    return {
        "id": ids,
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
        "trajectory": torch.stack(
            trajectories,
            dim=0,
        ),
    }


def build_loader(
    manifest: Sequence[Dict[str, Any]],
    branch: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple[
    UnifiedReasoningKVFlowDataset,
    DataLoader,
]:

    dataset = (
        UnifiedReasoningKVFlowDataset(
            manifest=manifest,
            branch=branch,
        )
    )

    generator = None

    if shuffle:
        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            seed
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
        generator=generator,
    )

    return (
        dataset,
        loader,
    )


def infer_input_dim(
    manifest: Sequence[Dict[str, Any]],
) -> int:

    cache = torch.load(
        manifest[0][
            "cache_file"
        ],
        map_location="cpu",
        weights_only=False,
    )

    direct = cache[
        "direct_kv"
    ]

    if direct.ndim != 2:
        raise RuntimeError(
            f"direct_kv must be [T,D], "
            f"got={tuple(direct.shape)}"
        )

    return int(
        direct.shape[-1]
    )


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    torch.Tensor,
]:

    memory = batch[
        "memory"
    ].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    memory_mask = batch[
        "memory_mask"
    ].to(
        device=device,
        non_blocking=True,
    )

    segment_ids = batch[
        "segment_ids"
    ].to(
        device=device,
        non_blocking=True,
    )

    gt = batch[
        "trajectory"
    ].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    return {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }, gt


# =============================================================================
# NORMALIZER
# =============================================================================

def compute_normalizer(
    train_manifest: Sequence[Dict[str, Any]],
) -> TrajectoryNormalizer:
    """
    TRAIN-only per-waypoint [10,3] statistics.
    """

    if not train_manifest:
        raise RuntimeError(
            "Cannot compute normalizer "
            "from empty training set"
        )

    sums = torch.zeros(
        ACTION_NUM_STEPS,
        3,
        dtype=torch.float64,
    )

    sq_sums = torch.zeros_like(
        sums
    )

    sample_count = 0

    for entry in train_manifest:

        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        if "trajectory" not in cache:
            raise RuntimeError(
                f"trajectory missing "
                f"id={entry['id']}"
            )

        traj = torch.as_tensor(
            cache["trajectory"],
            dtype=torch.float64,
        )

        if tuple(
            traj.shape
        ) != (ACTION_NUM_STEPS, 3):
            raise RuntimeError(
                f"trajectory must be "
                f"[{ACTION_NUM_STEPS},3], "
                f"id={entry['id']}, "
                f"got={tuple(traj.shape)}"
            )

        if not torch.isfinite(
            traj
        ).all():
            raise RuntimeError(
                f"Non-finite trajectory "
                f"id={entry['id']}"
            )

        sums += traj

        sq_sums += (
            traj
            * traj
        )

        sample_count += 1

    mean = (
        sums
        / float(
            sample_count
        )
    )

    var = (
        sq_sums
        / float(
            sample_count
        )
    ) - (
        mean
        * mean
    )

    std = torch.sqrt(
        var.clamp_min(
            1.0e-12
        )
    ).clamp_min(
        NORMALIZER_STD_FLOOR
    )

    return TrajectoryNormalizer(
        mean=mean.float(),
        std=std.float(),
    )


def print_normalizer(
    normalizer: TrajectoryNormalizer,
) -> None:

    mean = (
        normalizer
        .mean
        .detach()
        .cpu()
    )

    std = (
        normalizer
        .std
        .detach()
        .cpu()
    )

    print(
        "Normalizer mode :",
        normalizer.mode,
    )

    print(
        "Per-waypoint normalizer [x, y, yaw]"
    )

    for i in range(
        mean.shape[0]
    ):
        print(
            f"  wp{i:02d} "
            f"mean=["
            f"{mean[i,0]: .5f}, "
            f"{mean[i,1]: .5f}, "
            f"{mean[i,2]: .5f}] "
            f"std=["
            f"{std[i,0]: .5f}, "
            f"{std[i,1]: .5f}, "
            f"{std[i,2]: .5f}]"
        )


def verify_normalizer_on_training_subset(
    train_manifest: Sequence[Dict[str, Any]],
    normalizer: TrajectoryNormalizer,
    max_samples: int = 256,
) -> None:

    take = min(
        len(train_manifest),
        max_samples,
    )

    if take == 0:
        return

    values = []

    for entry in train_manifest[:take]:

        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        traj = torch.as_tensor(
            cache["trajectory"],
            dtype=torch.float32,
        )

        values.append(
            normalizer.normalize(
                traj
            )
        )

    z = torch.stack(
        values,
        dim=0,
    )

    if not torch.isfinite(
        z
    ).all():
        raise RuntimeError(
            "Normalized trajectory "
            "contains non-finite values"
        )

    abs_max = float(
        z.abs()
        .max()
        .item()
    )

    rms = float(
        torch.sqrt(
            torch.mean(
                z * z
            )
        ).item()
    )

    print(
        "Normalizer check:",
        f"subset={take}",
        f"rms={rms:.4f}",
        f"max_abs={abs_max:.4f}",
    )

    if abs_max > 50.0:
        print(
            "[WARN] Very large normalized "
            "trajectory value detected."
        )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    timestep_sampler: str,
) -> Dict[str, float]:

    model.eval()

    norm = normalizer.to(
        device
    )

    # Same fixed validation noise for both branches.
    noise_rng = torch.Generator(
        device="cpu"
    )

    noise_rng.manual_seed(
        VAL_NOISE_SEED
    )

    # Same fixed validation t/x0 stream for FM loss.
    flow_rng = torch.Generator(
        device="cpu"
    )

    flow_rng.manual_seed(
        VAL_FLOW_SEED
    )

    preds = []
    gts = []

    flow_loss_sum = 0.0
    sample_count = 0

    for batch in loader:

        inputs, gt = move_batch(
            batch,
            device,
        )

        gt_norm = norm.normalize(
            gt
        )

        x_t, t, v_target, _ = (
            flow_matching_batch(
                gt_normalized=gt_norm,
                rng=flow_rng,
                timestep_sampler=(
                    timestep_sampler
                ),
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            v_pred = model(
                x_t=x_t,
                t=t,
                **inputs,
            )

        fm_loss = F.mse_loss(
            v_pred.float(),
            v_target.float(),
        )

        bs = int(
            gt.shape[0]
        )

        flow_loss_sum += (
            float(
                fm_loss.item()
            )
            * bs
        )

        sample_count += bs

        pred = euler_sample(
            model=model,
            memory=inputs[
                "memory"
            ],
            memory_mask=inputs[
                "memory_mask"
            ],
            segment_ids=inputs[
                "segment_ids"
            ],
            normalizer=norm,
            solver_steps=(
                solver_steps
            ),
            rng=noise_rng,
        )

        preds.append(
            pred.float()
            .cpu()
            .numpy()
        )

        gts.append(
            gt.float()
            .cpu()
            .numpy()
        )

    if sample_count == 0:
        raise RuntimeError(
            "Empty validation loader"
        )

    pred_np = np.concatenate(
        preds,
        axis=0,
    )

    gt_np = np.concatenate(
        gts,
        axis=0,
    )

    metrics = (
        trajectory_metrics_np(
            pred_np,
            gt_np,
        )
    )

    metrics[
        "flow_mse"
    ] = (
        flow_loss_sum
        / float(
            sample_count
        )
    )

    return metrics


# =============================================================================
# TRAIN ONE BRANCH
# =============================================================================

def train_one_branch(
    branch: str,
    train_manifest: Sequence[Dict[str, Any]],
    val_manifest: Sequence[Dict[str, Any]],
    input_dim: int,
    cache_signature: str,
    normalizer: TrajectoryNormalizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:

    if branch not in BRANCHES:
        raise ValueError(
            branch
        )

    # Same model initialization and DataLoader shuffle order for both branches.
    set_seed(
        args.seed
    )

    train_dataset, train_loader = build_loader(
        manifest=train_manifest,
        branch=branch,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    val_dataset, val_loader = build_loader(
        manifest=val_manifest,
        branch=branch,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = build_flow_dit(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=ACTION_NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        gradient_checkpointing=(
            not args.no_gradient_checkpointing
        ),
    ).to(
        device=device,
        dtype=torch.float32,
    )

    trainable_params = (
        count_trainable_parameters(
            model
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / args.grad_accum
    )

    total_steps = max(
        1,
        steps_per_epoch
        * args.epochs,
    )

    warmup_steps = max(
        1,
        int(
            total_steps
            * args.warmup_ratio
        ),
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_steps
            ),
        )
    )

    model_dir = (
        args.model_root
        / branch
    )

    if (
        model_dir.exists()
        and args.overwrite
    ):
        shutil.rmtree(
            model_dir
        )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        model_dir
        / "best.pt"
    )

    history_path = (
        model_dir
        / "history.jsonl"
    )

    config_path = (
        model_dir
        / "config.json"
    )

    if (
        best_path.exists()
        and not args.overwrite
    ):
        raise RuntimeError(
            f"{best_path} already exists. "
            "Use --overwrite."
        )

    if history_path.exists():
        history_path.unlink()

    if branch == "traj_only":
        memory_source = (
            "Reasoning_VLM direct_kv"
        )
        reasoning_delta_used = False
    else:
        memory_source = (
            "Reasoning_VLM direct_kv + "
            "reasoning_delta_kv"
        )
        reasoning_delta_used = True

    ref_delta = None

    if input_dim == 2048:
        ref_delta = (
            trainable_params
            - TRANSFORMER_REFERENCE_PARAMS_2048
        )

    config = {
        "experiment": (
            "reasoning_vlm_controlled_flow_dit_2way_v2"
        ),
        "version": 2,
        "branch": branch,

        "vlm": EXPECTED_VLM,
        "cache_root": str(
            args.reasoning_cache_root
        ),
        "cache_version": (
            EXPECTED_CACHE_VERSION
        ),

        "common_vlm_input": (
            "3 cameras + speed_mps + route command "
            "+ same action8 reasoning prompt"
        ),
        "speed_conditioning": (
            "included inside VLM prompt/KV; "
            "not separately injected into Action Expert"
        ),

        "memory_source": (
            memory_source
        ),
        "reasoning_delta_used": (
            reasoning_delta_used
        ),

        "conditioning": (
            "KVMemoryProjector + "
            "decoder cross-attention"
        ),
        "objective": (
            "linear_flow_matching_velocity_mse"
        ),
        "flow_path": (
            "x_t=(1-t)*x0+t*x1; "
            "target=x1-x0"
        ),
        "action_input": (
            "raw_normalized_xt_linear_plus_fourier_t"
        ),
        "normalization": (
            "per_waypoint_xyz_train_only"
        ),
        "self_attention": (
            "non_causal_joint_action_denoising"
        ),

        "input_dim": int(
            input_dim
        ),
        "hidden_dim": int(
            args.hidden_dim
        ),
        "num_steps": (
            ACTION_NUM_STEPS
        ),
        "num_layers": int(
            args.num_layers
        ),
        "num_heads": int(
            args.num_heads
        ),
        "ff_dim": int(
            args.ff_dim
        ),
        "dropout": float(
            args.dropout
        ),

        "trainable_params": int(
            trainable_params
        ),
        "trainable_params_B": float(
            trainable_params
            / 1e9
        ),

        "transformer_reference_params_2048": (
            TRANSFORMER_REFERENCE_PARAMS_2048
            if input_dim == 2048
            else None
        ),
        "parameter_delta_vs_transformer": (
            int(ref_delta)
            if ref_delta is not None
            else None
        ),

        "train_samples": len(
            train_dataset
        ),
        "val_samples": len(
            val_dataset
        ),

        "batch_size": int(
            args.batch_size
        ),
        "grad_accum": int(
            args.grad_accum
        ),
        "effective_batch_size": int(
            args.batch_size
            * args.grad_accum
        ),

        "epochs": int(
            args.epochs
        ),
        "lr": float(
            args.lr
        ),
        "weight_decay": float(
            args.weight_decay
        ),
        "warmup_ratio": float(
            args.warmup_ratio
        ),
        "patience": int(
            args.patience
        ),

        "solver_steps": int(
            args.solver_steps
        ),
        "timestep_sampler": str(
            args.timestep_sampler
        ),

        "selection_metric": (
            "val_ade_m"
        ),
        "seed": int(
            args.seed
        ),

        "precision": (
            "fp32_master_bf16_autocast"
        ),

        "gradient_checkpointing": bool(
            not args.no_gradient_checkpointing
        ),

        "normalizer": (
            normalizer.to_dict()
        ),
        "normalizer_mode": (
            normalizer.mode
        ),

        "cache_signature": (
            cache_signature
        ),

        "controlled_difference": (
            "reasoning_delta_kv absent vs present"
        ),
    }

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"CONTROLLED FLOW-DiT v2 | {branch}"
    )

    print(
        "=" * 120
    )

    print(
        "VLM                :",
        EXPECTED_VLM,
    )

    print(
        "Common VLM input   :",
        "3 cameras + speed + route command",
    )

    print(
        "Memory             :",
        memory_source,
    )

    print(
        "Reasoning delta    :",
        (
            "USED"
            if reasoning_delta_used
            else "NOT USED"
        ),
    )

    print(
        "Train / Val        :",
        len(train_dataset),
        "/",
        len(val_dataset),
    )

    print(
        "KV input dim       :",
        input_dim,
    )

    print(
        "Architecture       :",
        f"H={args.hidden_dim} "
        f"L={args.num_layers} "
        f"heads={args.num_heads} "
        f"FF={args.ff_dim}",
    )

    print(
        "Action attention   :",
        "non-causal joint denoising",
    )

    print(
        "Action input       :",
        "normalized x_t Linear + Fourier(t)",
    )

    print(
        "Normalizer         :",
        normalizer.mode,
    )

    print(
        "Params             :",
        f"{trainable_params:,} "
        f"({trainable_params/1e9:.6f}B)",
    )

    if ref_delta is not None:
        print(
            "vs Transformer     :",
            f"{ref_delta:+,} params "
            f"("
            f"{100.0 * ref_delta / TRANSFORMER_REFERENCE_PARAMS_2048:+.3f}%"
            f")",
        )

    print(
        "Batch              :",
        f"{args.batch_size} "
        f"x accum {args.grad_accum} "
        f"= "
        f"{args.batch_size * args.grad_accum}",
    )

    print(
        "Timestep sampler   :",
        args.timestep_sampler,
    )

    print(
        "Euler steps        :",
        args.solver_steps,
    )

    print(
        "Precision          :",
        "FP32 master + BF16 autocast",
    )

    print(
        "Best selection     :",
        "validation ADE",
    )

    print(
        "Checkpoint         :",
        best_path,
    )

    best_ade = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0

    norm = normalizer.to(
        device
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        torch.cuda.reset_peak_memory_stats(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_start = (
            time.perf_counter()
        )

        # Identical x0/t stream at each epoch for both branches.
        flow_rng = torch.Generator(
            device="cpu"
        )

        flow_rng.manual_seed(
            args.seed
            + epoch * 1009
        )

        train_loss_sum = 0.0
        train_samples = 0

        for micro_idx, batch in enumerate(
            train_loader,
            1,
        ):

            inputs, gt = move_batch(
                batch,
                device,
            )

            gt_norm = norm.normalize(
                gt
            )

            if not torch.isfinite(
                gt_norm
            ).all():
                raise RuntimeError(
                    f"Non-finite normalized GT "
                    f"branch={branch} "
                    f"epoch={epoch} "
                    f"micro={micro_idx}"
                )

            x_t, t, v_target, _ = (
                flow_matching_batch(
                    gt_normalized=gt_norm,
                    rng=flow_rng,
                    timestep_sampler=(
                        args.timestep_sampler
                    ),
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                v_pred = model(
                    x_t=x_t,
                    t=t,
                    **inputs,
                )

            loss = F.mse_loss(
                v_pred.float(),
                v_target.float(),
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"branch={branch} "
                    f"epoch={epoch} "
                    f"micro={micro_idx}: "
                    f"{loss.item()}"
                )

            (
                loss
                / args.grad_accum
            ).backward()

            bs = int(
                gt.shape[0]
            )

            train_loss_sum += (
                float(
                    loss.detach()
                    .item()
                )
                * bs
            )

            train_samples += bs

            do_step = (
                micro_idx
                % args.grad_accum
                == 0
                or micro_idx
                == len(train_loader)
            )

            if do_step:

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP_NORM,
                    )
                )

                if not torch.isfinite(
                    torch.as_tensor(
                        grad_norm
                    )
                ):
                    raise RuntimeError(
                        f"Non-finite grad norm "
                        f"branch={branch}"
                    )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

        train_loss = (
            train_loss_sum
            / max(
                1,
                train_samples,
            )
        )

        val = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            normalizer=normalizer,
            solver_steps=(
                args.solver_steps
            ),
            timestep_sampler=(
                args.timestep_sampler
            ),
        )

        peak_vram_gb = (
            torch.cuda
            .max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

        elapsed_min = (
            time.perf_counter()
            - epoch_start
        ) / 60.0

        row = {
            "branch": branch,
            "epoch": epoch,
            "global_step": (
                global_step
            ),
            "train_flow_mse": float(
                train_loss
            ),
            "val_flow_mse": float(
                val["flow_mse"]
            ),
            "val_ade_m": float(
                val["ade_m"]
            ),
            "val_fde_m": float(
                val["fde_m"]
            ),
            "val_heading_mae_rad": float(
                val[
                    "heading_mae_rad"
                ]
            ),
            "lr": float(
                scheduler
                .get_last_lr()[0]
            ),
            "peak_vram_gb": float(
                peak_vram_gb
            ),
            "epoch_minutes": float(
                elapsed_min
            ),
        }

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"[{branch}] "
            f"e={epoch:03d}/{args.epochs:03d} "
            f"trainFM={train_loss:.6f} "
            f"valFM={val['flow_mse']:.6f} "
            f"ADE={val['ade_m']:.4f}m "
            f"FDE={val['fde_m']:.4f}m "
            f"Heading="
            f"{val['heading_mae_rad']:.4f}rad "
            f"lr="
            f"{scheduler.get_last_lr()[0]:.3e} "
            f"VRAM={peak_vram_gb:.2f}GB "
            f"time={elapsed_min:.2f}m"
        )

        improved = (
            val["ade_m"]
            < best_ade
            - MIN_DELTA_ADE
        )

        if improved:

            best_ade = float(
                val["ade_m"]
            )

            best_epoch = int(
                epoch
            )

            no_improve = 0

            checkpoint = {
                "experiment": (
                    "reasoning_vlm_controlled_flow_dit_2way_v2"
                ),
                "version": 2,
                "branch": branch,

                "vlm": (
                    EXPECTED_VLM
                ),
                "memory_source": (
                    memory_source
                ),
                "reasoning_delta_used": (
                    reasoning_delta_used
                ),

                "epoch": epoch,

                "model_state_dict": (
                    model.state_dict()
                ),

                "input_dim": int(
                    input_dim
                ),

                "trainable_params": int(
                    trainable_params
                ),

                "cache_signature": (
                    cache_signature
                ),

                "normalizer": (
                    normalizer.to_dict()
                ),

                "action_config": {
                    "hidden_dim": int(
                        args.hidden_dim
                    ),
                    "num_steps": (
                        ACTION_NUM_STEPS
                    ),
                    "num_layers": int(
                        args.num_layers
                    ),
                    "num_heads": int(
                        args.num_heads
                    ),
                    "ff_dim": int(
                        args.ff_dim
                    ),
                    "dropout": float(
                        args.dropout
                    ),
                    "gradient_checkpointing": bool(
                        not args.no_gradient_checkpointing
                    ),
                    "solver_steps": int(
                        args.solver_steps
                    ),
                    "timestep_sampler": str(
                        args.timestep_sampler
                    ),
                    "conditioning": (
                        "KVMemoryProjector + "
                        "decoder cross-attention"
                    ),
                    "self_attention": (
                        "non_causal"
                    ),
                    "action_input": (
                        "raw_normalized_xt_linear_plus_fourier_t"
                    ),
                    "normalization": (
                        "per_waypoint_xyz_train_only"
                    ),
                    "flow_path": (
                        "x_t=(1-t)*x0+t*x1; "
                        "target=x1-x0"
                    ),
                },

                "val_metrics": {
                    key: float(
                        value
                    )
                    for key, value
                    in val.items()
                },
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"[{branch}] "
                f"BEST -> "
                f"epoch={epoch} "
                f"val_ADE={best_ade:.6f} "
                f"saved={best_path}"
            )

        else:

            no_improve += 1

            print(
                f"[{branch}] "
                f"no ADE improvement "
                f"{no_improve}/"
                f"{args.patience} "
                f"(best={best_ade:.6f})"
            )

        if (
            no_improve
            >= args.patience
        ):

            print(
                f"[{branch}] "
                f"EARLY STOP | "
                f"best_epoch={best_epoch} "
                f"best_ADE={best_ade:.6f}"
            )

            break

    if not best_path.is_file():
        raise RuntimeError(
            f"No best checkpoint: "
            f"{best_path}"
        )

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    result = {
        "branch": branch,
        "vlm": EXPECTED_VLM,
        "memory_source": (
            memory_source
        ),
        "reasoning_delta_used": (
            reasoning_delta_used
        ),
        "best_epoch": int(
            best_checkpoint[
                "epoch"
            ]
        ),
        "trainable_params": int(
            trainable_params
        ),
        **{
            f"val_{key}": float(
                value
            )
            for key, value
            in best_checkpoint[
                "val_metrics"
            ].items()
        },
        "checkpoint": str(
            best_path
        ),
    }

    del (
        best_checkpoint,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )

    cleanup_cuda()

    return result


# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(
    root: Path,
    results: Sequence[Dict[str, Any]],
) -> None:

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        root
        / "summary.json"
    ).write_text(
        json.dumps(
            list(results),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 120,
        "Reasoning_VLM CONTROLLED FLOW-DiT v2 | 2-WAY SUMMARY",
        "=" * 120,
        f"VLM          : {EXPECTED_VLM}",
        (
            "Common input : "
            "3 cameras + speed_mps + route command "
            "+ same action8 prompt"
        ),
        (
            "Difference   : "
            "reasoning_delta_kv absent vs present"
        ),
        "",
        (
            f"{'BRANCH':<22}"
            f"{'PARAMS(B)':>12}"
            f"{'BEST_E':>9}"
            f"{'VAL FM':>12}"
            f"{'ADE':>11}"
            f"{'FDE':>11}"
            f"{'HEAD':>11}"
        ),
        "-" * 120,
    ]

    for result in results:

        lines.append(
            f"{result['branch']:<22}"
            f"{result['trainable_params']/1e9:>12.3f}"
            f"{result['best_epoch']:>9d}"
            f"{result['val_flow_mse']:>12.5f}"
            f"{result['val_ade_m']:>11.4f}"
            f"{result['val_fde_m']:>11.4f}"
            f"{result['val_heading_mae_rad']:>11.4f}"
        )

    by_branch = {
        result["branch"]: result
        for result in results
    }

    if (
        "traj_only" in by_branch
        and "coc_reasoning" in by_branch
    ):

        traj_ade = float(
            by_branch[
                "traj_only"
            ][
                "val_ade_m"
            ]
        )

        coc_ade = float(
            by_branch[
                "coc_reasoning"
            ][
                "val_ade_m"
            ]
        )

        if traj_ade != 0.0:
            rel_improvement = (
                100.0
                * (
                    traj_ade
                    - coc_ade
                )
                / traj_ade
            )
        else:
            rel_improvement = (
                float("nan")
            )

        lines.extend(
            [
                "",
                (
                    "CoC relative ADE improvement "
                    f"vs Traj-only: "
                    f"{rel_improvement:+.2f}%"
                ),
            ]
        )

    (
        root
        / "summary.txt"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reasoning-cache-root",
        type=Path,
        default=REASONING_CACHE_ROOT,
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--branches",
        nargs="+",
        choices=BRANCHES,
        default=list(
            BRANCHES
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--solver-steps",
        type=int,
        default=DEFAULT_SOLVER_STEPS,
    )

    parser.add_argument(
        "--timestep-sampler",
        choices=(
            "uniform",
            "beta",
        ),
        default=(
            DEFAULT_TIMESTEP_SAMPLER
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=ACTION_HIDDEN_DIM,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=ACTION_NUM_LAYERS,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=ACTION_NUM_HEADS,
    )

    parser.add_argument(
        "--ff-dim",
        type=int,
        default=ACTION_FF_DIM,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=ACTION_DROPOUT,
    )

    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is required. "
            "Use RTX 3080 Ti / GPU 0."
        )

    if (
        args.batch_size < 1
        or args.grad_accum < 1
    ):
        raise ValueError(
            "batch-size and grad-accum "
            "must be >= 1"
        )

    if args.epochs < 1:
        raise ValueError(
            "epochs must be >= 1"
        )

    if args.patience < 1:
        raise ValueError(
            "patience must be >= 1"
        )

    if args.solver_steps < 1:
        raise ValueError(
            "solver-steps must be >= 1"
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be > 0"
        )

    if not (
        0.0
        <= args.warmup_ratio
        < 1.0
    ):
        raise ValueError(
            "warmup-ratio must satisfy "
            "0 <= ratio < 1"
        )

    torch.cuda.set_device(
        args.gpu_id
    )

    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(
        args.seed
    )

    # Verify Flow sign / Euler direction before expensive training.
    oracle_error = (
        linear_flow_oracle_sanity_check(
            seed=args.seed,
            solver_steps=10,
        )
    )

    if oracle_error > 1.0e-5:
        raise RuntimeError(
            "Linear-flow oracle sanity "
            "check failed: "
            f"max_error="
            f"{oracle_error:.8e}"
        )

    meta = (
        validate_reasoning_cache_root(
            args.reasoning_cache_root
        )
    )

    train_manifest = load_manifest(
        args.reasoning_cache_root,
        "train",
    )

    val_manifest = load_manifest(
        args.reasoning_cache_root,
        "val",
    )

    validate_split_leakage(
        train_manifest,
        val_manifest,
    )

    # Check both direct and reasoning delta plus speed/route metadata.
    validate_cache_samples(
        train_manifest,
        max_samples=32,
    )

    validate_cache_samples(
        val_manifest,
        max_samples=32,
    )

    input_dim = infer_input_dim(
        train_manifest
    )

    val_input_dim = infer_input_dim(
        val_manifest
    )

    if (
        input_dim
        != val_input_dim
    ):
        raise RuntimeError(
            "Train/Val KV input_dim mismatch: "
            f"{input_dim} vs "
            f"{val_input_dim}"
        )

    normalizer = compute_normalizer(
        train_manifest
    )

    verify_normalizer_on_training_subset(
        train_manifest=train_manifest,
        normalizer=normalizer,
        max_samples=256,
    )

    meta_path = (
        args.reasoning_cache_root
        / "meta.json"
    )

    signature_payload = {
        "cache_meta_sha256": (
            sha256_file(
                meta_path
            )
        ),
        "train_ids": [
            str(x["id"])
            for x in train_manifest
        ],
        "val_ids": [
            str(x["id"])
            for x in val_manifest
        ],
        "branches": {
            "traj_only": (
                "direct_kv"
            ),
            "coc_reasoning": (
                "direct_kv+reasoning_delta_kv"
            ),
        },
        "normalizer": (
            normalizer.to_dict()
        ),
        "timestep_sampler": (
            args.timestep_sampler
        ),
        "solver_steps": int(
            args.solver_steps
        ),
    }

    cache_signature = (
        hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
            ).encode(
                "utf-8"
            )
        ).hexdigest()
    )

    print(
        "=" * 120
    )

    print(
        "Reasoning_VLM | CONTROLLED FLOW-DiT v2 | 2-WAY"
    )

    print(
        "=" * 120
    )

    print(
        "GPU          :",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )

    print(
        "VLM          :",
        meta["model_path"],
    )

    print(
        "Cache        :",
        args.reasoning_cache_root,
    )

    print(
        "Cache ver    :",
        meta["cache_version"],
    )

    print(
        "Common input :",
        "3 cameras + speed_mps + route command",
    )

    print(
        "Traj-only    :",
        "direct_kv ONLY",
    )

    print(
        "CoC          :",
        "direct_kv + reasoning_delta_kv",
    )

    print(
        "Train / Val  :",
        len(train_manifest),
        "/",
        len(val_manifest),
    )

    print(
        "KV dim       :",
        input_dim,
    )

    print(
        "Branches     :",
        ", ".join(
            args.branches
        ),
    )

    print(
        "Capacity     :",
        f"H={args.hidden_dim} "
        f"L={args.num_layers} "
        f"heads={args.num_heads} "
        f"FF={args.ff_dim}",
    )

    print(
        "Flow sampler :",
        args.timestep_sampler,
    )

    print(
        "Euler steps  :",
        args.solver_steps,
    )

    print(
        "Batch        :",
        f"{args.batch_size} "
        f"x accum "
        f"{args.grad_accum} "
        f"= "
        f"{args.batch_size * args.grad_accum}",
    )

    print(
        "Oracle error :",
        f"{oracle_error:.3e}",
    )

    print_normalizer(
        normalizer
    )

    args.model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for branch in args.branches:

        result = train_one_branch(
            branch=branch,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            input_dim=input_dim,
            cache_signature=(
                cache_signature
            ),
            normalizer=normalizer,
            device=device,
            args=args,
        )

        results.append(
            result
        )

        save_summary(
            args.model_root,
            results,
        )

    save_summary(
        args.model_root,
        results,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"{'BRANCH':<22}"
        f"{'PARAMS(B)':>12}"
        f"{'BEST_E':>9}"
        f"{'VAL FM':>12}"
        f"{'ADE':>11}"
        f"{'FDE':>11}"
        f"{'HEAD':>11}"
    )

    print(
        "-" * 120
    )

    for result in results:

        print(
            f"{result['branch']:<22}"
            f"{result['trainable_params']/1e9:>12.3f}"
            f"{result['best_epoch']:>9d}"
            f"{result['val_flow_mse']:>12.5f}"
            f"{result['val_ade_m']:>11.4f}"
            f"{result['val_fde_m']:>11.4f}"
            f"{result['val_heading_mae_rad']:>11.4f}"
        )

    by_branch = {
        result["branch"]: result
        for result in results
    }

    if (
        "traj_only" in by_branch
        and "coc_reasoning" in by_branch
    ):

        traj_ade = float(
            by_branch[
                "traj_only"
            ][
                "val_ade_m"
            ]
        )

        coc_ade = float(
            by_branch[
                "coc_reasoning"
            ][
                "val_ade_m"
            ]
        )

        rel = (
            100.0
            * (
                traj_ade
                - coc_ade
            )
            / traj_ade
            if traj_ade != 0.0
            else float("nan")
        )

        print()

        print(
            "CoC relative ADE improvement "
            "vs Traj-only:",
            f"{rel:+.2f}%",
        )

    print()

    print(
        "Completed models :",
        len(results),
    )

    print(
        "Summary          :",
        args.model_root
        / "summary.txt",
    )

    print(
        "Model root       :",
        args.model_root,
    )


if __name__ == "__main__":
    main()
