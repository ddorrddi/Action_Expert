#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the four traj_only Action Experts using Reasoning_VLM prompt-boundary KV.

IMPORTANT
---------
This fixes the previous traj_only / coc_reasoning VLM mismatch.

OLD comparison:
    traj_only:
        VLM_Baseline observation_kv

    coc_reasoning:
        Reasoning_VLM direct_kv + reasoning_delta_kv

NEW controlled comparison:
    traj_only:
        Reasoning_VLM direct_kv

    coc_reasoning:
        Reasoning_VLM direct_kv + reasoning_delta_kv

Therefore the only difference between traj_only and coc_reasoning becomes:
    reasoning_delta_kv absent vs present.

Existing Reasoning_VLM cache:
    /home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache

This cache was generated with:
    - Reasoning_VLM
    - speed_mps included
    - route command included
    - full action8 reasoning prompt
    - BF16
    - SDPA

Four traj_only architectures:
    encoder_self
    encoder_cross
    decoder_self
    decoder_cross
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
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from action_model_compare_8way import (
    ARCHITECTURES,
    build_action_expert,
    count_trainable_parameters,
    trajectory_loss,
    trajectory_metrics_np,
)


# =============================================================================
# PATHS
# =============================================================================

# This is the existing Reasoning_VLM generation-time KV cache.
#
# Each sample contains:
#   direct_kv
#   reasoning_delta_kv
#   trajectory
#
REASONING_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache"
)

# Keep these new controlled traj_only models separate from all old runs.
MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/"
    "alpamayo_ablation_05b_traj_only_reasoning_vlm"
)


# =============================================================================
# CONFIG
# =============================================================================

SEED = 20260823

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

MIN_DELTA_ADE = 1.0e-4
HEADING_LOSS_WEIGHT = 0.5
GRAD_CLIP_NORM = 1.0

EXPECTED_VLM = "/home/lhh/lab/models/vlm/Reasoning_VLM"
EXPECTED_CACHE_VERSION = "action8_generation_kv_v1"


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

    rows = []

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
    Make sure this is really the old Reasoning_VLM cache and not the
    VLM_Baseline traj-only cache or the newer Reasoning_VLM_1200 cache.
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

    if "prompt boundary" not in direct_description.lower():
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
        str(x["id"])
        for x in rows
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"Duplicate IDs in {path}"
        )

    fixed_rows = []

    for row in rows:
        row = dict(row)

        old_path = Path(
            str(row["cache_file"])
        )

        # -------------------------------------------------------------
        # 1) Original absolute path still exists
        # -------------------------------------------------------------
        if old_path.is_file():
            resolved = old_path

        else:
            # ---------------------------------------------------------
            # 2) Manifest contains stale absolute path.
            #    Recover using filename under the CURRENT cache root.
            #
            # OLD:
            # /home/lhh/lab/ActionExpert8/dataset/action_kv_cache/train/X.pt
            #
            # CURRENT:
            # /home/lhh/lab/Action_Expert/dataset/ActionExpert8/
            #     action_kv_cache/train/X.pt
            # ---------------------------------------------------------
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

        # Replace stale manifest path only in memory.
        row["cache_file"] = str(
            resolved
        )

        fixed_rows.append(row)

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

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            f"Train/Val ID leakage: "
            f"{sorted(overlap)[:5]}"
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
            f"Train/Val clip leakage: "
            f"{sorted(clip_overlap)[:5]}"
        )


# =============================================================================
# DATASET
# =============================================================================

class ReasoningVLMDirectKVDataset(Dataset):
    """
    traj_only dataset using ONLY Reasoning_VLM direct_kv.

    Does NOT use:
        observation_kv from VLM_Baseline

    Does NOT use:
        reasoning_delta_kv

    Memory:
        Reasoning_VLM prompt-boundary direct_kv
    """

    def __init__(
        self,
        manifest: Sequence[Dict[str, Any]],
    ):
        self.manifest = list(manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        entry = self.manifest[index]

        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        if "direct_kv" not in cache:
            raise RuntimeError(
                f"direct_kv missing id={entry['id']}"
            )

        if "trajectory" not in cache:
            raise RuntimeError(
                f"trajectory missing id={entry['id']}"
            )

        memory = cache[
            "direct_kv"
        ]

        trajectory = cache[
            "trajectory"
        ].float()

        if memory.ndim != 2:
            raise RuntimeError(
                f"direct_kv must be [T,D], "
                f"id={entry['id']}, "
                f"got={tuple(memory.shape)}"
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

        # traj_only contains only one segment.
        segment_ids = torch.zeros(
            memory.shape[0],
            dtype=torch.long,
        )

        return {
            "id": str(entry["id"]),
            "memory": memory,
            "segment_ids": segment_ids,
            "trajectory": trajectory,
        }


def collate_kv(
    items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    if not items:
        raise ValueError("Empty batch")

    batch_size = len(items)

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

    for i, item in enumerate(items):

        x = item["memory"]

        if int(x.shape[1]) != dim:
            raise RuntimeError(
                "KV dim mismatch "
                f"{x.shape[1]} vs {dim}"
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
            item["trajectory"]
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
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple[
    ReasoningVLMDirectKVDataset,
    DataLoader,
]:

    dataset = (
        ReasoningVLMDirectKVDataset(
            manifest
        )
    )

    generator = None

    if shuffle:
        generator = (
            torch.Generator()
        )
        generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
        generator=generator,
    )

    return dataset, loader


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
        device,
        dtype=torch.float32,
        non_blocking=True,
    )

    memory_mask = batch[
        "memory_mask"
    ].to(
        device,
        non_blocking=True,
    )

    segment_ids = batch[
        "segment_ids"
    ].to(
        device,
        non_blocking=True,
    )

    gt = batch[
        "trajectory"
    ].to(
        device,
        dtype=torch.float32,
        non_blocking=True,
    )

    inputs = {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }

    return inputs, gt


def infer_input_dim(
    manifest: Sequence[Dict[str, Any]],
) -> int:

    cache = torch.load(
        manifest[0]["cache_file"],
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


# =============================================================================
# EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    loss_sum = 0.0
    sample_count = 0

    preds = []
    gts = []

    for batch in loader:

        inputs, gt = move_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            pred = model(
                **inputs
            )

        loss, _, _ = trajectory_loss(
            pred,
            gt,
            heading_weight=(
                HEADING_LOSS_WEIGHT
            ),
        )

        bs = int(
            gt.shape[0]
        )

        loss_sum += (
            float(loss.item())
            * bs
        )

        sample_count += bs

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

    metrics = trajectory_metrics_np(
        pred_np,
        gt_np,
    )

    metrics["loss"] = (
        loss_sum
        / sample_count
    )

    return metrics


# =============================================================================
# TRAIN ONE ARCHITECTURE
# =============================================================================

def train_architecture(
    architecture: str,
    train_manifest: Sequence[Dict[str, Any]],
    val_manifest: Sequence[Dict[str, Any]],
    input_dim: int,
    cache_signature: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:

    if architecture not in ARCHITECTURES:
        raise ValueError(
            architecture
        )

    set_seed(
        args.seed
    )

    train_dataset, train_loader = build_loader(
        train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    val_dataset, val_loader = build_loader(
        val_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = build_action_expert(
        architecture=architecture,
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

    condition_name = (
        f"traj_only_{architecture}"
    )

    model_dir = (
        args.model_root
        / condition_name
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

    config = {
        "experiment": (
            "reasoning_vlm_controlled_traj_only_4way"
        ),
        "condition": condition_name,
        "branch": "traj_only",
        "architecture": architecture,

        "memory_source": (
            "Reasoning_VLM direct_kv"
        ),
        "memory_definition": (
            "last Qwen text-layer full K/V "
            "at prompt boundary"
        ),
        "reasoning_delta_used": False,

        "vlm": EXPECTED_VLM,
        "cache_root": str(
            args.reasoning_cache_root
        ),
        "cache_version": (
            EXPECTED_CACHE_VERSION
        ),

        "prompt": (
            "old action8 prompt: "
            "3 cameras + current speed + "
            "route command + reasoning instruction"
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

        "heading_loss_weight": (
            HEADING_LOSS_WEIGHT
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

        "parameter_matching_across_architectures": False,

        "cache_signature": (
            cache_signature
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
        f"TRAIN | {condition_name}"
    )

    print(
        "=" * 120
    )

    print(
        "VLM                :",
        EXPECTED_VLM,
    )

    print(
        "Memory             :",
        "Reasoning_VLM direct_kv ONLY",
    )

    print(
        "Reasoning delta    :",
        "NOT USED",
    )

    print(
        "Architecture       :",
        architecture,
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
        "Capacity           :",
        f"H={args.hidden_dim} "
        f"L={args.num_layers} "
        f"heads={args.num_heads} "
        f"FF={args.ff_dim}",
    )

    print(
        "Params             :",
        f"{trainable_params:,} "
        f"({trainable_params/1e9:.6f}B)",
    )

    print(
        "Batch              :",
        f"{args.batch_size} "
        f"x accum {args.grad_accum} "
        f"= "
        f"{args.batch_size * args.grad_accum}",
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

        train_loss_sum = 0.0
        train_samples = 0

        epoch_start = (
            time.perf_counter()
        )

        for micro_idx, batch in enumerate(
            train_loader,
            1,
        ):

            inputs, gt = move_batch(
                batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                pred = model(
                    **inputs
                )

            raw_loss, _, _ = trajectory_loss(
                pred,
                gt,
                heading_weight=(
                    HEADING_LOSS_WEIGHT
                ),
            )

            if not torch.isfinite(
                raw_loss
            ):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"condition={condition_name} "
                    f"epoch={epoch} "
                    f"micro={micro_idx}"
                )

            (
                raw_loss
                / args.grad_accum
            ).backward()

            bs = int(
                gt.shape[0]
            )

            train_loss_sum += (
                float(
                    raw_loss
                    .detach()
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
                        f"{condition_name}"
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

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        elapsed_min = (
            time.perf_counter()
            - epoch_start
        ) / 60.0

        peak_vram_gb = (
            torch.cuda
            .max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

        history = {
            "condition": condition_name,
            "epoch": epoch,
            "global_step": (
                global_step
            ),
            "train_loss": float(
                train_loss
            ),
            "val_loss": float(
                val_metrics["loss"]
            ),
            "val_ade_m": float(
                val_metrics["ade_m"]
            ),
            "val_fde_m": float(
                val_metrics["fde_m"]
            ),
            "val_heading_mae_rad": float(
                val_metrics[
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
                    history,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"[{condition_name}] "
            f"e={epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"ADE={val_metrics['ade_m']:.4f}m "
            f"FDE={val_metrics['fde_m']:.4f}m "
            f"Heading="
            f"{val_metrics['heading_mae_rad']:.4f}rad "
            f"lr="
            f"{scheduler.get_last_lr()[0]:.3e} "
            f"VRAM={peak_vram_gb:.2f}GB "
            f"time={elapsed_min:.2f}m"
        )

        improved = (
            val_metrics["ade_m"]
            < best_ade
            - MIN_DELTA_ADE
        )

        if improved:

            best_ade = float(
                val_metrics["ade_m"]
            )

            best_epoch = int(
                epoch
            )

            no_improve = 0

            checkpoint_payload = {
                "experiment": (
                    "reasoning_vlm_controlled_traj_only_4way"
                ),
                "condition": (
                    condition_name
                ),
                "branch": (
                    "traj_only"
                ),
                "architecture": (
                    architecture
                ),

                "memory_source": (
                    "Reasoning_VLM direct_kv"
                ),
                "reasoning_delta_used": (
                    False
                ),
                "vlm": EXPECTED_VLM,

                "epoch": epoch,

                "model_state_dict": (
                    model.state_dict()
                ),

                "input_dim": int(
                    input_dim
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
                },

                "trainable_params": int(
                    trainable_params
                ),

                "cache_signature": (
                    cache_signature
                ),

                "val_metrics": (
                    val_metrics
                ),
            }

            torch.save(
                checkpoint_payload,
                best_path,
            )

            print(
                f"[{condition_name}] "
                f"BEST -> "
                f"epoch={epoch} "
                f"val_ADE={best_ade:.6f} "
                f"saved={best_path}"
            )

        else:

            no_improve += 1

            print(
                f"[{condition_name}] "
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
                f"[{condition_name}] "
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
        "condition": (
            condition_name
        ),
        "branch": "traj_only",
        "architecture": (
            architecture
        ),
        "vlm": EXPECTED_VLM,
        "memory_source": (
            "Reasoning_VLM direct_kv"
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

    json_path = (
        root
        / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            list(results),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 120,
        "Reasoning_VLM DIRECT-KV | TRAJ-ONLY 4-WAY SUMMARY",
        "=" * 120,
        f"VLM          : {EXPECTED_VLM}",
        f"Memory       : Reasoning_VLM direct_kv only",
        f"Reasoning KV : NOT USED",
        "",
        (
            f"{'CONDITION':<36}"
            f"{'PARAMS(B)':>12}"
            f"{'BEST_E':>9}"
            f"{'ADE':>11}"
            f"{'FDE':>11}"
            f"{'HEAD':>11}"
        ),
        "-" * 120,
    ]

    for result in results:

        lines.append(
            f"{result['condition']:<36}"
            f"{result['trainable_params']/1e9:>12.3f}"
            f"{result['best_epoch']:>9d}"
            f"{result['val_ade_m']:>11.4f}"
            f"{result['val_fde_m']:>11.4f}"
            f"{result['val_heading_mae_rad']:>11.4f}"
        )

    (
        root
        / "summary.txt"
    ).write_text(
        "\n".join(lines)
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
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(
            ARCHITECTURES
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

    # -------------------------------------------------------------------------
    # Validate that this really is the Reasoning_VLM cache.
    # -------------------------------------------------------------------------

    meta = validate_reasoning_cache_root(
        args.reasoning_cache_root
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
        "memory_source": (
            "direct_kv"
        ),
    }

    cache_signature = (
        hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    print(
        "=" * 120
    )

    print(
        "Reasoning_VLM | CONTROLLED TRAJ-ONLY 4-WAY"
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
        "Memory       :",
        "direct_kv ONLY",
    )

    print(
        "Reason KV    :",
        "NOT USED",
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
        "Architectures:",
        ", ".join(
            args.architectures
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
        "Batch        :",
        f"{args.batch_size} "
        f"x accum "
        f"{args.grad_accum} "
        f"= "
        f"{args.batch_size * args.grad_accum}",
    )

    args.model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # -------------------------------------------------------------------------
    # Train exactly the four traj_only architectures sequentially.
    # -------------------------------------------------------------------------

    for architecture in args.architectures:

        result = train_architecture(
            architecture=architecture,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            input_dim=input_dim,
            cache_signature=cache_signature,
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
        f"{'CONDITION':<36}"
        f"{'PARAMS(B)':>12}"
        f"{'BEST_E':>9}"
        f"{'ADE':>11}"
        f"{'FDE':>11}"
        f"{'HEAD':>11}"
    )

    print(
        "-" * 120
    )

    for result in results:

        print(
            f"{result['condition']:<36}"
            f"{result['trainable_params']/1e9:>12.3f}"
            f"{result['best_epoch']:>9d}"
            f"{result['val_ade_m']:>11.4f}"
            f"{result['val_fde_m']:>11.4f}"
            f"{result['val_heading_mae_rad']:>11.4f}"
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