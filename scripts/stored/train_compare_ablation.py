#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the two branches for the ~0.5B capacity-control Alpamayo-style ablation.

Research question
-----------------
Was the previous negative CoC result caused by an undersized Action Expert?

Old experiment:
  H512 / 3 layers / 8 heads / FF2048

This experiment:
  H1536 / 13 layers / 12 heads / FF6144
  ~= 0.499B when input_dim=2048

CRITICAL CONTROL
----------------
The Action Expert formulation is intentionally NOT changed:
  - decoder + cross-attention
  - causal query self-attention
  - direct waypoint regression
  - same loss
  - same paired train/val split
  - same optimizer / LR / scheduler
  - same random seed
  - same effective batch size = 8

Only capacity is scaled. Physical batch size is reduced for VRAM:
  old: 4 x accum 2 = 8
  new: 1 x accum 8 = 8

Training precision:
  FP32 master parameters + BF16 autocast
  This keeps optimizer/master weights in FP32 while reducing activation VRAM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from action_model_compare import (
    build_action_expert,
    count_trainable_parameters,
    trajectory_loss,
    trajectory_metrics_np,
)


# =============================================================================
# PATHS / CONFIG
# =============================================================================

TRAJ_ONLY_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpertAlpamayo/traj_only_kv_cache"
)
REASONING_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache"
)

# DO NOT overwrite the previous small-model result.
MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_capacity"
)

BRANCHES = ("traj_only", "coc_reasoning")
SEED = 20260823

# ~0.499B @ input_dim=2048 with the ORIGINAL decoder-cross architecture.
ACTION_HIDDEN_DIM = 1536
ACTION_NUM_LAYERS = 13
ACTION_NUM_HEADS = 12
ACTION_FF_DIM = 6144
ACTION_DROPOUT = 0.1
ACTION_NUM_STEPS = 10

# Keep training protocol matched to the old experiment as far as possible.
DEFAULT_EPOCHS = 50

# Old: 4 x accum 2 = effective 8.
# New: 1 x accum 8 = effective 8, only for VRAM.
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8

# Keep these identical to the old experiment to isolate model capacity.
DEFAULT_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-2
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_PATIENCE = 10
MIN_DELTA_ADE = 1.0e-4

HEADING_LOSS_WEIGHT = 0.5
GRAD_CLIP_NORM = 1.0


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# PAIRED MANIFESTS
# =============================================================================

@dataclass(frozen=True)
class PairEntry:
    id: str
    clip: str
    traj_cache_file: str
    reasoning_cache_file: str


def load_paired_manifest(
    split: str,
    traj_root: Path,
    reasoning_root: Path,
) -> List[PairEntry]:
    traj_path = traj_root / split / "manifest.jsonl"
    reasoning_path = reasoning_root / split / "manifest.jsonl"

    for p in (traj_path, reasoning_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    traj_manifest = read_jsonl(traj_path)
    reasoning_manifest = read_jsonl(reasoning_path)

    traj_by_id = {
        str(x["id"]): x
        for x in traj_manifest
    }
    reason_by_id = {
        str(x["id"]): x
        for x in reasoning_manifest
    }

    traj_ids = set(traj_by_id)
    reason_ids = set(reason_by_id)

    if traj_ids != reason_ids:
        only_traj = sorted(traj_ids - reason_ids)
        only_reason = sorted(reason_ids - traj_ids)

        raise RuntimeError(
            f"Paired cache IDs differ for split={split}: "
            f"traj_only_only={len(only_traj)}, "
            f"reasoning_only={len(only_reason)}. "
            f"Examples traj={only_traj[:3]}, "
            f"reasoning={only_reason[:3]}"
        )

    # Preserve reasoning manifest order.
    pairs: List[PairEntry] = []

    for reason_entry in reasoning_manifest:
        sample_id = str(reason_entry["id"])
        traj_entry = traj_by_id[sample_id]

        clip_a = str(traj_entry.get("clip", ""))
        clip_b = str(reason_entry.get("clip", ""))

        if clip_a and clip_b and clip_a != clip_b:
            raise RuntimeError(
                f"Clip mismatch id={sample_id}: "
                f"traj={clip_a}, reasoning={clip_b}"
            )

        # IMPORTANT:
        # manifest 안의 예전 절대경로는 사용하지 않고,
        # 현재 cache root + split + filename으로 경로를 재구성한다.
        traj_cache_file = (
            traj_root
            / split
            / Path(traj_entry["cache_file"]).name
        )

        reasoning_cache_file = (
            reasoning_root
            / split
            / Path(reason_entry["cache_file"]).name
        )

        if not traj_cache_file.is_file():
            raise FileNotFoundError(
                f"Missing traj_only cache "
                f"id={sample_id}: {traj_cache_file}"
            )

        if not reasoning_cache_file.is_file():
            raise FileNotFoundError(
                f"Missing reasoning cache "
                f"id={sample_id}: {reasoning_cache_file}"
            )

        pairs.append(
            PairEntry(
                id=sample_id,
                clip=clip_a or clip_b,
                traj_cache_file=str(
                    traj_cache_file
                ),
                reasoning_cache_file=str(
                    reasoning_cache_file
                ),
            )
        )

    return pairs


def validate_split_leakage(
    train_pairs: Sequence[PairEntry],
    val_pairs: Sequence[PairEntry],
) -> None:
    train_ids = {
        x.id
        for x in train_pairs
    }
    val_ids = {
        x.id
        for x in val_pairs
    }

    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(
            f"Train/Val ID leakage: "
            f"{list(overlap)[:5]}"
        )

    train_clips = {
        x.clip
        for x in train_pairs
        if x.clip
    }
    val_clips = {
        x.clip
        for x in val_pairs
        if x.clip
    }

    clip_overlap = train_clips & val_clips
    if clip_overlap:
        raise RuntimeError(
            f"Train/Val clip leakage: "
            f"{list(clip_overlap)[:5]}"
        )


# =============================================================================
# DATASET
# =============================================================================

class PairedBranchDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairEntry],
        branch: str,
    ):
        if branch not in BRANCHES:
            raise ValueError(branch)

        self.pairs = list(pairs)
        self.branch = branch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        entry = self.pairs[index]

        if self.branch == "traj_only":
            cache = torch.load(
                entry.traj_cache_file,
                map_location="cpu",
                weights_only=False,
            )

            memory = cache["observation_kv"]
            segment_ids = torch.zeros(
                memory.shape[0],
                dtype=torch.long,
            )
            trajectory = cache["trajectory"].float()

        else:
            cache = torch.load(
                entry.reasoning_cache_file,
                map_location="cpu",
                weights_only=False,
            )

            direct = cache["direct_kv"]
            delta = cache["reasoning_delta_kv"]

            if direct.ndim != 2 or delta.ndim != 2:
                raise RuntimeError(
                    f"Bad reasoning KV rank for id={entry.id}"
                )

            if direct.shape[-1] != delta.shape[-1]:
                raise RuntimeError(
                    f"Reasoning KV dim mismatch for id={entry.id}"
                )

            memory = torch.cat(
                [direct, delta],
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

            trajectory = cache["trajectory"].float()

        if memory.ndim != 2:
            raise RuntimeError(
                f"memory must be [T,D], "
                f"id={entry.id}, "
                f"got={tuple(memory.shape)}"
            )

        if tuple(trajectory.shape) != (10, 3):
            raise RuntimeError(
                f"trajectory must be [10,3], "
                f"id={entry.id}, "
                f"got={tuple(trajectory.shape)}"
            )

        return {
            "id": entry.id,
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
        int(item["memory"].shape[0])
        for item in items
    )
    dim = int(
        items[0]["memory"].shape[1]
    )
    dtype = items[0]["memory"].dtype

    memory = torch.zeros(
        (batch_size, max_len, dim),
        dtype=dtype,
    )
    memory_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
    )
    segment_ids = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    trajectories = []
    ids = []

    for i, item in enumerate(items):
        x = item["memory"]

        if int(x.shape[1]) != dim:
            raise RuntimeError(
                f"KV dim mismatch inside batch: "
                f"{x.shape[1]} vs {dim}"
            )

        length = int(x.shape[0])
        memory[i, :length] = x
        memory_mask[i, :length] = True
        segment_ids[i, :length] = item["segment_ids"]

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
    pairs: Sequence[PairEntry],
    branch: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple[PairedBranchDataset, DataLoader]:
    dataset = PairedBranchDataset(
        pairs,
        branch=branch,
    )

    generator = None
    if shuffle:
        generator = torch.Generator()
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
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    # Keep source cache in FP32; CUDA BF16 autocast handles GEMMs/attention.
    memory = batch["memory"].to(
        device,
        non_blocking=True,
    ).float()

    memory_mask = batch["memory_mask"].to(
        device,
        non_blocking=True,
    )

    segment_ids = batch["segment_ids"].to(
        device,
        non_blocking=True,
    )

    gt = batch["trajectory"].to(
        device,
        non_blocking=True,
    ).float()

    return {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }, gt


# =============================================================================
# EVAL
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
            pred = model(**inputs)

        loss, _, _ = trajectory_loss(
            pred,
            gt,
            heading_weight=HEADING_LOSS_WEIGHT,
        )

        bs = int(gt.shape[0])

        loss_sum += (
            float(loss.item())
            * bs
        )
        sample_count += bs

        preds.append(
            pred.float().cpu().numpy()
        )
        gts.append(
            gt.float().cpu().numpy()
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
# TRAIN ONE BRANCH
# =============================================================================

def infer_input_dim(
    train_pairs: Sequence[PairEntry],
) -> Tuple[int, int]:
    first = train_pairs[0]

    traj_cache = torch.load(
        first.traj_cache_file,
        map_location="cpu",
        weights_only=False,
    )
    reason_cache = torch.load(
        first.reasoning_cache_file,
        map_location="cpu",
        weights_only=False,
    )

    traj_dim = int(
        traj_cache["observation_kv"].shape[-1]
    )
    reason_dim = int(
        reason_cache["direct_kv"].shape[-1]
    )

    if traj_dim != reason_dim:
        raise RuntimeError(
            f"Baseline/Reasoning KV dimension mismatch: "
            f"{traj_dim} vs {reason_dim}"
        )

    return traj_dim, reason_dim


def train_one_branch(
    branch: str,
    train_pairs: Sequence[PairEntry],
    val_pairs: Sequence[PairEntry],
    input_dim: int,
    cache_signature: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(branch)

    # Critical fairness control:
    # identical initialization for traj_only / coc_reasoning.
    set_seed(args.seed)

    train_dataset, train_loader = build_loader(
        train_pairs,
        branch=branch,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    val_dataset, val_loader = build_loader(
        val_pairs,
        branch=branch,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    # Keep FP32 master parameters.
    # Forward/backward activations use BF16 autocast.
    model = build_action_expert(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=ACTION_NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    trainable_params = count_trainable_parameters(
        model
    )

    # foreach=False reduces optimizer peak-memory overhead for ~0.5B params.
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

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_dir = (
        args.model_root
        / branch
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

    if history_path.exists():
        history_path.unlink()

    config = {
        "experiment": "alpamayo_style_reasoning_ablation_capacity_05b",
        "research_question": "does scaling only the direct Transformer Action Expert reverse the previous negative CoC result?",
        "branch": branch,
        "fixed_action_expert": "decoder_cross_direct_regression",
        "capacity_control": True,

        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_steps": ACTION_NUM_STEPS,
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),

        "trainable_params": int(trainable_params),
        "trainable_params_B": float(
            trainable_params
            / 1.0e9
        ),

        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),

        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "effective_batch_size": int(
            args.batch_size
            * args.grad_accum
        ),

        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),

        "heading_loss_weight": HEADING_LOSS_WEIGHT,
        "selection_metric": "val_ade_m",
        "seed": int(args.seed),

        "precision": "fp32_master_bf16_autocast",
        "gradient_checkpointing": bool(
            not args.no_gradient_checkpointing
        ),
        "optimizer_foreach": False,

        "cache_signature": cache_signature,
    }

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 116)
    print(
        f"TRAIN {branch} | ~0.5B CAPACITY CONTROL"
    )
    print("=" * 116)

    print(
        "Action Expert      : "
        "SAME decoder + causal self-attn + cross-attn + direct regression"
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
        "Params             :",
        f"{trainable_params:,} "
        f"({trainable_params/1e9:.6f}B)",
    )
    print(
        "Batch              :",
        f"{args.batch_size} "
        f"x accum {args.grad_accum} "
        f"= {args.batch_size * args.grad_accum}",
    )
    print(
        "Optimizer          :",
        f"AdamW lr={args.lr:.2e} "
        f"wd={args.weight_decay:.2e}",
    )
    print(
        "Precision          :",
        "FP32 master + BF16 autocast",
    )
    print(
        "Grad checkpointing :",
        not args.no_gradient_checkpointing,
    )
    print(
        "Best selection     :",
        "val ADE",
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
                heading_weight=HEADING_LOSS_WEIGHT,
            )

            if not torch.isfinite(
                raw_loss
            ):
                raise RuntimeError(
                    f"Non-finite loss: "
                    f"branch={branch}, "
                    f"epoch={epoch}, "
                    f"micro={micro_idx}, "
                    f"loss={raw_loss.item()}"
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
                    raw_loss.detach().item()
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
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
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

        history = {
            "branch": branch,
            "epoch": epoch,
            "global_step": global_step,

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
                scheduler.get_last_lr()[0]
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
            f"[{branch}] "
            f"e={epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} "
            f"val={val_metrics['loss']:.6f} "
            f"ADE={val_metrics['ade_m']:.4f}m "
            f"FDE={val_metrics['fde_m']:.4f}m "
            f"Heading={val_metrics['heading_mae_rad']:.4f}rad "
            f"lr={scheduler.get_last_lr()[0]:.3e} "
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
                "experiment": "alpamayo_style_reasoning_ablation_capacity_05b",
                "branch": branch,
                "fixed_action_expert": "decoder_cross_direct_regression",
                "capacity_control": True,

                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": int(input_dim),

                "action_config": {
                    "hidden_dim": int(
                        args.hidden_dim
                    ),
                    "num_steps": ACTION_NUM_STEPS,
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
                "cache_signature": cache_signature,
                "val_metrics": val_metrics,
            }

            torch.save(
                checkpoint_payload,
                best_path,
            )

            print(
                f"[{branch}] BEST -> "
                f"epoch={epoch} "
                f"val_ADE={best_ade:.6f} "
                f"saved={best_path}"
            )

        else:
            no_improve += 1

            print(
                f"[{branch}] "
                f"no ADE improvement "
                f"{no_improve}/{args.patience} "
                f"(best={best_ade:.6f})"
            )

        if no_improve >= args.patience:
            print(
                f"[{branch}] EARLY STOP | "
                f"best_epoch={best_epoch} "
                f"best_ADE={best_ade:.6f}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(
            f"No best checkpoint produced: "
            f"{best_path}"
        )

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    result = {
        "branch": branch,
        "best_epoch": int(
            best_checkpoint["epoch"]
        ),
        "trainable_params": int(
            trainable_params
        ),
        **{
            f"val_{k}": float(v)
            for k, v
            in best_checkpoint[
                "val_metrics"
            ].items()
        },
        "checkpoint": str(
            best_path
        ),
    }

    del (
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
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--traj-only-cache-root",
        type=Path,
        default=TRAJ_ONLY_CACHE_ROOT,
    )
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
        default=list(BRANCHES),
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
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

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Selected GPU does not support BF16. "
            "Use RTX 3080 Ti (GPU 0)."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for root in (
        args.traj_only_cache_root,
        args.reasoning_cache_root,
    ):
        if not root.exists():
            raise FileNotFoundError(
                root
            )

    train_pairs = load_paired_manifest(
        "train",
        args.traj_only_cache_root,
        args.reasoning_cache_root,
    )
    val_pairs = load_paired_manifest(
        "val",
        args.traj_only_cache_root,
        args.reasoning_cache_root,
    )

    validate_split_leakage(
        train_pairs,
        val_pairs,
    )

    input_dim, _ = infer_input_dim(
        train_pairs
    )

    traj_meta = (
        args.traj_only_cache_root
        / "meta.json"
    )
    reason_meta = (
        args.reasoning_cache_root
        / "meta.json"
    )

    sig_payload = {
        "traj_only_meta": (
            sha256_file(traj_meta)
            if traj_meta.is_file()
            else None
        ),
        "reasoning_meta": (
            sha256_file(reason_meta)
            if reason_meta.is_file()
            else None
        ),
        "train_ids": [
            x.id
            for x in train_pairs
        ],
        "val_ids": [
            x.id
            for x in val_pairs
        ],
    }

    cache_signature = hashlib.sha256(
        json.dumps(
            sig_payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    args.model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 116)
    print(
        "ALPAMAYO-STYLE ~0.5B CAPACITY-CONTROL "
        "TWO-WAY ACTION EXPERT TRAINING"
    )
    print("=" * 116)

    print(
        "GPU                 :",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )
    print(
        "Trajectory-only KV  :",
        args.traj_only_cache_root,
    )
    print(
        "CoC reasoning KV    :",
        args.reasoning_cache_root,
    )
    print(
        "Model root          :",
        args.model_root,
    )
    print(
        "Paired Train / Val  :",
        len(train_pairs),
        "/",
        len(val_pairs),
    )
    print(
        "KV input dim        :",
        input_dim,
    )
    print(
        "Action Expert       :",
        "decoder + causal self-attn + cross-attn "
        "+ direct regression (UNCHANGED)",
    )
    print(
        "Capacity            :",
        f"H={args.hidden_dim} "
        f"L={args.num_layers} "
        f"heads={args.num_heads} "
        f"FF={args.ff_dim}",
    )
    print(
        "Precision           :",
        "FP32 master + BF16 autocast",
    )
    print(
        "Best checkpoint     :",
        "minimum validation ADE",
    )
    print(
        "Branches            :",
        ", ".join(args.branches),
    )

    results = []

    for branch in args.branches:
        result = train_one_branch(
            branch=branch,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            input_dim=input_dim,
            cache_signature=cache_signature,
            device=device,
            args=args,
        )

        results.append(
            result
        )

        (
            args.model_root
            / "summary.json"
        ).write_text(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n" + "=" * 116)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 116)

    print(
        f"{'BRANCH':<22}"
        f"{'PARAMS(B)':>12}"
        f"{'VAL LOSS':>12}"
        f"{'ADE':>10}"
        f"{'FDE':>10}"
        f"{'HEAD':>10}"
    )
    print("-" * 78)

    for r in results:
        print(
            f"{r['branch']:<22}"
            f"{r['trainable_params']/1e9:>12.3f}"
            f"{r['val_loss']:>12.5f}"
            f"{r['val_ade_m']:>10.4f}"
            f"{r['val_fde_m']:>10.4f}"
            f"{r['val_heading_mae_rad']:>10.4f}"
        )

    if {
        r["branch"]
        for r in results
    } == set(BRANCHES):
        by = {
            r["branch"]: r
            for r in results
        }

        base = by[
            "traj_only"
        ]["val_ade_m"]

        coc = by[
            "coc_reasoning"
        ]["val_ade_m"]

        improvement = (
            100.0
            * (base - coc)
            / base
            if base != 0
            else float("nan")
        )

        print(
            "\nCoC relative ADE improvement "
            f"vs Traj-only: "
            f"{improvement:+.2f}%"
        )

    print(
        "Saved models:",
        args.model_root,
    )


if __name__ == "__main__":
    main()
