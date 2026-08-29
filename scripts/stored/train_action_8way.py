#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train all 8 Action Expert models from cached nuReasoning Part1 VLM K/V.

Eight checkpoints are saved independently:
  direct_encoder_self
  direct_encoder_cross
  direct_decoder_self
  direct_decoder_cross
  reasoning_encoder_self
  reasoning_encoder_cross
  reasoning_decoder_self
  reasoning_decoder_cross

Test split is NEVER loaded by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from action_model_8way import (
    MODEL_NAMES,
    build_action_expert,
    count_trainable_parameters,
    parse_model_name,
    trajectory_loss,
    trajectory_metrics_np,
)


# =============================================================================
# PATHS
# =============================================================================

CACHE_ROOT = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache")
MODEL_ROOT = Path("/home/lhh/lab/models/action_expert/8way")


# =============================================================================
# TRAIN CONFIG
# =============================================================================

SEED = 20260823

ACTION_HIDDEN_DIM = 512
ACTION_NUM_LAYERS = 3
ACTION_NUM_HEADS = 8
ACTION_FF_DIM = 2048
ACTION_DROPOUT = 0.1
ACTION_NUM_STEPS = 10

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2  # effective batch = 8
DEFAULT_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-2
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_PATIENCE = 10
MIN_DELTA = 1.0e-5

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
        return [json.loads(line) for line in f if line.strip()]


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
# DATASET
# =============================================================================

class CachedKVDataset(Dataset):
    def __init__(self, manifest: Sequence[Dict[str, Any]], stage: str):
        if stage not in ("direct", "reasoning"):
            raise ValueError(stage)
        self.manifest = list(manifest)
        self.stage = stage

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.manifest[index]
        cache = torch.load(
            entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        direct = cache["direct_kv"]
        if direct.ndim != 2:
            raise RuntimeError(
                f"direct_kv must be [T,D], got {tuple(direct.shape)} for {entry['id']}"
            )

        if self.stage == "direct":
            memory = direct
            segment_ids = torch.zeros(memory.shape[0], dtype=torch.long)
        else:
            delta = cache["reasoning_delta_kv"]
            if delta.ndim != 2 or delta.shape[-1] != direct.shape[-1]:
                raise RuntimeError(
                    f"Bad reasoning_delta_kv for {entry['id']}: {tuple(delta.shape)}"
                )
            memory = torch.cat([direct, delta], dim=0)
            segment_ids = torch.cat(
                [
                    torch.zeros(direct.shape[0], dtype=torch.long),
                    torch.ones(delta.shape[0], dtype=torch.long),
                ],
                dim=0,
            )

        trajectory = cache["trajectory"].float()
        if tuple(trajectory.shape) != (10, 3):
            raise RuntimeError(
                f"trajectory must be [10,3], got {tuple(trajectory.shape)} for {entry['id']}"
            )

        return {
            "id": str(entry["id"]),
            "memory": memory,
            "segment_ids": segment_ids,
            "trajectory": trajectory,
        }


def collate_kv(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("Empty batch")

    batch_size = len(items)
    max_len = max(int(item["memory"].shape[0]) for item in items)
    dim = int(items[0]["memory"].shape[1])
    dtype = items[0]["memory"].dtype

    memory = torch.zeros((batch_size, max_len, dim), dtype=dtype)
    memory_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    segment_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    trajectories = []
    ids = []

    for i, item in enumerate(items):
        x = item["memory"]
        if int(x.shape[1]) != dim:
            raise RuntimeError(
                f"KV dim mismatch inside batch: {x.shape[1]} vs {dim}"
            )
        length = int(x.shape[0])
        memory[i, :length] = x
        memory_mask[i, :length] = True
        segment_ids[i, :length] = item["segment_ids"]
        trajectories.append(item["trajectory"])
        ids.append(item["id"])

    return {
        "id": ids,
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
        "trajectory": torch.stack(trajectories, dim=0),
    }


def build_loader(
    manifest: Sequence[Dict[str, Any]],
    stage: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple[CachedKVDataset, DataLoader]:
    dataset = CachedKVDataset(manifest, stage=stage)
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


# =============================================================================
# FORWARD / EVALUATION
# =============================================================================

def move_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    # Action Expert is trained with FP32 master/forward for stable comparison.
    memory = batch["memory"].to(device, non_blocking=True).float()
    memory_mask = batch["memory_mask"].to(device, non_blocking=True)
    segment_ids = batch["segment_ids"].to(device, non_blocking=True)
    gt = batch["trajectory"].to(device, non_blocking=True).float()

    inputs = {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }
    return inputs, gt


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
        inputs, gt = move_batch(batch, device)
        pred = model(**inputs)
        loss, _, _ = trajectory_loss(
            pred,
            gt,
            heading_weight=HEADING_LOSS_WEIGHT,
        )

        bs = int(gt.shape[0])
        loss_sum += float(loss.item()) * bs
        sample_count += bs
        preds.append(pred.float().cpu().numpy())
        gts.append(gt.float().cpu().numpy())

    if sample_count == 0:
        raise RuntimeError("Empty validation loader")

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)
    metrics["loss"] = loss_sum / sample_count
    return metrics


# =============================================================================
# TRAIN ONE OF EIGHT
# =============================================================================

def train_one_model(
    model_name: str,
    train_manifest: Sequence[Dict[str, Any]],
    val_manifest: Sequence[Dict[str, Any]],
    input_dim: int,
    cache_signature: str,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stage, family, attention = parse_model_name(model_name)

    # Same seed before every ablation model.
    set_seed(args.seed)

    train_dataset, train_loader = build_loader(
        manifest=train_manifest,
        stage=stage,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset, val_loader = build_loader(
        manifest=val_manifest,
        stage=stage,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = build_action_expert(
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=ACTION_NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device=device, dtype=torch.float32)

    trainable_params = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_dir = args.model_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    history_path = model_dir / "history.jsonl"
    config_path = model_dir / "config.json"

    if history_path.exists():
        history_path.unlink()

    config = {
        "model_name": model_name,
        "cache_stage": stage,
        "transformer_family": family,
        "attention_interface": attention,
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_steps": ACTION_NUM_STEPS,
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
        "trainable_params": int(trainable_params),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size * args.grad_accum),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "heading_loss_weight": HEADING_LOSS_WEIGHT,
        "seed": int(args.seed),
        "precision": "fp32",
        "cache_signature": cache_signature,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 110)
    print(f"TRAIN {model_name}")
    print("=" * 110)
    print(f"Cache stage       : {stage}")
    print(f"Transformer       : {family}")
    print(f"Attention         : {attention}")
    print(f"Train / Val       : {len(train_dataset)} / {len(val_dataset)}")
    print(f"KV input dim      : {input_dim}")
    print(f"Params            : {trainable_params:,}")
    print(f"Batch             : {args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Optimizer         : AdamW lr={args.lr:.2e} wd={args.weight_decay:.2e}")
    print(f"Checkpoint        : {best_path}")

    best_val = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_samples = 0
        epoch_start = time.perf_counter()

        for micro_idx, batch in enumerate(train_loader, 1):
            inputs, gt = move_batch(batch, device)
            pred = model(**inputs)
            raw_loss, _, _ = trajectory_loss(
                pred,
                gt,
                heading_weight=HEADING_LOSS_WEIGHT,
            )
            (raw_loss / args.grad_accum).backward()

            bs = int(gt.shape[0])
            train_loss_sum += float(raw_loss.detach().item()) * bs
            train_samples += bs

            do_step = (
                micro_idx % args.grad_accum == 0
                or micro_idx == len(train_loader)
            )
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = train_loss_sum / max(1, train_samples)
        val_metrics = evaluate(model, val_loader, device)
        elapsed_min = (time.perf_counter() - epoch_start) / 60.0

        history = {
            "model_name": model_name,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_ade_m": float(val_metrics["ade_m"]),
            "val_fde_m": float(val_metrics["fde_m"]),
            "val_heading_mae_rad": float(val_metrics["heading_mae_rad"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_minutes": float(elapsed_min),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(history, ensure_ascii=False) + "\n")

        print(
            f"[{model_name}] e={epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} val={val_metrics['loss']:.6f} "
            f"ADE={val_metrics['ade_m']:.4f}m "
            f"FDE={val_metrics['fde_m']:.4f}m "
            f"Heading={val_metrics['heading_mae_rad']:.4f}rad "
            f"lr={scheduler.get_last_lr()[0]:.3e} "
            f"time={elapsed_min:.2f}m"
        )

        improved = val_metrics["loss"] < best_val - MIN_DELTA
        if improved:
            best_val = float(val_metrics["loss"])
            best_epoch = int(epoch)
            no_improve = 0

            checkpoint = {
                "model_name": model_name,
                "cache_stage": stage,
                "transformer_family": family,
                "attention_interface": attention,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": int(input_dim),
                "action_config": {
                    "hidden_dim": int(args.hidden_dim),
                    "num_steps": ACTION_NUM_STEPS,
                    "num_layers": int(args.num_layers),
                    "num_heads": int(args.num_heads),
                    "ff_dim": int(args.ff_dim),
                    "dropout": float(args.dropout),
                },
                "trainable_params": int(trainable_params),
                "cache_signature": cache_signature,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, best_path)
            print(
                f"[{model_name}] BEST -> epoch={epoch} val={best_val:.6f} "
                f"saved={best_path}"
            )
        else:
            no_improve += 1
            print(
                f"[{model_name}] no improvement {no_improve}/{args.patience} "
                f"(best={best_val:.6f})"
            )

        if no_improve >= args.patience:
            print(
                f"[{model_name}] EARLY STOP | best_epoch={best_epoch} best_val={best_val:.6f}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint produced: {best_path}")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "model_name": model_name,
        "best_epoch": int(best_checkpoint["epoch"]),
        "trainable_params": int(trainable_params),
        **{f"val_{k}": float(v) for k, v in best_checkpoint["val_metrics"].items()},
        "checkpoint": str(best_path),
    }

    del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    cleanup_cuda()
    return result


# =============================================================================
# CACHE VALIDATION
# =============================================================================

def load_cache_manifests(cache_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str]:
    root_meta_path = cache_root / "meta.json"
    train_manifest_path = cache_root / "train" / "manifest.jsonl"
    val_manifest_path = cache_root / "val" / "manifest.jsonl"

    for p in (root_meta_path, train_manifest_path, val_manifest_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run cache_action_kv.py before train_action_8way.py."
            )

    train_manifest = read_jsonl(train_manifest_path)
    val_manifest = read_jsonl(val_manifest_path)
    if not train_manifest or not val_manifest:
        raise RuntimeError("Empty train/val cache manifest")

    # ID leakage check.
    train_ids = {str(x["id"]) for x in train_manifest}
    val_ids = {str(x["id"]) for x in val_manifest}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Train/Val ID leakage detected: {list(overlap)[:5]}")

    # Clip leakage check.
    train_clips = {str(x.get("clip", "")) for x in train_manifest}
    val_clips = {str(x.get("clip", "")) for x in val_manifest}
    clip_overlap = (train_clips & val_clips) - {""}
    if clip_overlap:
        raise RuntimeError(f"Train/Val clip leakage detected: {list(clip_overlap)[:5]}")

    # Inspect first samples to infer KV dimension and make sure all stage data exists.
    first = torch.load(
        train_manifest[0]["cache_file"], map_location="cpu", weights_only=False
    )
    direct = first["direct_kv"]
    delta = first["reasoning_delta_kv"]
    if direct.ndim != 2 or delta.ndim != 2:
        raise RuntimeError("Bad cache tensor rank")
    if direct.shape[-1] != delta.shape[-1]:
        raise RuntimeError("Direct/reasoning KV dimension mismatch")
    input_dim = int(direct.shape[-1])

    cache_signature = sha256_file(root_meta_path)
    return train_manifest, val_manifest, input_dim, cache_signature


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--hidden-dim", type=int, default=ACTION_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=ACTION_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=ACTION_NUM_HEADS)
    parser.add_argument("--ff-dim", type=int, default=ACTION_FF_DIM)
    parser.add_argument("--dropout", type=float, default=ACTION_DROPOUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must be >= 1")

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_manifest, val_manifest, input_dim, cache_signature = load_cache_manifests(
        args.cache_root
    )
    args.model_root.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("ACTION EXPERT 8-WAY TRAINING")
    print("=" * 110)
    print("GPU             :", torch.cuda.get_device_name(args.gpu_id))
    print("Cache root      :", args.cache_root)
    print("Model root      :", args.model_root)
    print("Train / Val     :", len(train_manifest), "/", len(val_manifest))
    print("KV input dim    :", input_dim)
    print("Precision       : FP32")
    print("Models          :")
    for name in args.models:
        print("  -", name)
    print()

    results = []
    for model_name in args.models:
        result = train_one_model(
            model_name=model_name,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            input_dim=input_dim,
            cache_signature=cache_signature,
            device=device,
            args=args,
        )
        results.append(result)

        # Persist progress after every model so long runs are recoverable.
        (args.model_root / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n" + "=" * 110)
    print("8-WAY TRAINING COMPLETE")
    print("=" * 110)
    print(
        f"{'MODEL':<36}{'PARAMS':>14}{'VAL LOSS':>12}{'ADE':>10}{'FDE':>10}{'HEAD':>10}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r['model_name']:<36}"
            f"{r['trainable_params']:>14,}"
            f"{r['val_loss']:>12.5f}"
            f"{r['val_ade_m']:>10.4f}"
            f"{r['val_fde_m']:>10.4f}"
            f"{r['val_heading_mae_rad']:>10.4f}"
        )
    print("\nSaved models:", args.model_root)
    print("Each subdirectory contains best.pt, config.json, history.jsonl")


if __name__ == "__main__":
    main()
