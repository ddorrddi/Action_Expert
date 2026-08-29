#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the controlled ~0.5B Flow-Matching / DiT-style Action Expert.

Comparison target
-----------------
Current direct-regression Transformer:
  H1536 / 13L / 12 heads / FF6144
  learned KVMemoryProjector + decoder cross-attention
  direct 10 x (x,y,yaw) regression
  ~0.499051B @ input_dim=2048

This Flow model keeps the same cache interface and backbone scale, while changing:
  - learnable trajectory queries -> noisy trajectory x_t + timestep embedding
  - causal query self-attn -> non-causal joint denoising self-attn
  - direct trajectory loss -> flow velocity MSE
  - one-shot inference -> Euler integration

Fairness controls
-----------------
  - exact same paired cache manifests / train-val IDs
  - exact same branch definitions: traj_only / coc_reasoning
  - same H/L/heads/FF
  - same effective batch size = 8
  - same AdamW / LR / weight decay / warmup / seed by default
  - FP32 master weights + BF16 autocast + gradient checkpointing
  - same x0/t RNG stream for the two branches per epoch
  - fixed validation noise/timestep RNG
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
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    count_trainable_parameters,
    euler_sample,
    flow_matching_batch,
    trajectory_metrics_np,
)
from train_compare_ablation import (
    BRANCHES,
    REASONING_CACHE_ROOT,
    TRAJ_ONLY_CACHE_ROOT,
    PairEntry,
    build_loader,
    infer_input_dim,
    load_paired_manifest,
    sha256_file,
    validate_split_leakage,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_flow_dit_05b_controlled"
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

# Match the current direct Transformer training defaults.
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-2
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_PATIENCE = 10

DEFAULT_SOLVER_STEPS = 10
DEFAULT_TIMESTEP_SAMPLER = "beta"

GRAD_CLIP_NORM = 1.0
MIN_DELTA_ADE = 1.0e-4

# Current direct Transformer count at input_dim=2048. Used only for reporting.
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


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    # Match the direct Transformer: source memory and GT stay FP32; BF16 is autocast.
    memory = batch["memory"].to(device=device, dtype=torch.float32, non_blocking=True)
    memory_mask = batch["memory_mask"].to(device, non_blocking=True)
    segment_ids = batch["segment_ids"].to(device, non_blocking=True)
    gt = batch["trajectory"].to(device=device, dtype=torch.float32, non_blocking=True)
    return {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }, gt


def compute_normalizer(train_pairs: Sequence[PairEntry]) -> TrajectoryNormalizer:
    """Compute normalization from training GT only; never touch val/test GT statistics."""
    sums = torch.zeros(3, dtype=torch.float64)
    sq_sums = torch.zeros(3, dtype=torch.float64)
    count = 0

    for entry in train_pairs:
        cache = torch.load(
            entry.traj_cache_file,
            map_location="cpu",
            weights_only=False,
        )
        traj = cache["trajectory"].double()
        if tuple(traj.shape) != (ACTION_NUM_STEPS, 3):
            raise RuntimeError(
                f"trajectory must be [{ACTION_NUM_STEPS},3], id={entry.id}, "
                f"got={tuple(traj.shape)}"
            )
        flat = traj.reshape(-1, 3)
        sums += flat.sum(dim=0)
        sq_sums += (flat * flat).sum(dim=0)
        count += int(flat.shape[0])

    mean = sums / max(1, count)
    var = (sq_sums / max(1, count)) - mean * mean
    std = torch.sqrt(var.clamp_min(1.0e-8)).clamp_min(1.0e-3)
    return TrajectoryNormalizer(mean=mean.float(), std=std.float())


# =============================================================================
# EVAL
# =============================================================================

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    timestep_sampler: str,
) -> Dict[str, float]:
    model.eval()
    norm = normalizer.to(device)

    # Fixed streams => validation ADE is directly comparable across epochs/branches.
    noise_rng = torch.Generator(device="cpu")
    noise_rng.manual_seed(VAL_NOISE_SEED)
    flow_rng = torch.Generator(device="cpu")
    flow_rng.manual_seed(VAL_FLOW_SEED)

    preds = []
    gts = []
    flow_loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        inputs, gt = move_batch(batch, device)
        gt_norm = norm.normalize(gt)

        x_t, t, v_target, _ = flow_matching_batch(
            gt_norm,
            flow_rng,
            timestep_sampler=timestep_sampler,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = model(x_t=x_t, t=t, **inputs)
        fm_loss = F.mse_loss(v_pred.float(), v_target.float())

        bs = int(gt.shape[0])
        flow_loss_sum += float(fm_loss.item()) * bs
        sample_count += bs

        pred = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=norm,
            solver_steps=solver_steps,
            rng=noise_rng,
        )

        preds.append(pred.cpu().numpy())
        gts.append(gt.cpu().numpy())

    if sample_count == 0:
        raise RuntimeError("Empty validation loader")

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)
    metrics["flow_mse"] = flow_loss_sum / sample_count
    return metrics


# =============================================================================
# TRAIN ONE BRANCH
# =============================================================================

def train_one_branch(
    branch: str,
    train_pairs: Sequence[PairEntry],
    val_pairs: Sequence[PairEntry],
    input_dim: int,
    cache_signature: str,
    normalizer: TrajectoryNormalizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(branch)

    # Same initialization for traj_only and coc_reasoning.
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

    # Match direct Transformer precision strategy: FP32 master + BF16 autocast.
    model = build_flow_dit(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=ACTION_NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)

    trainable_params = count_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_dir = args.model_root / branch
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    history_path = model_dir / "history.jsonl"
    config_path = model_dir / "config.json"
    if history_path.exists():
        history_path.unlink()

    ref_delta = None
    if input_dim == 2048:
        ref_delta = trainable_params - TRANSFORMER_REFERENCE_PARAMS_2048

    config = {
        "experiment": "controlled_flow_matching_vs_direct_transformer_05b",
        "branch": branch,
        "conditioning": "KVMemoryProjector + decoder cross-attention",
        "objective": "linear_flow_matching_velocity_mse",
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_steps": ACTION_NUM_STEPS,
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
        "self_attention": "non_causal_joint_action_denoising",
        "trainable_params": int(trainable_params),
        "transformer_reference_params_2048": (
            TRANSFORMER_REFERENCE_PARAMS_2048 if input_dim == 2048 else None
        ),
        "parameter_delta_vs_transformer": int(ref_delta) if ref_delta is not None else None,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size * args.grad_accum),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "patience": int(args.patience),
        "solver_steps": int(args.solver_steps),
        "timestep_sampler": str(args.timestep_sampler),
        "flow_path": "x_t=(1-t)*x0+t*x1; target=x1-x0",
        "seed": int(args.seed),
        "precision": "fp32_master_bf16_autocast",
        "gradient_checkpointing": bool(not args.no_gradient_checkpointing),
        "normalizer": normalizer.to_dict(),
        "cache_signature": cache_signature,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 118)
    print(f"CONTROLLED FLOW-MATCHING ~0.5B | {branch}")
    print("=" * 118)
    print("Train / Val           :", len(train_dataset), "/", len(val_dataset))
    print("KV conditioning       : learned projector + decoder cross-attention")
    print("Architecture          :", f"H={args.hidden_dim} L={args.num_layers} heads={args.num_heads} FF={args.ff_dim}")
    print("Action self-attention : non-causal")
    print("Trainable params      :", f"{trainable_params:,}", f"({trainable_params/1e9:.6f}B)")
    if ref_delta is not None:
        print(
            "vs Transformer       :",
            f"{ref_delta:+,} params ({100.0 * ref_delta / TRANSFORMER_REFERENCE_PARAMS_2048:+.3f}%)",
        )
    print("Batch                 :", f"{args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print("Optimizer             :", f"AdamW lr={args.lr:.2e} wd={args.weight_decay:.2e}")
    print("Flow timestep sampler :", args.timestep_sampler)
    print("Euler solver steps    :", args.solver_steps)
    print("Precision             : FP32 master + BF16 autocast")
    print("Grad checkpointing    :", not args.no_gradient_checkpointing)
    print("Best selection        : minimum validation ADE")
    print("Checkpoint            :", best_path)

    best_ade = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0
    norm = normalizer.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()

        # Same random x0/t stream for both branches at the same epoch.
        flow_rng = torch.Generator(device="cpu")
        flow_rng.manual_seed(args.seed + epoch * 1009)

        train_loss_sum = 0.0
        train_samples = 0

        for micro_idx, batch in enumerate(train_loader, 1):
            inputs, gt = move_batch(batch, device)
            gt_norm = norm.normalize(gt)

            x_t, t, v_target, _ = flow_matching_batch(
                gt_norm,
                flow_rng,
                timestep_sampler=args.timestep_sampler,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = model(x_t=x_t, t=t, **inputs)
            loss = F.mse_loss(v_pred.float(), v_target.float())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss branch={branch} epoch={epoch} "
                    f"micro={micro_idx}: {loss.item()}"
                )

            (loss / args.grad_accum).backward()

            bs = int(gt.shape[0])
            train_loss_sum += float(loss.detach().item()) * bs
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
        val = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            normalizer=normalizer,
            solver_steps=args.solver_steps,
            timestep_sampler=args.timestep_sampler,
        )

        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        elapsed_min = (time.perf_counter() - epoch_start) / 60.0

        row = {
            "branch": branch,
            "epoch": epoch,
            "global_step": global_step,
            "train_flow_mse": float(train_loss),
            "val_flow_mse": float(val["flow_mse"]),
            "val_ade_m": float(val["ade_m"]),
            "val_fde_m": float(val["fde_m"]),
            "val_heading_mae_rad": float(val["heading_mae_rad"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "peak_vram_gb": float(peak_gb),
            "epoch_minutes": float(elapsed_min),
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[{branch}] e={epoch:03d}/{args.epochs:03d} "
            f"trainFM={train_loss:.6f} valFM={val['flow_mse']:.6f} "
            f"ADE={val['ade_m']:.4f}m FDE={val['fde_m']:.4f}m "
            f"Heading={val['heading_mae_rad']:.4f}rad "
            f"lr={scheduler.get_last_lr()[0]:.3e} "
            f"VRAM={peak_gb:.2f}GB time={elapsed_min:.2f}m"
        )

        improved = val["ade_m"] < best_ade - MIN_DELTA_ADE
        if improved:
            best_ade = float(val["ade_m"])
            best_epoch = int(epoch)
            no_improve = 0

            checkpoint = {
                "experiment": "controlled_flow_matching_vs_direct_transformer_05b",
                "branch": branch,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": int(input_dim),
                "trainable_params": int(trainable_params),
                "cache_signature": cache_signature,
                "normalizer": normalizer.to_dict(),
                "action_config": {
                    "hidden_dim": int(args.hidden_dim),
                    "num_steps": ACTION_NUM_STEPS,
                    "num_layers": int(args.num_layers),
                    "num_heads": int(args.num_heads),
                    "ff_dim": int(args.ff_dim),
                    "dropout": float(args.dropout),
                    "gradient_checkpointing": bool(not args.no_gradient_checkpointing),
                    "solver_steps": int(args.solver_steps),
                    "timestep_sampler": str(args.timestep_sampler),
                    "conditioning": "KVMemoryProjector + decoder cross-attention",
                    "self_attention": "non_causal",
                    "flow_path": "x_t=(1-t)*x0+t*x1; target=x1-x0",
                },
                "val_metrics": {k: float(v) for k, v in val.items()},
            }
            torch.save(checkpoint, best_path)
            print(f"[{branch}] BEST -> epoch={epoch} val_ADE={best_ade:.6f} saved={best_path}")
        else:
            no_improve += 1
            print(
                f"[{branch}] no ADE improvement {no_improve}/{args.patience} "
                f"(best={best_ade:.6f})"
            )

        if no_improve >= args.patience:
            print(
                f"[{branch}] EARLY STOP | best_epoch={best_epoch} "
                f"best_ADE={best_ade:.6f}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint produced: {best_path}")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "branch": branch,
        "best_epoch": int(ckpt["epoch"]),
        "trainable_params": int(ckpt["trainable_params"]),
        **{f"val_{k}": float(v) for k, v in ckpt["val_metrics"].items()},
        "checkpoint": str(best_path),
    }

    del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    cleanup_cuda()
    return result


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj-only-cache-root", type=Path, default=TRAJ_ONLY_CACHE_ROOT)
    p.add_argument("--reasoning-cache-root", type=Path, default=REASONING_CACHE_ROOT)
    p.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES))

    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--solver-steps", type=int, default=DEFAULT_SOLVER_STEPS)
    p.add_argument(
        "--timestep-sampler",
        choices=("beta", "uniform"),
        default=DEFAULT_TIMESTEP_SAMPLER,
    )

    p.add_argument("--hidden-dim", type=int, default=ACTION_HIDDEN_DIM)
    p.add_argument("--num-layers", type=int, default=ACTION_NUM_LAYERS)
    p.add_argument("--num-heads", type=int, default=ACTION_NUM_HEADS)
    p.add_argument("--ff-dim", type=int, default=ACTION_FF_DIM)
    p.add_argument("--dropout", type=float, default=ACTION_DROPOUT)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must be >= 1")

    torch.cuda.set_device(args.gpu_id)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Selected GPU does not support BF16. Use RTX 3080 Ti (GPU 0)."
        )
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_pairs = load_paired_manifest(
        "train", args.traj_only_cache_root, args.reasoning_cache_root
    )
    val_pairs = load_paired_manifest(
        "val", args.traj_only_cache_root, args.reasoning_cache_root
    )
    validate_split_leakage(train_pairs, val_pairs)

    input_dim, _ = infer_input_dim(train_pairs)
    normalizer = compute_normalizer(train_pairs)

    traj_meta = args.traj_only_cache_root / "meta.json"
    reason_meta = args.reasoning_cache_root / "meta.json"
    sig_payload = {
        "traj_only_meta": sha256_file(traj_meta) if traj_meta.is_file() else None,
        "reasoning_meta": sha256_file(reason_meta) if reason_meta.is_file() else None,
        "train_ids": [x.id for x in train_pairs],
        "val_ids": [x.id for x in val_pairs],
        "normalizer": normalizer.to_dict(),
    }
    cache_signature = hashlib.sha256(
        json.dumps(sig_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    args.model_root.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("CONTROLLED ~0.5B FLOW-MATCHING vs DIRECT TRANSFORMER ABLATION")
    print("=" * 118)
    print("GPU                :", torch.cuda.get_device_name(args.gpu_id))
    print("Paired Train / Val :", len(train_pairs), "/", len(val_pairs))
    print("KV input dim       :", input_dim)
    print("Conditioning       : same learned projector + cross-attention as Transformer")
    print("Backbone           :", f"H={args.hidden_dim} L={args.num_layers} heads={args.num_heads} FF={args.ff_dim}")
    print("Flow               : noisy trajectory+t -> velocity -> Euler")
    print("Branches           :", ", ".join(args.branches))
    print("Model root         :", args.model_root)

    results = []
    for branch in args.branches:
        result = train_one_branch(
            branch=branch,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            input_dim=input_dim,
            cache_signature=cache_signature,
            normalizer=normalizer,
            device=device,
            args=args,
        )
        results.append(result)
        (args.model_root / "summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 118)
    print("TRAINING COMPLETE")
    print("=" * 118)
    print(f"{'BRANCH':<22}{'PARAMS(B)':>12}{'VAL FM':>12}{'ADE':>10}{'FDE':>10}{'HEAD':>10}")
    for r in results:
        print(
            f"{r['branch']:<22}"
            f"{r['trainable_params']/1e9:>12.3f}"
            f"{r['val_flow_mse']:>12.5f}"
            f"{r['val_ade_m']:>10.4f}"
            f"{r['val_fde_m']:>10.4f}"
            f"{r['val_heading_mae_rad']:>10.4f}"
        )

    if {r["branch"] for r in results} == set(BRANCHES):
        by = {r["branch"]: r for r in results}
        base = by["traj_only"]["val_ade_m"]
        coc = by["coc_reasoning"]["val_ade_m"]
        improvement = 100.0 * (base - coc) / base if base != 0 else float("nan")
        print(f"\nCoC relative ADE improvement vs Traj-only: {improvement:+.2f}%")

    print("Saved models:", args.model_root)


if __name__ == "__main__":
    main()
