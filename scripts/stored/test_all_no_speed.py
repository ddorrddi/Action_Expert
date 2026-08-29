#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test all Action Experts trained by final_train_all.py

Test set:
    Part2 held-out 300

Models:
    Transformer 8-way
      traj_only_encoder_self
      coc_reasoning_encoder_self
      traj_only_encoder_cross
      coc_reasoning_encoder_cross
      traj_only_decoder_self
      coc_reasoning_decoder_self
      traj_only_decoder_cross
      coc_reasoning_decoder_cross

    Flow / DiT v2
      traj_only
      coc_reasoning

Important:
    - Reasoning_VLM_1200
    - 3 camera images + mission command ONLY
    - NO speed
    - target greedy AR reasoning
    - NO DFlash
    - Part2 is TEST ONLY
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


# =============================================================================
# IMPORT TRAINING IMPLEMENTATION
# =============================================================================

from final_train_all import (
    VLM_PATH,
    TRAIN_JSONL,
    VAL_JSONL,
    MODEL_ROOT,
    CACHE_VERSION,
    PROMPT_MODE,
    ATTN_IMPLEMENTATION,
    NUM_STEPS,
    SEED,
    VAL_NOISE_SEED,
    VAL_FLOW_SEED,
    read_jsonl,
    write_jsonl,
    sha256_file,
    cache_filename,
    resolve_dtype,
    load_split,
    load_target_model,
    extract_one,
    cleanup_cuda,
    infer_input_dim,
    build_loader,
    move_batch,
    eval_direct,
    eval_flow,
    set_seed,
)

from action_model_compare_8way import (
    build_action_expert as build_direct,
)

from action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    euler_sample,
)


# =============================================================================
# PATHS
# =============================================================================

TEST_JSONL = Path(
    "/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl"
)

TEST_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/"
    "ActionExpert10_1200/part2_action_kv_cache"
)

RESULT_ROOT = MODEL_ROOT / "part2_test"


# =============================================================================
# TEST CONFIG
# =============================================================================

TEST_VLM_DTYPE = "fp16"
MAX_NEW_TOKENS = 128

GPU_ID = 0

FLOW_SOLVER_STEPS = 10

LATENCY_WARMUP = 5


TRANSFORMER_ORDER = (
    ("encoder_self", "traj_only"),
    ("encoder_self", "coc_reasoning"),
    ("encoder_cross", "traj_only"),
    ("encoder_cross", "coc_reasoning"),
    ("decoder_self", "traj_only"),
    ("decoder_self", "coc_reasoning"),
    ("decoder_cross", "traj_only"),
    ("decoder_cross", "coc_reasoning"),
)

FLOW_ORDER = (
    "traj_only",
    "coc_reasoning",
)


# =============================================================================
# HELPERS
# =============================================================================

def ids_from_jsonl(path: Path) -> set[str]:
    return {
        str(row["id"])
        for row in read_jsonl(path)
        if row.get("id") is not None
    }


def validate_test_no_leakage(test_rows) -> None:
    test_ids = {str(row["id"]) for row in test_rows}

    train_ids = ids_from_jsonl(TRAIN_JSONL)
    val_ids = ids_from_jsonl(VAL_JSONL)

    overlap_train = test_ids & train_ids
    overlap_val = test_ids & val_ids

    if overlap_train:
        raise RuntimeError(
            "Part2 TEST overlaps TRAIN IDs: "
            f"{sorted(overlap_train)[:10]}"
        )

    if overlap_val:
        raise RuntimeError(
            "Part2 TEST overlaps VAL IDs: "
            f"{sorted(overlap_val)[:10]}"
        )

    print(
        "[LEAKAGE] PASS | "
        f"test={len(test_ids)} "
        f"train_overlap=0 val_overlap=0"
    )


def test_cache_request(args) -> Dict[str, Any]:
    config_path = args.vlm / "config.json"

    return {
        "cache_version": CACHE_VERSION,
        "vlm": str(args.vlm.resolve()),
        "vlm_config_sha256": (
            sha256_file(config_path)
            if config_path.is_file()
            else None
        ),
        "test_jsonl": str(args.test_jsonl.resolve()),
        "test_sha256": sha256_file(args.test_jsonl),
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": int(args.max_new_tokens),
        "allow_truncated": bool(args.allow_truncated),
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "attention_implementation": ATTN_IMPLEMENTATION,
    }


def load_valid_test_cache(args, expected_ids):
    meta_path = args.cache_root / "meta.json"
    manifest_path = args.cache_root / "manifest.jsonl"

    if not meta_path.is_file():
        return None

    if not manifest_path.is_file():
        return None

    try:
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None

    if meta.get("request") != test_cache_request(args):
        return None

    manifest = read_jsonl(manifest_path)

    manifest_ids = [str(x["id"]) for x in manifest]

    if manifest_ids != list(expected_ids):
        return None

    for item in manifest:
        if not Path(item["cache_file"]).is_file():
            return None

    return manifest


# =============================================================================
# PART2 TARGET VLM CACHE
# =============================================================================

def build_test_cache(
    args,
    test_rows,
    device,
    vlm_dtype,
):
    expected_ids = [str(x["id"]) for x in test_rows]

    if args.cache_root.exists() and not args.rebuild_cache:
        cached = load_valid_test_cache(
            args,
            expected_ids,
        )

        if cached is not None:
            print(
                f"[TEST CACHE] reuse | samples={len(cached)}"
            )
            return cached

        raise RuntimeError(
            "\nExisting Part2 cache is incompatible:\n"
            f"    {args.cache_root}\n\n"
            "Do NOT silently reuse an old VLM cache.\n"
            "Run again with --rebuild-cache."
        )

    if args.cache_root.exists():
        shutil.rmtree(args.cache_root)

    args.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 120)
    print("STAGE 1 | PART2 TARGET-ONLY VLM CACHE")
    print("=" * 120)
    print("VLM        :", args.vlm)
    print("Test       :", args.test_jsonl)
    print("Samples    :", len(test_rows))
    print("Prompt     :", PROMPT_MODE)
    print("Input      : 3 cameras + mission command")
    print("Speed      : NOT USED")
    print("Reasoning  : target greedy autoregressive")
    print("DFlash     : NOT USED")
    print("Cache      :", args.cache_root)

    model, processor = load_target_model(
        args.vlm,
        device,
        vlm_dtype,
    )

    manifest = []

    for index, row in enumerate(test_rows, 1):
        sid = str(row["id"])
        t0 = time.perf_counter()

        out_path = (
            args.cache_root
            / cache_filename(index, sid)
        )

        record = extract_one(
            model=model,
            processor=processor,
            row=row,
            device=device,
            dtype=vlm_dtype,
            max_new_tokens=args.max_new_tokens,
            allow_truncated=args.allow_truncated,
        )

        torch.save(record, out_path)

        manifest.append({
            "id": sid,
            "clip": str(row.get("clip", "")),
            "cache_file": str(out_path),
            "kv_dim": int(
                record["direct_kv"].shape[-1]
            ),
            "prompt_cache_length": int(
                record["prompt_cache_length"]
            ),
            "reasoning_cached_tokens": int(
                record["reasoning_cached_tokens"]
            ),
            "reasoning_text": record["reasoning_text"],
            "prefix_max_abs_diff": float(
                record["prefix_max_abs_diff"]
            ),
        })

        print(
            f"[cache {index:04d}/{len(test_rows):04d}] "
            f"id={sid} "
            f"directT={record['direct_kv'].shape[0]} "
            f"reasonT={record['reasoning_delta_kv'].shape[0]} "
            f"D={record['direct_kv'].shape[-1]} "
            f"prefix_diff="
            f"{record['prefix_max_abs_diff']:.3e} "
            f"time={time.perf_counter()-t0:.2f}s",
            flush=True,
        )

        del record

    write_jsonl(
        args.cache_root / "manifest.jsonl",
        manifest,
    )

    meta = {
        "request": test_cache_request(args),
        "samples": len(manifest),
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    (
        args.cache_root / "meta.json"
    ).write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    del model, processor
    cleanup_cuda()

    return manifest


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_transformer_checkpoint(
    checkpoint_path: Path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["action_config"]

    model = build_direct(
        architecture=checkpoint["architecture"],
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    return model, checkpoint


def load_flow_checkpoint(
    checkpoint_path: Path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["action_config"]

    model = build_flow_dit(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    normalizer_dict = checkpoint["normalizer"]

    if (
        "mean" not in normalizer_dict
        or "std" not in normalizer_dict
    ):
        raise RuntimeError(
            "Checkpoint normalizer must contain mean/std"
        )

    normalizer = TrajectoryNormalizer(
        mean=torch.as_tensor(
            normalizer_dict["mean"],
            dtype=torch.float32,
        ),
        std=torch.as_tensor(
            normalizer_dict["std"],
            dtype=torch.float32,
        ),
    )

    return model, normalizer, checkpoint


# =============================================================================
# LATENCY
# =============================================================================

@torch.inference_mode()
def benchmark_direct_latency(
    model,
    loader,
    device,
    warmup=5,
):
    times_ms = []

    for index, batch in enumerate(loader):
        inputs, _ = move_batch(
            batch,
            device,
        )

        if index < warmup:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                _ = model(**inputs)
            continue

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _ = model(**inputs)

        torch.cuda.synchronize(device)

        times_ms.append(
            (time.perf_counter() - t0) * 1000.0
        )

    if not times_ms:
        return math.nan

    return float(np.mean(times_ms))


@torch.inference_mode()
def benchmark_flow_latency(
    model,
    loader,
    device,
    normalizer,
    solver_steps,
    warmup=5,
):
    norm = normalizer.to(device)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(VAL_NOISE_SEED)

    times_ms = []

    for index, batch in enumerate(loader):
        inputs, _ = move_batch(
            batch,
            device,
        )

        if index < warmup:
            _ = euler_sample(
                model=model,
                memory=inputs["memory"],
                memory_mask=inputs["memory_mask"],
                segment_ids=inputs["segment_ids"],
                normalizer=norm,
                solver_steps=solver_steps,
                rng=rng,
            )
            continue

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        _ = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=norm,
            solver_steps=solver_steps,
            rng=rng,
        )

        torch.cuda.synchronize(device)

        times_ms.append(
            (time.perf_counter() - t0) * 1000.0
        )

    if not times_ms:
        return math.nan

    return float(np.mean(times_ms))


# =============================================================================
# TRANSFORMER TEST
# =============================================================================

def test_transformer(
    architecture,
    branch,
    manifest,
    device,
    args,
):
    name = f"{branch}_{architecture}"

    checkpoint_path = (
        args.model_root
        / "transformer"
        / name
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        print(
            f"[SKIP] missing checkpoint: "
            f"{checkpoint_path}"
        )
        return None

    print()
    print("=" * 120)
    print(f"TEST TRANSFORMER | {name}")
    print("=" * 120)
    print("Checkpoint :", checkpoint_path)

    model, checkpoint = load_transformer_checkpoint(
        checkpoint_path,
        device,
    )

    _, loader = build_loader(
        manifest=manifest,
        branch=branch,
        batch_size=1,
        shuffle=False,
        seed=SEED,
    )

    metrics = eval_direct(
        model,
        loader,
        device,
    )

    ae_ms = benchmark_direct_latency(
        model=model,
        loader=loader,
        device=device,
        warmup=args.latency_warmup,
    )

    print(
        f"[RESULT] {name} | "
        f"ADE={metrics['ade_m']:.4f}m "
        f"FDE={metrics['fde_m']:.4f}m "
        f"Heading={metrics['heading_mae_rad']:.4f}rad "
        f"AE={ae_ms:.3f}ms"
    )

    result = {
        "family": "transformer",
        "condition": name,
        "branch": branch,
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "best_val_epoch": int(checkpoint["epoch"]),
        "test_loss": float(metrics["loss"]),
        "test_ade_m": float(metrics["ade_m"]),
        "test_fde_m": float(metrics["fde_m"]),
        "test_heading_mae_rad": float(
            metrics["heading_mae_rad"]
        ),
        "action_expert_ms": float(ae_ms),
    }

    del model, checkpoint, loader
    cleanup_cuda()

    return result


# =============================================================================
# FLOW TEST
# =============================================================================

def test_flow(
    branch,
    manifest,
    device,
    args,
):
    checkpoint_path = (
        args.model_root
        / "flow"
        / branch
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        print(
            f"[SKIP] missing checkpoint: "
            f"{checkpoint_path}"
        )
        return None

    print()
    print("=" * 120)
    print(f"TEST FLOW / DiT v2 | {branch}")
    print("=" * 120)
    print("Checkpoint :", checkpoint_path)

    model, normalizer, checkpoint = (
        load_flow_checkpoint(
            checkpoint_path,
            device,
        )
    )

    cfg = checkpoint["action_config"]

    solver_steps = int(
        cfg.get(
            "solver_steps",
            FLOW_SOLVER_STEPS,
        )
    )

    timestep_sampler = str(
        cfg.get(
            "timestep_sampler",
            "uniform",
        )
    )

    _, loader = build_loader(
        manifest=manifest,
        branch=branch,
        batch_size=1,
        shuffle=False,
        seed=SEED,
    )

    # eval_flow() resets its validation RNG internally.
    # Therefore traj_only / coc_reasoning receive the same
    # deterministic flow/noise streams.
    metrics = eval_flow(
        model=model,
        loader=loader,
        device=device,
        normalizer=normalizer,
        solver_steps=solver_steps,
        timestep_sampler=timestep_sampler,
    )

    ae_ms = benchmark_flow_latency(
        model=model,
        loader=loader,
        device=device,
        normalizer=normalizer,
        solver_steps=solver_steps,
        warmup=args.latency_warmup,
    )

    print(
        f"[RESULT] flow/{branch} | "
        f"ADE={metrics['ade_m']:.4f}m "
        f"FDE={metrics['fde_m']:.4f}m "
        f"Heading={metrics['heading_mae_rad']:.4f}rad "
        f"FM={metrics['flow_mse']:.6f} "
        f"AE={ae_ms:.3f}ms"
    )

    result = {
        "family": "flow",
        "condition": f"{branch}_flow_dit",
        "branch": branch,
        "architecture": "flow_dit_decoder_cross",
        "checkpoint": str(checkpoint_path),
        "best_val_epoch": int(checkpoint["epoch"]),
        "solver_steps": solver_steps,
        "test_flow_mse": float(
            metrics["flow_mse"]
        ),
        "test_ade_m": float(metrics["ade_m"]),
        "test_fde_m": float(metrics["fde_m"]),
        "test_heading_mae_rad": float(
            metrics["heading_mae_rad"]
        ),
        "action_expert_ms": float(ae_ms),
    }

    del (
        model,
        normalizer,
        checkpoint,
        loader,
    )
    cleanup_cuda()

    return result


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_results(
    args,
    results,
    test_samples,
):
    args.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": (
            "reasoning_vlm1200_10model_part2_test"
        ),
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "test_jsonl": str(args.test_jsonl),
        "test_samples": int(test_samples),
        "vlm": str(args.vlm),
        "prompt_mode": PROMPT_MODE,
        "speed_used": False,
        "dflash_used": False,
        "decode_backend": "target_autoregressive",
        "results": results,
    }

    json_path = (
        args.result_root
        / "part2_300_results.json"
    )

    txt_path = (
        args.result_root
        / "part2_300_results.txt"
    )

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 145,
        (
            "Reasoning_VLM_1200 | "
            "10 ACTION EXPERTS | PART2 HELD-OUT TEST"
        ),
        "=" * 145,
        f"VLM          : {args.vlm}",
        f"Test         : {args.test_jsonl}",
        f"Samples      : {test_samples}",
        f"Prompt       : {PROMPT_MODE}",
        "Input        : 3 cameras + mission command",
        "Speed        : NOT USED",
        "Reasoning    : Target greedy AR",
        "DFlash       : NOT USED",
        "",
        (
            f"{'FAMILY':<13}"
            f"{'CONDITION':<38}"
            f"{'BEST_E':>8}"
            f"{'ADE(m)':>12}"
            f"{'FDE(m)':>12}"
            f"{'HEAD(rad)':>12}"
            f"{'AE(ms)':>12}"
        ),
        "-" * 145,
    ]

    for r in results:
        lines.append(
            f"{r['family']:<13}"
            f"{r['condition']:<38}"
            f"{r['best_val_epoch']:>8d}"
            f"{r['test_ade_m']:>12.4f}"
            f"{r['test_fde_m']:>12.4f}"
            f"{r['test_heading_mae_rad']:>12.4f}"
            f"{r['action_expert_ms']:>12.3f}"
        )

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return txt_path, json_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--vlm",
        type=Path,
        default=VLM_PATH,
    )

    p.add_argument(
        "--test-jsonl",
        type=Path,
        default=TEST_JSONL,
    )

    p.add_argument(
        "--cache-root",
        type=Path,
        default=TEST_CACHE_ROOT,
    )

    p.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
    )

    p.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
    )

    p.add_argument(
        "--gpu-id",
        type=int,
        default=GPU_ID,
    )

    p.add_argument(
        "--vlm-dtype",
        choices=("fp16", "bf16"),
        default=TEST_VLM_DTYPE,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    p.add_argument(
        "--allow-truncated",
        action="store_true",
    )

    p.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    p.add_argument(
        "--latency-warmup",
        type=int,
        default=LATENCY_WARMUP,
    )

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)

    for path in (
        args.vlm,
        args.test_jsonl,
        args.model_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    test_rows = load_split(
        args.test_jsonl
    )

    validate_test_no_leakage(
        test_rows
    )

    print("=" * 120)
    print(
        "Reasoning_VLM_1200 | "
        "10 ACTION EXPERTS | PART2 TEST"
    )
    print("=" * 120)
    print(
        "GPU          :",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )
    print("VLM          :", args.vlm)
    print("Test         :", args.test_jsonl)
    print("Samples      :", len(test_rows))
    print("Model root   :", args.model_root)
    print("Cache root   :", args.cache_root)
    print("Result root  :", args.result_root)
    print("Prompt       :", PROMPT_MODE)
    print(
        "Input        : "
        "front_left + front + front_right "
        "+ mission command"
    )
    print("Speed        : NOT USED")
    print("DFlash       : NOT USED")

    # ---------------------------------------------------------
    # 1. Build/reuse one unified Part2 VLM cache
    # ---------------------------------------------------------

    manifest = build_test_cache(
        args=args,
        test_rows=test_rows,
        device=device,
        vlm_dtype=resolve_dtype(
            args.vlm_dtype
        ),
    )

    input_dim = infer_input_dim(
        manifest
    )

    print()
    print(
        f"[CACHE] input_dim={input_dim} "
        f"samples={len(manifest)}"
    )

    # ---------------------------------------------------------
    # 2. Evaluate 8 Transformer models
    # ---------------------------------------------------------

    results = []

    for architecture, branch in TRANSFORMER_ORDER:
        result = test_transformer(
            architecture=architecture,
            branch=branch,
            manifest=manifest,
            device=device,
            args=args,
        )

        if result is not None:
            results.append(result)

    # ---------------------------------------------------------
    # 3. Evaluate 2 Flow models
    # ---------------------------------------------------------

    for branch in FLOW_ORDER:
        result = test_flow(
            branch=branch,
            manifest=manifest,
            device=device,
            args=args,
        )

        if result is not None:
            results.append(result)

    # ---------------------------------------------------------
    # 4. Save final summary
    # ---------------------------------------------------------

    txt_path, json_path = save_results(
        args=args,
        results=results,
        test_samples=len(test_rows),
    )

    print()
    print("=" * 145)
    print("FINAL PART2 TEST")
    print("=" * 145)

    print(
        f"{'FAMILY':<13}"
        f"{'CONDITION':<38}"
        f"{'BEST_E':>8}"
        f"{'ADE(m)':>12}"
        f"{'FDE(m)':>12}"
        f"{'HEAD(rad)':>12}"
        f"{'AE(ms)':>12}"
    )

    print("-" * 145)

    for r in results:
        print(
            f"{r['family']:<13}"
            f"{r['condition']:<38}"
            f"{r['best_val_epoch']:>8d}"
            f"{r['test_ade_m']:>12.4f}"
            f"{r['test_fde_m']:>12.4f}"
            f"{r['test_heading_mae_rad']:>12.4f}"
            f"{r['action_expert_ms']:>12.3f}"
        )

    print()
    print("Expected models :", 10)
    print("Tested models   :", len(results))
    print("TXT             :", txt_path)
    print("JSON            :", json_path)

    if len(results) != 10:
        print()
        print(
            "[WARNING] Not all 10 checkpoints "
            "were found/tested."
        )


if __name__ == "__main__":
    main()