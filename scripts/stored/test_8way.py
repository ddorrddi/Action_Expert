#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuReasoning Part2 held-out test for the capacity-scaled Action Expert 8-way ablation.

This script evaluates the actual best checkpoints, not the validation metrics
saved inside them.

Clean test-time input reproduction
----------------------------------
traj_only:
    VLM_Baseline
      -> same 3-image + speed + route-command prompt used at training
      -> prompt-boundary last-layer K/V
      -> Action Expert

coc_reasoning:
    Reasoning_VLM
      -> same prompt used at training
      -> prompt-boundary direct K/V
      -> generation-time reasoning K/V delta
      -> concat(direct_kv, reasoning_delta_kv)
      -> Action Expert

Eight conditions
----------------
  traj_only_encoder_self
  coc_reasoning_encoder_self
  traj_only_encoder_cross
  coc_reasoning_encoder_cross
  traj_only_decoder_self
  coc_reasoning_decoder_self
  traj_only_decoder_cross
  coc_reasoning_decoder_cross

Memory policy
-------------
The ~0.5B Action Experts are NEVER loaded together.
Each checkpoint is loaded -> evaluated on exactly the same paired Part2 samples
-> deleted before the next checkpoint is loaded.

Default Part2 source:
    /home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl

Default quick test:
    --limit 50

Recommended:
    CUDA_VISIBLE_DEVICES=0 python3 test_capacity_8way_part2.py \
        --limit 50 \
        --rebuild-cache

After the feature cache exists, re-running the same command without
--rebuild-cache skips VLM feature extraction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


# =============================================================================
# PROJECT PATHS / IMPORTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ALPAMAYO_SCRIPT_DIR = Path("/home/lhh/lab/Action_Expert/scripts")
ACTION8_SCRIPT_DIR = Path("/home/lhh/lab/Action_Expert/scripts")

for _p in (SCRIPT_DIR, ALPAMAYO_SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import cache_traj_only_kv as traj_cache_tools
except ImportError as exc:
    raise ImportError(
        "cache_traj_only_kv.py was not found."
    ) from exc

try:
    import cache_action_8way as reasoning_cache_tools
except ImportError as exc:
    raise ImportError(
        "cache_action_8way.py was not found beside test_8way.py."
    ) from exc

try:
    from scripts.stored.action_model_compare_8way import (
        ARCHITECTURES,
        build_action_expert,
        per_sample_metrics_np,
        trajectory_metrics_np,
    )
except ImportError as exc:
    raise ImportError(
        "action_model_compare_8way.py was not found."
    ) from exc


# =============================================================================
# DEFAULT PATHS / CONFIG
# =============================================================================

PART2_DATASET = Path("/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl")

BASELINE_VLM = Path("/home/lhh/lab/models/vlm/VLM_Baseline")
REASONING_VLM = Path("/home/lhh/lab/models/vlm/Reasoning_VLM")

EXISTING_DECODER_CROSS_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_capacity"
)
REMAINING6_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_remaining6"
)

FEATURE_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpertAlpamayo/part2_capacity_8way_cache"
)
RESULT_ROOT = Path(
    "/home/lhh/lab/VLM/Results/Action_Expert/part2_capacity_8way"
)

BRANCHES = ("traj_only", "coc_reasoning")

DEFAULT_LIMIT = 50
DEFAULT_SEED = 20260825
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_WARMUP = 1
DEFAULT_BOOTSTRAP = 5000

CACHE_VERSION = "part2_capacity_8way_clean_inputs_v1"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
                raise RuntimeError(f"JSON parse failed: {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cache_name(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}_{digest}.pt"


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def resolve_checkpoint(
    existing_root: Path,
    remaining_root: Path,
    branch: str,
    architecture: str,
) -> Path:
    if architecture == "decoder_cross":
        return existing_root / branch / "best.pt"
    return remaining_root / f"{branch}_{architecture}" / "best.pt"


# =============================================================================
# PART2 ROW VALIDATION / SELECTION
# =============================================================================

def normalize_part2_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)

    if not row.get("command"):
        row["command"] = row.get("mission_command", "UNKNOWN")

    if row.get("speed_mps") is None:
        raise ValueError(f"speed_mps missing: id={row.get('id')}")

    images = row.get("images")

    # nuReasoning Part2 raw/prepared format:
    # {
    #   "front_left": "...",
    #   "front": "...",
    #   "front_right": "..."
    # }
    if isinstance(images, dict):
        required = ("front_left", "front", "front_right")
        missing = [key for key in required if not images.get(key)]
        if missing:
            raise ValueError(
                f"Missing image keys: id={row.get('id')} missing={missing}"
            )

        images = [
            images["front_left"],
            images["front"],
            images["front_right"],
        ]

    elif isinstance(images, list):
        if len(images) != 3:
            raise ValueError(
                f"Expected exactly 3 images: "
                f"id={row.get('id')} images={images}"
            )

    else:
        raise ValueError(
            f"Unsupported images format: "
            f"id={row.get('id')} type={type(images).__name__}"
        )

    # 이후 cache_action_8way.py / cache_traj_only_kv.py가
    # 기대하는 list 형식으로 통일
    row["images"] = images

    for image_path in images:
        p = Path(str(image_path))
        if not p.is_file():
            raise FileNotFoundError(p)

        if "/part_2/" not in str(p):
            raise RuntimeError(
                f"Not a nuReasoning Part2 image: "
                f"id={row.get('id')} path={p}"
            )

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (10, 3):
        raise ValueError(
            f"GT trajectory must be [10,3]: "
            f"id={row.get('id')} shape={gt.shape}"
        )

    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Empty sample id")

    row["id"] = sample_id

    return row


def select_rows(
    all_rows: Sequence[Dict[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    normalized = [normalize_part2_row(x) for x in all_rows]

    ids = [x["id"] for x in normalized]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate IDs in Part2 dataset")

    if limit <= 0:
        raise ValueError("--limit must be > 0")
    if limit > len(normalized):
        raise RuntimeError(
            f"Requested --limit {limit}, but dataset has only {len(normalized)} rows"
        )

    # Deterministic held-out subset rather than blindly using the first N.
    indices = list(range(len(normalized)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = indices[:limit]

    return [normalized[i] for i in chosen]


# =============================================================================
# FEATURE CACHE BUILD
# =============================================================================

def feature_request_meta(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "selected_ids": [str(x["id"]) for x in rows],
        "baseline_vlm": str(args.baseline_vlm.resolve()),
        "reasoning_vlm": str(args.reasoning_vlm.resolve()),
        "dtype": args.vlm_dtype,
        "attention": args.attn,
        "max_new_tokens": int(args.max_new_tokens),
        "allow_truncated": bool(args.allow_truncated),
        "seed": int(args.seed),
        "limit": int(args.limit),
        "representation": {
            "traj_only": "VLM_Baseline prompt-boundary last-layer KV",
            "coc_reasoning": "Reasoning_VLM direct_kv + generation-time reasoning_delta_kv",
        },
    }


def validate_cached_manifest(
    manifest: Sequence[Dict[str, Any]],
) -> bool:
    if not manifest:
        return False

    for item in manifest:
        for key in ("traj_cache_file", "reasoning_cache_file"):
            p = Path(str(item.get(key, "")))
            if not p.is_file():
                return False
    return True


def maybe_reuse_feature_cache(
    args: argparse.Namespace,
    request_meta: Dict[str, Any],
) -> List[Dict[str, Any]] | None:
    meta_path = args.cache_root / "meta.json"
    manifest_path = args.cache_root / "manifest.jsonl"

    if not meta_path.is_file() or not manifest_path.is_file():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("request") != request_meta:
        return None

    manifest = read_jsonl(manifest_path)
    if not validate_cached_manifest(manifest):
        return None

    print(
        f"[CACHE] Reusing valid Part2 feature cache: "
        f"{len(manifest)} paired samples"
    )
    return manifest


def build_feature_cache(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    vlm_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    request_meta = feature_request_meta(args, rows)

    if args.cache_root.exists() and not args.rebuild_cache:
        reused = maybe_reuse_feature_cache(args, request_meta)
        if reused is not None:
            return reused

        raise RuntimeError(
            f"Existing cache is missing/incompatible: {args.cache_root}\n"
            "Re-run with --rebuild-cache."
        )

    if args.cache_root.exists():
        shutil.rmtree(args.cache_root)

    traj_dir = args.cache_root / "traj_only"
    reason_dir = args.cache_root / "coc_reasoning"
    traj_dir.mkdir(parents=True, exist_ok=True)
    reason_dir.mkdir(parents=True, exist_ok=True)

    baseline_records: Dict[str, Dict[str, Any]] = {}
    skips: List[Dict[str, str]] = []

    # -------------------------------------------------------------------------
    # PASS 1: VLM_Baseline -> prompt-boundary observation KV
    # -------------------------------------------------------------------------
    print("\n" + "=" * 116)
    print("PART2 FEATURE PASS 1/2 | traj_only | VLM_Baseline prompt-boundary KV")
    print("=" * 116)
    print("Model     :", args.baseline_vlm)
    print("Samples   :", len(rows))
    print("Precision :", args.vlm_dtype)
    print("Attention :", args.attn)

    baseline_processor = traj_cache_tools.load_processor(args.baseline_vlm)
    baseline_model = traj_cache_tools.load_model(
        args.baseline_vlm,
        device,
        vlm_dtype,
        args.attn,
    )

    for index, row in enumerate(rows, 1):
        sample_id = str(row["id"])
        out_path = traj_dir / cache_name(index, sample_id)

        try:
            t0 = time.perf_counter()
            record = traj_cache_tools.extract_one(
                model=baseline_model,
                processor=baseline_processor,
                row=row,
                device=device,
                dtype=vlm_dtype,
            )
            torch.save(record, out_path)
            baseline_records[sample_id] = {
                "cache_file": str(out_path),
                "kv_dim": int(record["observation_kv"].shape[-1]),
                "prompt_cache_length": int(record["prompt_cache_length"]),
            }

            print(
                f"[traj {index:03d}/{len(rows):03d}] "
                f"id={sample_id} "
                f"T={record['observation_kv'].shape[0]} "
                f"D={record['observation_kv'].shape[1]} "
                f"{time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        except Exception as exc:
            skips.append(
                {
                    "id": sample_id,
                    "stage": "traj_only",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[traj {index:03d}/{len(rows):03d}] "
                f"SKIP id={sample_id} | {type(exc).__name__}: {exc}",
                flush=True,
            )

    del baseline_model, baseline_processor
    cleanup_cuda()

    if not baseline_records:
        raise RuntimeError("No traj_only Part2 feature was successfully created")

    # -------------------------------------------------------------------------
    # PASS 2: Reasoning_VLM -> direct KV + generated reasoning delta KV
    # -------------------------------------------------------------------------
    print("\n" + "=" * 116)
    print("PART2 FEATURE PASS 2/2 | coc_reasoning | Reasoning_VLM generation KV")
    print("=" * 116)
    print("Model     :", args.reasoning_vlm)
    print("Candidates:", len(baseline_records))
    print("Max new   :", args.max_new_tokens)
    print("Precision :", args.vlm_dtype)
    print("Attention :", args.attn)

    reasoning_processor = reasoning_cache_tools.load_processor(args.reasoning_vlm)
    reasoning_model = reasoning_cache_tools.load_model(
        args.reasoning_vlm,
        device,
        vlm_dtype,
        args.attn,
    )

    manifest: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        sample_id = str(row["id"])
        baseline_entry = baseline_records.get(sample_id)
        if baseline_entry is None:
            continue

        out_path = reason_dir / cache_name(index, sample_id)

        try:
            t0 = time.perf_counter()
            record = reasoning_cache_tools.extract_one(
                model=reasoning_model,
                processor=reasoning_processor,
                row=row,
                device=device,
                dtype=vlm_dtype,
                max_new_tokens=args.max_new_tokens,
                allow_truncated=args.allow_truncated,
            )
            torch.save(record, out_path)

            reason_dim = int(record["direct_kv"].shape[-1])
            if reason_dim != int(baseline_entry["kv_dim"]):
                raise RuntimeError(
                    f"KV dim mismatch: VLM_Baseline={baseline_entry['kv_dim']} "
                    f"Reasoning_VLM={reason_dim}"
                )

            baseline_cache = torch.load(
                baseline_entry["cache_file"],
                map_location="cpu",
                weights_only=False,
            )
            baseline_gt = baseline_cache["trajectory"].float()
            reason_gt = record["trajectory"].float()
            max_gt_diff = float((baseline_gt - reason_gt).abs().max().item())
            if max_gt_diff > 1e-6:
                raise RuntimeError(
                    f"GT mismatch between branches: max_diff={max_gt_diff}"
                )

            manifest.append(
                {
                    "id": sample_id,
                    "clip": str(row.get("clip", "")),
                    "scenario_type": str(row.get("scenario_type", "")),
                    "traj_cache_file": str(baseline_entry["cache_file"]),
                    "reasoning_cache_file": str(out_path),
                    "kv_dim": reason_dim,
                    "traj_prompt_cache_length": int(
                        baseline_entry["prompt_cache_length"]
                    ),
                    "reason_prompt_cache_length": int(
                        record["prompt_cache_length"]
                    ),
                    "reasoning_cached_tokens": int(
                        record["reasoning_cached_tokens"]
                    ),
                    "reasoning_text": str(record.get("reasoning_text", "")),
                }
            )

            print(
                f"[coc  {index:03d}/{len(rows):03d}] "
                f"id={sample_id} "
                f"promptT={record['prompt_cache_length']} "
                f"reasonT={record['reasoning_cached_tokens']} "
                f"D={reason_dim} "
                f"{time.perf_counter() - t0:.2f}s",
                flush=True,
            )

            del baseline_cache, baseline_gt, reason_gt

        except Exception as exc:
            if out_path.exists():
                out_path.unlink()
            skips.append(
                {
                    "id": sample_id,
                    "stage": "coc_reasoning",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[coc  {index:03d}/{len(rows):03d}] "
                f"SKIP id={sample_id} | {type(exc).__name__}: {exc}",
                flush=True,
            )

    del reasoning_model, reasoning_processor
    cleanup_cuda()

    if not manifest:
        raise RuntimeError("No paired Part2 feature records were produced")

    write_jsonl(args.cache_root / "manifest.jsonl", manifest)
    write_jsonl(args.cache_root / "skipped.jsonl", skips)

    meta = {
        "request": request_meta,
        "requested_samples": len(rows),
        "paired_samples": len(manifest),
        "skipped": len(skips),
    }
    (args.cache_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[CACHE] complete")
    print("Requested :", len(rows))
    print("Paired    :", len(manifest))
    print("Skipped   :", len(skips))
    print("Root      :", args.cache_root)

    return manifest


# =============================================================================
# CHECKPOINT / INPUT LOADING
# =============================================================================

def load_checkpoint_model(
    checkpoint_path: Path,
    architecture: str,
    branch: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_branch = str(ckpt.get("branch", branch))
    if saved_branch != branch:
        raise RuntimeError(
            f"Checkpoint branch mismatch: expected={branch}, saved={saved_branch}, "
            f"path={checkpoint_path}"
        )

    saved_arch = ckpt.get("architecture")
    if saved_arch is not None and str(saved_arch) != architecture:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: expected={architecture}, "
            f"saved={saved_arch}, path={checkpoint_path}"
        )

    if "model_state_dict" not in ckpt:
        raise KeyError(f"model_state_dict missing: {checkpoint_path}")
    if "action_config" not in ckpt:
        raise KeyError(f"action_config missing: {checkpoint_path}")

    cfg = ckpt["action_config"]

    model = build_action_expert(
        architecture=architecture,
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    )

    state_dict = ckpt.pop("model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    model.to(device=device, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "epoch": int(ckpt.get("epoch", -1)),
        "input_dim": int(ckpt["input_dim"]),
        "trainable_params": int(ckpt.get("trainable_params", 0)),
        "val_metrics": dict(ckpt.get("val_metrics", {})),
        "checkpoint": str(checkpoint_path),
        "action_config": dict(cfg),
    }

    del ckpt
    return model, meta


def load_branch_sample(
    item: Dict[str, Any],
    branch: str,
    expected_input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if branch == "traj_only":
        cache = torch.load(
            item["traj_cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        memory = cache["observation_kv"].float()
        segment_ids = torch.zeros(memory.shape[0], dtype=torch.long)
        trajectory = cache["trajectory"].float()
    elif branch == "coc_reasoning":
        cache = torch.load(
            item["reasoning_cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        direct = cache["direct_kv"].float()
        delta = cache["reasoning_delta_kv"].float()

        if direct.ndim != 2 or delta.ndim != 2:
            raise RuntimeError(
                f"Bad reasoning KV rank for id={item['id']}: "
                f"direct={tuple(direct.shape)} delta={tuple(delta.shape)}"
            )
        if direct.shape[-1] != delta.shape[-1]:
            raise RuntimeError(
                f"Reasoning KV dim mismatch for id={item['id']}"
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
    else:
        raise ValueError(branch)

    if memory.ndim != 2:
        raise RuntimeError(
            f"memory must be [T,D], id={item['id']}, got={tuple(memory.shape)}"
        )
    if int(memory.shape[-1]) != int(expected_input_dim):
        raise RuntimeError(
            f"Input dim mismatch for id={item['id']}: "
            f"cache={memory.shape[-1]} checkpoint={expected_input_dim}"
        )
    if tuple(trajectory.shape) != (10, 3):
        raise RuntimeError(
            f"trajectory must be [10,3], id={item['id']}, "
            f"got={tuple(trajectory.shape)}"
        )

    return memory, segment_ids, trajectory


# =============================================================================
# MODEL EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate_condition(
    *,
    name: str,
    branch: str,
    architecture: str,
    checkpoint_path: Path,
    manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model, meta = load_checkpoint_model(
        checkpoint_path,
        architecture,
        branch,
        device,
    )

    # Warm-up on the same first input, excluded from latency / metrics.
    first_memory, first_seg, _ = load_branch_sample(
        manifest[0],
        branch,
        meta["input_dim"],
    )
    warm_memory = first_memory.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    warm_mask = torch.ones(
        (1, warm_memory.shape[1]),
        dtype=torch.bool,
        device=device,
    )
    warm_seg = first_seg.unsqueeze(0).to(device=device)

    for _ in range(max(0, warmup)):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(
                memory=warm_memory,
                memory_mask=warm_mask,
                segment_ids=warm_seg,
            )
    sync_cuda(device)

    del first_memory, first_seg, warm_memory, warm_mask, warm_seg

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    ids: List[str] = []
    latency_ms: List[float] = []

    print("\n" + "=" * 116)
    print(f"TEST {name}")
    print("=" * 116)
    print("Checkpoint :", checkpoint_path)
    print("Epoch      :", meta["epoch"])
    print("Params     :", f"{meta['trainable_params']:,}")
    print("Samples    :", len(manifest))

    for index, item in enumerate(manifest, 1):
        memory, segment_ids, trajectory = load_branch_sample(
            item,
            branch,
            meta["input_dim"],
        )

        memory = memory.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        memory_mask = torch.ones(
            (1, memory.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        segment_ids = segment_ids.unsqueeze(0).to(
            device=device,
            non_blocking=True,
        )
        gt = trajectory.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        # Measure Action Expert only. CPU->GPU transfer is excluded.
        sync_cuda(device)
        t0 = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(
                memory=memory,
                memory_mask=memory_mask,
                segment_ids=segment_ids,
            )

        sync_cuda(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_np = pred.float().cpu().numpy()
        gt_np = gt.float().cpu().numpy()

        preds.append(pred_np)
        gts.append(gt_np)
        ids.append(str(item["id"]))
        latency_ms.append(float(elapsed_ms))

        sample_metric = trajectory_metrics_np(pred_np, gt_np)
        print(
            f"[{index:03d}/{len(manifest):03d}] "
            f"id={item['id']} "
            f"ADE={sample_metric['ade_m']:.3f}m "
            f"FDE={sample_metric['fde_m']:.3f}m "
            f"AE={elapsed_ms:.2f}ms",
            flush=True,
        )

        del memory, memory_mask, segment_ids, gt, pred

    pred_all = np.concatenate(preds, axis=0)
    gt_all = np.concatenate(gts, axis=0)

    overall = trajectory_metrics_np(pred_all, gt_all)
    per_sample = per_sample_metrics_np(pred_all, gt_all)

    lat = np.asarray(latency_ms, dtype=np.float64)

    result: Dict[str, Any] = {
        "condition": name,
        "branch": branch,
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(meta["epoch"]),
        "trainable_params": int(meta["trainable_params"]),
        "val_ade_m": float(
            meta["val_metrics"].get("ade_m", float("nan"))
        ),
        "samples": len(ids),
        "ade_m": float(overall["ade_m"]),
        "fde_m": float(overall["fde_m"]),
        "heading_mae_rad": float(overall["heading_mae_rad"]),
        "mean_action_ms": float(lat.mean()),
        "p50_action_ms": float(np.percentile(lat, 50)),
        "p95_action_ms": float(np.percentile(lat, 95)),
        "ids": ids,
        "sample_ade_m": per_sample["ade_m"].tolist(),
        "sample_fde_m": per_sample["fde_m"].tolist(),
        "sample_heading_mae_rad": per_sample["heading_mae_rad"].tolist(),
    }

    samples: List[Dict[str, Any]] = []
    for i, sample_id in enumerate(ids):
        samples.append(
            {
                "condition": name,
                "id": sample_id,
                "ade_m": float(per_sample["ade_m"][i]),
                "fde_m": float(per_sample["fde_m"][i]),
                "heading_mae_rad": float(
                    per_sample["heading_mae_rad"][i]
                ),
                "action_ms": float(latency_ms[i]),
                "pred_trajectory": pred_all[i].tolist(),
                "gt_trajectory": gt_all[i].tolist(),
            }
        )

    del model, pred_all, gt_all, preds, gts
    cleanup_cuda()

    return result, samples


# =============================================================================
# PAIRED STATISTICS / OUTPUT
# =============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = values.size

    for i in range(samples):
        draw = rng.integers(0, n, size=n)
        means[i] = values[draw].mean()

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def build_pairwise_summary(
    results: Dict[str, Dict[str, Any]],
    bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    for arch_idx, architecture in enumerate(ARCHITECTURES):
        traj = results[f"traj_only_{architecture}"]
        coc = results[f"coc_reasoning_{architecture}"]

        if traj["ids"] != coc["ids"]:
            raise RuntimeError(
                f"Paired ID order mismatch for architecture={architecture}"
            )

        traj_ade = np.asarray(traj["sample_ade_m"], dtype=np.float64)
        coc_ade = np.asarray(coc["sample_ade_m"], dtype=np.float64)
        delta = coc_ade - traj_ade

        ci_lo, ci_hi = bootstrap_mean_ci(
            delta,
            samples=bootstrap,
            seed=seed + arch_idx,
        )

        base = float(traj["ade_m"])
        coc_mean = float(coc["ade_m"])
        improvement_pct = (
            100.0 * (base - coc_mean) / base
            if base != 0.0
            else float("nan")
        )

        summary[architecture] = {
            "traj_only_ade_m": base,
            "coc_reasoning_ade_m": coc_mean,
            "mean_delta_coc_minus_traj_m": float(delta.mean()),
            "relative_ade_improvement_pct": float(improvement_pct),
            "coc_win_rate_pct": float((delta < 0.0).mean() * 100.0),
            "bootstrap_95_ci_delta_m": [ci_lo, ci_hi],
        }

    return summary


def format_summary(
    results: Dict[str, Dict[str, Any]],
    pairwise: Dict[str, Dict[str, Any]],
    paired_samples: int,
) -> str:
    lines: List[str] = []

    lines += [
        "=" * 132,
        "CAPACITY-SCALED ACTION EXPERT 8-WAY | nuReasoning PART2 HELD-OUT TEST",
        "=" * 132,
        f"Paired Part2 samples : {paired_samples}",
        "",
        (
            f"{'CONDITION':<38}"
            f"{'PARAMS(B)':>11}"
            f"{'VAL_ADE':>11}"
            f"{'TEST_ADE':>11}"
            f"{'TEST_FDE':>11}"
            f"{'HEAD(rad)':>12}"
            f"{'AE(ms)':>11}"
        ),
        "-" * 132,
    ]

    for architecture in ARCHITECTURES:
        for branch in BRANCHES:
            name = f"{branch}_{architecture}"
            r = results[name]
            lines.append(
                f"{name:<38}"
                f"{r['trainable_params']/1e9:>11.3f}"
                f"{r['val_ade_m']:>11.4f}"
                f"{r['ade_m']:>11.4f}"
                f"{r['fde_m']:>11.4f}"
                f"{r['heading_mae_rad']:>12.4f}"
                f"{r['mean_action_ms']:>11.3f}"
            )

    lines += [
        "",
        "=" * 132,
        "CoC reasoning vs traj_only | SAME ARCHITECTURE | PART2 PAIRED TEST",
        "=" * 132,
    ]

    for architecture in ARCHITECTURES:
        x = pairwise[architecture]
        lo, hi = x["bootstrap_95_ci_delta_m"]
        lines.append(
            f"{architecture:<20}: "
            f"{x['traj_only_ade_m']:.4f}m -> "
            f"{x['coc_reasoning_ade_m']:.4f}m | "
            f"relative improvement={x['relative_ade_improvement_pct']:+.2f}% | "
            f"CoC win={x['coc_win_rate_pct']:.1f}% | "
            f"delta(CoC-Traj)={x['mean_delta_coc_minus_traj_m']:+.4f}m | "
            f"95% CI=[{lo:+.4f}, {hi:+.4f}]"
        )

    ranking = sorted(
        results.items(),
        key=lambda kv: kv[1]["ade_m"],
    )

    lines += [
        "",
        "=" * 132,
        "PART2 ADE RANKING",
        "=" * 132,
    ]

    for rank, (name, r) in enumerate(ranking, 1):
        lines.append(
            f"{rank:>2d}. {name:<38} "
            f"ADE={r['ade_m']:.4f}m "
            f"FDE={r['fde_m']:.4f}m "
            f"Heading={r['heading_mae_rad']:.4f}rad"
        )

    lines += [
        "",
        "Interpretation:",
        "  relative improvement > 0  -> CoC reasoning improved ADE.",
        "  delta(CoC-Traj) < 0        -> CoC reasoning is better.",
        "  95% CI entirely below 0   -> paired Part2 samples consistently support the CoC improvement.",
        "  AE(ms)                     -> Action Expert forward only; VLM feature extraction is excluded.",
    ]

    return "\n".join(lines)


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=Path, default=PART2_DATASET)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--baseline-vlm", type=Path, default=BASELINE_VLM)
    p.add_argument("--reasoning-vlm", type=Path, default=REASONING_VLM)
    p.add_argument(
        "--vlm-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    p.add_argument("--attn", default="sdpa")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    p.add_argument("--allow-truncated", action="store_true")

    p.add_argument(
        "--existing-root",
        type=Path,
        default=EXISTING_DECODER_CROSS_ROOT,
    )
    p.add_argument(
        "--remaining-root",
        type=Path,
        default=REMAINING6_ROOT,
    )

    p.add_argument("--cache-root", type=Path, default=FEATURE_CACHE_ROOT)
    p.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    p.add_argument("--rebuild-cache", action="store_true")

    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be >= 0")

    for path in (
        args.dataset,
        args.baseline_vlm,
        args.reasoning_vlm,
        args.existing_root,
        args.remaining_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    # Fail before expensive VLM extraction if even one of the eight checkpoints
    # is missing.
    checkpoint_map: Dict[str, Path] = {}
    for architecture in ARCHITECTURES:
        for branch in BRANCHES:
            name = f"{branch}_{architecture}"
            ckpt = resolve_checkpoint(
                args.existing_root,
                args.remaining_root,
                branch,
                architecture,
            )
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint for {name}: {ckpt}"
                )
            checkpoint_map[name] = ckpt

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    vlm_dtype = resolve_dtype(args.vlm_dtype)
    if vlm_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 selected but this GPU does not support BF16. "
            "Use the RTX 3080 Ti or pass --vlm-dtype fp16."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_rows = read_jsonl(args.dataset)
    selected_rows = select_rows(
        all_rows,
        limit=args.limit,
        seed=args.seed,
    )

    print("=" * 116)
    print("nuReasoning PART2 | CAPACITY-SCALED ACTION EXPERT 8-WAY TEST")
    print("=" * 116)
    print("GPU              :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset          :", args.dataset)
    print("Available rows   :", len(all_rows))
    print("Requested subset :", len(selected_rows))
    print("Sampling         :", f"deterministic random seed={args.seed}")
    print("VLM_Baseline     :", args.baseline_vlm)
    print("Reasoning_VLM    :", args.reasoning_vlm)
    print("VLM precision    :", args.vlm_dtype)
    print("VLM attention    :", args.attn)
    print("Feature cache    :", args.cache_root)
    print("Model loading    : ONE Action Expert at a time")
    print()

    manifest = build_feature_cache(
        args,
        selected_rows,
        device,
        vlm_dtype,
    )

    # Cross-check all paired feature dimensions before loading a checkpoint.
    dims = {int(x["kv_dim"]) for x in manifest}
    if len(dims) != 1:
        raise RuntimeError(f"Mixed KV dimensions in Part2 cache: {sorted(dims)}")
    cache_input_dim = next(iter(dims))

    for name, ckpt_path in checkpoint_map.items():
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        ckpt_dim = int(ckpt["input_dim"])
        del ckpt
        if ckpt_dim != cache_input_dim:
            raise RuntimeError(
                f"KV dim mismatch before test: {name} checkpoint={ckpt_dim}, "
                f"Part2 cache={cache_input_dim}"
            )

    results: Dict[str, Dict[str, Any]] = {}
    all_sample_results: List[Dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        for branch in BRANCHES:
            name = f"{branch}_{architecture}"

            result, sample_results = evaluate_condition(
                name=name,
                branch=branch,
                architecture=architecture,
                checkpoint_path=checkpoint_map[name],
                manifest=manifest,
                device=device,
                warmup=args.warmup,
            )
            results[name] = result
            all_sample_results.extend(sample_results)

    pairwise = build_pairwise_summary(
        results,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    summary_text = format_summary(
        results,
        pairwise,
        paired_samples=len(manifest),
    )

    print("\n" + summary_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.result_root / (
        f"part2_capacity8_{len(manifest)}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_payload = {
        "dataset": str(args.dataset),
        "requested_samples": int(args.limit),
        "paired_samples": len(manifest),
        "seed": int(args.seed),
        "baseline_vlm": str(args.baseline_vlm),
        "reasoning_vlm": str(args.reasoning_vlm),
        "vlm_dtype": args.vlm_dtype,
        "attention": args.attn,
        "feature_cache": str(args.cache_root),
        "results": results,
        "paired_comparisons": pairwise,
    }

    (run_dir / "summary.txt").write_text(
        summary_text + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "samples.jsonl", all_sample_results)
    write_jsonl(run_dir / "test_manifest.jsonl", manifest)

    config = {
        "argv": sys.argv,
        "existing_root": str(args.existing_root),
        "remaining_root": str(args.remaining_root),
        "cache_root": str(args.cache_root),
        "result_root": str(args.result_root),
        "warmup": int(args.warmup),
        "bootstrap": int(args.bootstrap),
        "max_new_tokens": int(args.max_new_tokens),
        "allow_truncated": bool(args.allow_truncated),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRESULT SAVED")
    print("Run dir :", run_dir)
    print("Summary :", run_dir / "summary.txt")
    print("JSON    :", run_dir / "summary.json")
    print("Samples :", run_dir / "samples.jsonl")


if __name__ == "__main__":
    main()
