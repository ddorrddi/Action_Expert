#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final Action Expert evaluator.

Current:
    Transformer 8-way

Future:
    Flow / DiT 2-way

Total registry:
    10 model slots

Final naming axes
-----------------
1. Input
    - Direct KV
    - Direct + Reasoning KV

2. Structure
    - Encoder
    - Decoder

3. Attention mask
    - Bidirectional
    - Causal

Transformer mapping
-------------------
Legacy architecture      Final interpretation
---------------------------------------------------------
encoder_self             Encoder + Bidirectional
encoder_cross            Decoder + Bidirectional
decoder_self             Encoder + Causal
decoder_cross            Decoder + Causal

Branches
--------
traj_only:
    Reasoning_VLM prompt-boundary direct_kv only

coc_reasoning:
    Reasoning_VLM direct_kv + reasoning_delta_kv

Current VLM/cache provenance
----------------------------
VLM:
    /home/lhh/lab/models/vlm/Reasoning_VLM

Cache version:
    action8_generation_kv_v1

VLM prompt:
    3 cameras
    + speed_mps
    + route command
    + driving reasoning instruction

Test:
    Part2 held-out 300

DFlash:
    NOT USED

Important
---------
This script DOES NOT regenerate VLM KV.

It reuses the existing ActionExpert8 test KV cache so that
the tested Action Experts receive the same KV representation
used by the controlled Reasoning_VLM experiments.

Current Transformer 8 models are tested automatically.

Future Flow/DiT checkpoints are also already registered.
If the corresponding best.pt files do not exist, they are SKIPPED.
Once they are added, no code change is required.
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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


# =============================================================================
# COMMON TRAINING / EVALUATION IMPLEMENTATION
# =============================================================================
#
# We reuse only dataset/evaluation utilities here.
#
# IMPORTANT:
# We DO NOT use final_train_all.extract_one() here because the older
# Reasoning_VLM_1200 experiment used a different prompt/input setup.
#
# The VLM KV itself is loaded from the already generated ActionExpert8
# test cache.
# =============================================================================

from final_train_all import (
    TRAIN_JSONL,
    VAL_JSONL,
    SEED,
    VAL_NOISE_SEED,
    read_jsonl,
    load_split,
    build_loader,
    move_batch,
    eval_direct,
    eval_flow,
    set_seed,
    cleanup_cuda,
)


# =============================================================================
# ACTION MODEL IMPLEMENTATIONS
# =============================================================================

from action_model_compare_8way import (
    build_action_expert as build_direct,
)

from action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    euler_sample,
)


# =============================================================================
# CURRENT EXPERIMENT PROVENANCE
# =============================================================================

EXPECTED_VLM_PATH = Path(
    "/home/lhh/lab/models/vlm/Reasoning_VLM"
)

EXPECTED_CACHE_VERSION = "action8_generation_kv_v1"

EXPECTED_TEST_SAMPLES = 300


# =============================================================================
# PATHS
# =============================================================================

TEST_JSONL = Path(
    "/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl"
)

# Current Reasoning_VLM / ActionExpert8 cache.
#
# Existing structure:
#
#   action_kv_cache/
#       train/
#       val/
#       test/
#           manifest.jsonl
#           meta.json
#           ...
#
TEST_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/"
    "ActionExpert8/action_kv_cache/test"
)

FINAL_MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert_final"
)

RESULT_ROOT = (
    FINAL_MODEL_ROOT
    / "part2_test"
)


# =============================================================================
# TEST CONFIG
# =============================================================================

GPU_ID = 0

FLOW_SOLVER_STEPS = 10

LATENCY_WARMUP = 5


# =============================================================================
# FINAL 10-MODEL REGISTRY
# =============================================================================
#
# Current:
#   Transformer 8 models
#
# Future:
#   Flow / DiT 2 models
#
# Missing best.pt -> automatic SKIP.
#
# Once Flow checkpoints are copied to:
#
#   action_expert_final/direct_kv_flow_dit/best.pt
#   action_expert_final/reasoning_kv_flow_dit/best.pt
#
# they will automatically be evaluated.
# =============================================================================

MODEL_SPECS: Sequence[Dict[str, Any]] = (

    # =========================================================================
    # Transformer | Direct KV
    # =========================================================================

    {
        "family": "transformer",
        "name": "direct_kv_encoder_bidirectional",
        "input_type": "Direct KV",
        "structure": "Encoder",
        "attention_mask": "Bidirectional",
        "branch": "traj_only",
        "legacy_architecture": "encoder_self",
    },

    {
        "family": "transformer",
        "name": "direct_kv_decoder_bidirectional",
        "input_type": "Direct KV",
        "structure": "Decoder",
        "attention_mask": "Bidirectional",
        "branch": "traj_only",
        "legacy_architecture": "encoder_cross",
    },

    {
        "family": "transformer",
        "name": "direct_kv_encoder_causal",
        "input_type": "Direct KV",
        "structure": "Encoder",
        "attention_mask": "Causal",
        "branch": "traj_only",
        "legacy_architecture": "decoder_self",
    },

    {
        "family": "transformer",
        "name": "direct_kv_decoder_causal",
        "input_type": "Direct KV",
        "structure": "Decoder",
        "attention_mask": "Causal",
        "branch": "traj_only",
        "legacy_architecture": "decoder_cross",
    },


    # =========================================================================
    # Transformer | Direct + Reasoning KV
    # =========================================================================

    {
        "family": "transformer",
        "name": "reasoning_kv_encoder_bidirectional",
        "input_type": "Direct + Reasoning KV",
        "structure": "Encoder",
        "attention_mask": "Bidirectional",
        "branch": "coc_reasoning",
        "legacy_architecture": "encoder_self",
    },

    {
        "family": "transformer",
        "name": "reasoning_kv_decoder_bidirectional",
        "input_type": "Direct + Reasoning KV",
        "structure": "Decoder",
        "attention_mask": "Bidirectional",
        "branch": "coc_reasoning",
        "legacy_architecture": "encoder_cross",
    },

    {
        "family": "transformer",
        "name": "reasoning_kv_encoder_causal",
        "input_type": "Direct + Reasoning KV",
        "structure": "Encoder",
        "attention_mask": "Causal",
        "branch": "coc_reasoning",
        "legacy_architecture": "decoder_self",
    },

    {
        "family": "transformer",
        "name": "reasoning_kv_decoder_causal",
        "input_type": "Direct + Reasoning KV",
        "structure": "Decoder",
        "attention_mask": "Causal",
        "branch": "coc_reasoning",
        "legacy_architecture": "decoder_cross",
    },


    # =========================================================================
    # Flow / DiT | Future
    # =========================================================================

    {
        "family": "flow",
        "name": "direct_kv_flow_dit",
        "input_type": "Direct KV",
        "structure": "Flow/DiT",
        "attention_mask": "Non-causal",
        "branch": "traj_only",
        "legacy_architecture": None,
    },

    {
        "family": "flow",
        "name": "reasoning_kv_flow_dit",
        "input_type": "Direct + Reasoning KV",
        "structure": "Flow/DiT",
        "attention_mask": "Non-causal",
        "branch": "coc_reasoning",
        "legacy_architecture": None,
    },
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ids_from_jsonl(
    path: Path,
) -> set[str]:

    return {
        str(row["id"])
        for row in read_jsonl(path)
        if row.get("id") is not None
    }


def validate_test_no_leakage(
    test_rows,
) -> None:

    test_ids = {
        str(row["id"])
        for row in test_rows
    }

    train_ids = ids_from_jsonl(
        TRAIN_JSONL
    )

    val_ids = ids_from_jsonl(
        VAL_JSONL
    )

    overlap_train = (
        test_ids
        & train_ids
    )

    overlap_val = (
        test_ids
        & val_ids
    )

    if overlap_train:
        raise RuntimeError(
            "Part2 TEST overlaps TRAIN IDs:\n"
            f"{sorted(overlap_train)[:20]}"
        )

    if overlap_val:
        raise RuntimeError(
            "Part2 TEST overlaps VAL IDs:\n"
            f"{sorted(overlap_val)[:20]}"
        )

    print(
        "[LEAKAGE] PASS | "
        f"test={len(test_ids)} "
        f"train_overlap=0 "
        f"val_overlap=0"
    )


# =============================================================================
# CACHE PROVENANCE HELPERS
# =============================================================================

def collect_string_values(
    obj: Any,
) -> List[str]:

    out: List[str] = []

    if isinstance(obj, dict):

        for key, value in obj.items():
            out.append(str(key))
            out.extend(
                collect_string_values(
                    value
                )
            )

    elif isinstance(
        obj,
        (list, tuple),
    ):

        for value in obj:
            out.extend(
                collect_string_values(
                    value
                )
            )

    elif obj is not None:
        out.append(str(obj))

    return out


def validate_cache_meta(
    cache_root: Path,
) -> None:

    meta_candidates = [
        cache_root / "meta.json",
        cache_root.parent / "meta.json",
    ]

    metas = []

    for path in meta_candidates:

        if not path.is_file():
            continue

        try:
            meta = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            print(
                f"[CACHE META] warning | "
                f"could not parse {path}: {exc}"
            )
            continue

        metas.append(
            (path, meta)
        )

    if not metas:
        print(
            "[CACHE META] warning | "
            "no meta.json found"
        )
        return

    all_strings: List[str] = []

    for path, meta in metas:
        print(
            "[CACHE META] found |",
            path,
        )

        all_strings.extend(
            collect_string_values(
                meta
            )
        )

    joined = "\n".join(
        all_strings
    )

    # -------------------------------------------------------------------------
    # Explicitly reject the old Reasoning_VLM_1200 provenance.
    # -------------------------------------------------------------------------

    if "Reasoning_VLM_1200" in joined:

        raise RuntimeError(
            "\n"
            "WRONG TEST CACHE DETECTED\n"
            "------------------------------------------------------------\n"
            "The selected cache contains Reasoning_VLM_1200 provenance.\n"
            "\n"
            "Current final 8-way comparison requires:\n"
            f"    {EXPECTED_VLM_PATH}\n"
            "\n"
            "Do not test the current models with the old 1200 cache.\n"
        )

    # -------------------------------------------------------------------------
    # Positive provenance checks when metadata exposes these fields.
    # -------------------------------------------------------------------------

    expected_vlm = str(
        EXPECTED_VLM_PATH
    )

    if expected_vlm in joined:
        print(
            "[CACHE META] VLM provenance PASS |",
            expected_vlm,
        )
    else:
        print(
            "[CACHE META] VLM path not explicitly "
            "present in metadata; continuing with "
            "record-level validation."
        )

    if EXPECTED_CACHE_VERSION in joined:
        print(
            "[CACHE META] cache version PASS |",
            EXPECTED_CACHE_VERSION,
        )
    else:
        print(
            "[CACHE META] cache version not explicitly "
            "present in this split metadata."
        )


# =============================================================================
# TEST CACHE LOADING
# =============================================================================

def resolve_manifest_cache_file(
    raw_path: str,
    cache_root: Path,
) -> Path:

    path = Path(
        raw_path
    )

    if path.is_absolute():
        return path

    candidate = (
        cache_root
        / path
    )

    if candidate.is_file():
        return candidate

    candidate_parent = (
        cache_root.parent
        / path
    )

    if candidate_parent.is_file():
        return candidate_parent

    # Return first candidate so the caller gives
    # a useful FileNotFoundError.
    return candidate


def load_and_validate_test_cache(
    args,
    test_rows,
):

    manifest_path = (
        args.cache_root
        / "manifest.jsonl"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "\nTest cache manifest not found:\n"
            f"    {manifest_path}\n"
        )

    validate_cache_meta(
        args.cache_root
    )

    manifest_raw = read_jsonl(
        manifest_path
    )

    if not manifest_raw:
        raise RuntimeError(
            f"Empty manifest: {manifest_path}"
        )

    # -------------------------------------------------------------------------
    # Check duplicate IDs.
    # -------------------------------------------------------------------------

    cache_by_id: Dict[str, Dict[str, Any]] = {}

    duplicate_ids: List[str] = []

    for item in manifest_raw:

        sid = str(
            item["id"]
        )

        if sid in cache_by_id:
            duplicate_ids.append(
                sid
            )

        cache_by_id[sid] = dict(
            item
        )

    if duplicate_ids:
        raise RuntimeError(
            "Duplicate IDs in test cache:\n"
            f"{sorted(set(duplicate_ids))[:20]}"
        )

    # -------------------------------------------------------------------------
    # Compare cache IDs against Part2 held-out JSONL.
    # -------------------------------------------------------------------------

    expected_ids = [
        str(row["id"])
        for row in test_rows
    ]

    expected_set = set(
        expected_ids
    )

    cache_set = set(
        cache_by_id.keys()
    )

    missing = (
        expected_set
        - cache_set
    )

    extra = (
        cache_set
        - expected_set
    )

    if missing or extra:

        raise RuntimeError(
            "\n"
            "TEST CACHE ID MISMATCH\n"
            "------------------------------------------------------------\n"
            f"Expected : {len(expected_ids)}\n"
            f"Cached   : {len(cache_by_id)}\n"
            f"Missing  : {len(missing)}\n"
            f"Extra    : {len(extra)}\n"
            "\n"
            f"Missing examples: {sorted(missing)[:10]}\n"
            f"Extra examples  : {sorted(extra)[:10]}\n"
        )

    # -------------------------------------------------------------------------
    # Reorder cache to exact Part2 JSONL order.
    #
    # Important for deterministic Flow comparisons.
    # -------------------------------------------------------------------------

    manifest = []

    for sid in expected_ids:

        item = dict(
            cache_by_id[sid]
        )

        if "cache_file" not in item:
            raise RuntimeError(
                f"cache_file missing for id={sid}"
            )

        cache_file = resolve_manifest_cache_file(
            str(item["cache_file"]),
            args.cache_root,
        )

        if not cache_file.is_file():
            raise FileNotFoundError(
                f"Missing cache file:\n"
                f"    id={sid}\n"
                f"    {cache_file}"
            )

        item["cache_file"] = str(
            cache_file
        )

        manifest.append(
            item
        )

    print(
        "[CACHE IDs] PASS | "
        f"samples={len(manifest)} "
        f"order=Part2 JSONL"
    )

    # -------------------------------------------------------------------------
    # Inspect first cache record.
    # -------------------------------------------------------------------------

    first_path = Path(
        manifest[0]["cache_file"]
    )

    first = torch.load(
        first_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = (
        "direct_kv",
        "reasoning_delta_kv",
    )

    for key in required_keys:

        if key not in first:
            raise RuntimeError(
                f"Current ActionExpert8 cache "
                f"must contain '{key}'.\n"
                f"File: {first_path}"
            )

    direct_kv = first[
        "direct_kv"
    ]

    reasoning_kv = first[
        "reasoning_delta_kv"
    ]

    if not torch.is_tensor(
        direct_kv
    ):
        raise TypeError(
            "direct_kv is not a tensor"
        )

    if not torch.is_tensor(
        reasoning_kv
    ):
        raise TypeError(
            "reasoning_delta_kv is not a tensor"
        )

    if direct_kv.ndim != 2:
        raise RuntimeError(
            "direct_kv must have shape [T, D], "
            f"got {tuple(direct_kv.shape)}"
        )

    if reasoning_kv.ndim != 2:
        raise RuntimeError(
            "reasoning_delta_kv must have shape [T, D], "
            f"got {tuple(reasoning_kv.shape)}"
        )

    if (
        direct_kv.shape[-1]
        != reasoning_kv.shape[-1]
    ):
        raise RuntimeError(
            "direct/reasoning KV dimension mismatch:\n"
            f"    direct    = {tuple(direct_kv.shape)}\n"
            f"    reasoning = {tuple(reasoning_kv.shape)}"
        )

    if reasoning_kv.shape[0] <= 0:
        raise RuntimeError(
            "reasoning_delta_kv is empty. "
            "Reasoning-KV models cannot be tested."
        )

    kv_dim = int(
        direct_kv.shape[-1]
    )

    print(
        "[CACHE RECORD] PASS | "
        f"directT={direct_kv.shape[0]} "
        f"reasonT={reasoning_kv.shape[0]} "
        f"D={kv_dim}"
    )

    if "speed_mps" in first:
        print(
            "[CACHE RECORD] speed_mps metadata present"
        )
    else:
        print(
            "[CACHE RECORD] warning | "
            "speed_mps metadata not stored in record"
        )

    if "command" in first:
        print(
            "[CACHE RECORD] route command metadata present"
        )
    else:
        print(
            "[CACHE RECORD] warning | "
            "command metadata not stored in record"
        )

    del first
    gc.collect()

    return manifest, kv_dim


# =============================================================================
# MODEL DISCOVERY
# =============================================================================

def checkpoint_path_for_spec(
    model_root: Path,
    spec: Dict[str, Any],
) -> Path:

    return (
        model_root
        / spec["name"]
        / "best.pt"
    )


def discover_models(
    model_root: Path,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    found = []
    missing = []

    print()
    print("=" * 120)
    print("MODEL DISCOVERY | FINAL 10-SLOT REGISTRY")
    print("=" * 120)

    for index, spec in enumerate(
        MODEL_SPECS,
        1,
    ):

        checkpoint = (
            checkpoint_path_for_spec(
                model_root,
                spec,
            )
        )

        row = dict(
            spec
        )

        row["checkpoint"] = str(
            checkpoint
        )

        if checkpoint.is_file():

            found.append(
                row
            )

            print(
                f"[FOUND {index:02d}/10] "
                f"{spec['name']}"
            )

        else:

            missing.append(
                row
            )

            print(
                f"[MISS  {index:02d}/10] "
                f"{spec['name']}"
            )

    print()
    print(
        f"Available : {len(found)}/10"
    )

    print(
        f"Missing   : {len(missing)}/10"
    )

    return (
        found,
        missing,
    )


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def get_model_state_dict(
    checkpoint: Dict[str, Any],
):

    if "model_state_dict" in checkpoint:
        return checkpoint[
            "model_state_dict"
        ]

    if "state_dict" in checkpoint:
        return checkpoint[
            "state_dict"
        ]

    raise KeyError(
        "Checkpoint contains neither "
        "'model_state_dict' nor 'state_dict'"
    )


def get_checkpoint_epoch(
    checkpoint: Dict[str, Any],
) -> int:

    value = checkpoint.get(
        "epoch",
        -1,
    )

    try:
        return int(
            value
        )
    except Exception:
        return -1


# =============================================================================
# TRANSFORMER MODEL LOADING
# =============================================================================

def load_transformer_checkpoint(
    checkpoint_path: Path,
    spec: Dict[str, Any],
    device: torch.device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "action_config" not in checkpoint:
        raise KeyError(
            f"action_config missing:\n"
            f"    {checkpoint_path}"
        )

    cfg = checkpoint[
        "action_config"
    ]

    expected_architecture = str(
        spec["legacy_architecture"]
    )

    actual_architecture = str(
        checkpoint.get(
            "architecture",
            expected_architecture,
        )
    )

    # -------------------------------------------------------------------------
    # Safety check:
    # ensure the copied checkpoint actually matches its final folder name.
    # -------------------------------------------------------------------------

    if (
        actual_architecture
        != expected_architecture
    ):

        raise RuntimeError(
            "\n"
            "CHECKPOINT ARCHITECTURE MISMATCH\n"
            "------------------------------------------------------------\n"
            f"Final model : {spec['name']}\n"
            f"Expected    : {expected_architecture}\n"
            f"Checkpoint  : {actual_architecture}\n"
            f"Path        : {checkpoint_path}\n"
            "\n"
            "The wrong best.pt may have been copied into this folder.\n"
        )

    if "input_dim" not in checkpoint:
        raise KeyError(
            f"input_dim missing:\n"
            f"    {checkpoint_path}"
        )

    model = build_direct(
        architecture=actual_architecture,
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dim=int(
            cfg["hidden_dim"]
        ),
        num_steps=int(
            cfg["num_steps"]
        ),
        num_layers=int(
            cfg["num_layers"]
        ),
        num_heads=int(
            cfg["num_heads"]
        ),
        ff_dim=int(
            cfg["ff_dim"]
        ),
        dropout=float(
            cfg["dropout"]
        ),
        gradient_checkpointing=False,
    )

    model.load_state_dict(
        get_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    return (
        model,
        checkpoint,
        actual_architecture,
    )


# =============================================================================
# FLOW / DIT MODEL LOADING
# =============================================================================

def load_flow_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "action_config" not in checkpoint:
        raise KeyError(
            f"action_config missing:\n"
            f"    {checkpoint_path}"
        )

    if "input_dim" not in checkpoint:
        raise KeyError(
            f"input_dim missing:\n"
            f"    {checkpoint_path}"
        )

    cfg = checkpoint[
        "action_config"
    ]

    model = build_flow_dit(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dim=int(
            cfg["hidden_dim"]
        ),
        num_steps=int(
            cfg["num_steps"]
        ),
        num_layers=int(
            cfg["num_layers"]
        ),
        num_heads=int(
            cfg["num_heads"]
        ),
        ff_dim=int(
            cfg["ff_dim"]
        ),
        dropout=float(
            cfg["dropout"]
        ),
        gradient_checkpointing=False,
    )

    model.load_state_dict(
        get_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    if "normalizer" not in checkpoint:

        raise KeyError(
            f"normalizer missing:\n"
            f"    {checkpoint_path}"
        )

    normalizer_dict = checkpoint[
        "normalizer"
    ]

    if (
        "mean" not in normalizer_dict
        or "std" not in normalizer_dict
    ):

        raise RuntimeError(
            "Flow checkpoint normalizer "
            "must contain mean/std"
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

    return (
        model,
        normalizer,
        checkpoint,
    )


# =============================================================================
# TRANSFORMER LATENCY
# =============================================================================

@torch.inference_mode()
def benchmark_direct_latency(
    model,
    loader,
    device,
    warmup: int = 5,
):

    times_ms = []

    for index, batch in enumerate(
        loader
    ):

        inputs, _ = move_batch(
            batch,
            device,
        )

        if index < warmup:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                _ = model(
                    **inputs
                )

            continue

        torch.cuda.synchronize(
            device
        )

        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                device.type
                == "cuda"
            ),
        ):
            _ = model(
                **inputs
            )

        torch.cuda.synchronize(
            device
        )

        times_ms.append(
            (
                time.perf_counter()
                - t0
            )
            * 1000.0
        )

    if not times_ms:
        return math.nan

    return float(
        np.mean(
            times_ms
        )
    )


# =============================================================================
# FLOW LATENCY
# =============================================================================

@torch.inference_mode()
def benchmark_flow_latency(
    model,
    loader,
    device,
    normalizer,
    solver_steps: int,
    warmup: int = 5,
):

    norm = normalizer.to(
        device
    )

    rng = torch.Generator(
        device="cpu"
    )

    rng.manual_seed(
        VAL_NOISE_SEED
    )

    times_ms = []

    for index, batch in enumerate(
        loader
    ):

        inputs, _ = move_batch(
            batch,
            device,
        )

        if index < warmup:

            _ = euler_sample(
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
                solver_steps=solver_steps,
                rng=rng,
            )

            continue

        torch.cuda.synchronize(
            device
        )

        t0 = time.perf_counter()

        _ = euler_sample(
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
            solver_steps=solver_steps,
            rng=rng,
        )

        torch.cuda.synchronize(
            device
        )

        times_ms.append(
            (
                time.perf_counter()
                - t0
            )
            * 1000.0
        )

    if not times_ms:
        return math.nan

    return float(
        np.mean(
            times_ms
        )
    )


# =============================================================================
# TRANSFORMER TEST
# =============================================================================

def test_transformer(
    spec: Dict[str, Any],
    manifest,
    device,
    args,
):

    name = str(
        spec["name"]
    )

    branch = str(
        spec["branch"]
    )

    checkpoint_path = (
        checkpoint_path_for_spec(
            args.model_root,
            spec,
        )
    )

    if not checkpoint_path.is_file():

        print(
            f"[SKIP] missing checkpoint | "
            f"{name}"
        )

        return None

    print()
    print("=" * 120)
    print(
        f"TEST TRANSFORMER | {name}"
    )
    print("=" * 120)

    print(
        "Checkpoint     :",
        checkpoint_path,
    )

    print(
        "Input          :",
        spec["input_type"],
    )

    print(
        "Structure      :",
        spec["structure"],
    )

    print(
        "Attention mask :",
        spec["attention_mask"],
    )

    print(
        "Legacy branch  :",
        branch,
    )

    model, checkpoint, architecture = (
        load_transformer_checkpoint(
            checkpoint_path=checkpoint_path,
            spec=spec,
            device=device,
        )
    )

    print(
        "Legacy arch    :",
        architecture,
        "(validated)",
    )

    _, loader = build_loader(
        manifest=manifest,
        branch=branch,
        batch_size=1,
        shuffle=False,
        seed=SEED,
    )

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    metrics = eval_direct(
        model,
        loader,
        device,
    )

    # -------------------------------------------------------------------------
    # Action Expert latency
    # -------------------------------------------------------------------------

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

        # -----------------------------------------------------
        # New final naming axes
        # -----------------------------------------------------

        "input_type": str(
            spec["input_type"]
        ),

        "structure": str(
            spec["structure"]
        ),

        "attention_mask": str(
            spec["attention_mask"]
        ),

        # -----------------------------------------------------
        # Legacy implementation identifiers
        # -----------------------------------------------------

        "branch": branch,

        "legacy_architecture": (
            architecture
        ),

        # -----------------------------------------------------
        # Checkpoint
        # -----------------------------------------------------

        "checkpoint": str(
            checkpoint_path
        ),

        "best_val_epoch": (
            get_checkpoint_epoch(
                checkpoint
            )
        ),

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        "test_loss": float(
            metrics["loss"]
        ),

        "test_ade_m": float(
            metrics["ade_m"]
        ),

        "test_fde_m": float(
            metrics["fde_m"]
        ),

        "test_heading_mae_rad": float(
            metrics[
                "heading_mae_rad"
            ]
        ),

        "action_expert_ms": float(
            ae_ms
        ),
    }

    del (
        model,
        checkpoint,
        loader,
    )

    cleanup_cuda()

    return result


# =============================================================================
# FLOW / DIT TEST
# =============================================================================

def test_flow(
    spec: Dict[str, Any],
    manifest,
    device,
    args,
):

    name = str(
        spec["name"]
    )

    branch = str(
        spec["branch"]
    )

    checkpoint_path = (
        checkpoint_path_for_spec(
            args.model_root,
            spec,
        )
    )

    if not checkpoint_path.is_file():

        print(
            f"[SKIP] missing checkpoint | "
            f"{name}"
        )

        return None

    print()
    print("=" * 120)
    print(
        f"TEST FLOW / DiT | {name}"
    )
    print("=" * 120)

    print(
        "Checkpoint     :",
        checkpoint_path,
    )

    print(
        "Input          :",
        spec["input_type"],
    )

    print(
        "Structure      :",
        spec["structure"],
    )

    print(
        "Attention mask :",
        spec["attention_mask"],
    )

    print(
        "Legacy branch  :",
        branch,
    )

    (
        model,
        normalizer,
        checkpoint,
    ) = load_flow_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    cfg = checkpoint[
        "action_config"
    ]

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

    # -------------------------------------------------------------------------
    # eval_flow() resets deterministic validation noise internally.
    #
    # Therefore Direct / Reasoning conditions receive the same
    # deterministic flow/noise sequence.
    # -------------------------------------------------------------------------

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
        f"[RESULT] {name} | "
        f"ADE={metrics['ade_m']:.4f}m "
        f"FDE={metrics['fde_m']:.4f}m "
        f"Heading={metrics['heading_mae_rad']:.4f}rad "
        f"FM={metrics['flow_mse']:.6f} "
        f"AE={ae_ms:.3f}ms"
    )

    result = {

        "family": "flow",

        "condition": name,

        "input_type": str(
            spec["input_type"]
        ),

        "structure": str(
            spec["structure"]
        ),

        "attention_mask": str(
            spec["attention_mask"]
        ),

        "branch": branch,

        "legacy_architecture": (
            "flow_dit"
        ),

        "checkpoint": str(
            checkpoint_path
        ),

        "best_val_epoch": (
            get_checkpoint_epoch(
                checkpoint
            )
        ),

        "solver_steps": int(
            solver_steps
        ),

        "test_flow_mse": float(
            metrics["flow_mse"]
        ),

        "test_ade_m": float(
            metrics["ade_m"]
        ),

        "test_fde_m": float(
            metrics["fde_m"]
        ),

        "test_heading_mae_rad": float(
            metrics[
                "heading_mae_rad"
            ]
        ),

        "action_expert_ms": float(
            ae_ms
        ),
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
# SUMMARY TABLE
# =============================================================================

def result_table_lines(
    results,
) -> List[str]:

    lines = []

    header = (
        f"{'FAMILY':<12}"
        f"{'INPUT':<24}"
        f"{'STRUCT':<12}"
        f"{'MASK':<15}"
        f"{'BEST_E':>8}"
        f"{'ADE(m)':>11}"
        f"{'FDE(m)':>11}"
        f"{'HEAD(rad)':>12}"
        f"{'AE(ms)':>11}"
    )

    lines.append(
        header
    )

    lines.append(
        "-" * len(header)
    )

    for r in results:

        lines.append(
            f"{r['family']:<12}"
            f"{r['input_type']:<24}"
            f"{r['structure']:<12}"
            f"{r['attention_mask']:<15}"
            f"{r['best_val_epoch']:>8d}"
            f"{r['test_ade_m']:>11.4f}"
            f"{r['test_fde_m']:>11.4f}"
            f"{r['test_heading_mae_rad']:>12.4f}"
            f"{r['action_expert_ms']:>11.3f}"
        )

    return lines


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    args,
    results,
    missing_specs,
    test_samples: int,
    kv_dim: int,
):

    args.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        args.result_root
        / f"run_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payload = {

        "experiment": (
            "action_expert_final_part2_test"
        ),

        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "test_jsonl": str(
            args.test_jsonl
        ),

        "test_samples": int(
            test_samples
        ),

        "test_cache_root": str(
            args.cache_root
        ),

        "expected_vlm": str(
            EXPECTED_VLM_PATH
        ),

        "expected_cache_version": (
            EXPECTED_CACHE_VERSION
        ),

        "kv_dim": int(
            kv_dim
        ),

        "vlm_prompt_input": (
            "3 cameras + speed_mps + route command"
        ),

        "dflash_used": False,

        "model_root": str(
            args.model_root
        ),

        "registered_models": int(
            len(MODEL_SPECS)
        ),

        "tested_models": int(
            len(results)
        ),

        "missing_models": [
            spec["name"]
            for spec in missing_specs
        ],

        "results": results,
    }

    json_path = (
        run_dir
        / "part2_300_results.json"
    )

    txt_path = (
        run_dir
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

        "=" * 130,

        (
            "FINAL ACTION EXPERT | "
            "PART2 HELD-OUT TEST"
        ),

        "=" * 130,

        (
            f"Expected VLM : "
            f"{EXPECTED_VLM_PATH}"
        ),

        (
            f"Cache ver    : "
            f"{EXPECTED_CACHE_VERSION}"
        ),

        (
            f"Test         : "
            f"{args.test_jsonl}"
        ),

        (
            f"Samples      : "
            f"{test_samples}"
        ),

        (
            f"KV dim       : "
            f"{kv_dim}"
        ),

        (
            "VLM input    : "
            "3 cameras + speed_mps + route command"
        ),

        (
            "Direct       : "
            "prompt-boundary direct_kv"
        ),

        (
            "Reasoning    : "
            "direct_kv + reasoning_delta_kv"
        ),

        "DFlash       : NOT USED",

        (
            f"Model root   : "
            f"{args.model_root}"
        ),

        (
            f"Tested       : "
            f"{len(results)}/"
            f"{len(MODEL_SPECS)}"
        ),

        "",
    ]

    lines.extend(
        result_table_lines(
            results
        )
    )

    if missing_specs:

        lines.append(
            ""
        )

        lines.append(
            "MISSING / SKIPPED MODELS"
        )

        lines.append(
            "-" * 80
        )

        for spec in missing_specs:

            lines.append(
                f"- {spec['name']}"
            )

    txt_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Also keep easy-to-find latest copies.
    # -------------------------------------------------------------------------

    latest_json = (
        args.result_root
        / "latest_results.json"
    )

    latest_txt = (
        args.result_root
        / "latest_results.txt"
    )

    shutil.copy2(
        json_path,
        latest_json,
    )

    shutil.copy2(
        txt_path,
        latest_txt,
    )

    return (
        run_dir,
        txt_path,
        json_path,
        latest_txt,
        latest_json,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=TEST_JSONL,
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=TEST_CACHE_ROOT,
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=FINAL_MODEL_ROOT,
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=GPU_ID,
    )

    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=LATENCY_WARMUP,
    )

    parser.add_argument(
        "--strict-10",
        action="store_true",
        help=(
            "Fail unless all 10 registered "
            "checkpoints exist."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    # =========================================================================
    # CUDA
    # =========================================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    torch.cuda.set_device(
        args.gpu_id
    )

    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    torch.backends.cudnn.allow_tf32 = (
        True
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    set_seed(
        SEED
    )

    # =========================================================================
    # PATH VALIDATION
    # =========================================================================

    for path in (
        args.test_jsonl,
        args.cache_root,
        args.model_root,
    ):

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # =========================================================================
    # TEST DATA
    # =========================================================================

    test_rows = load_split(
        args.test_jsonl
    )

    if (
        len(test_rows)
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "\n"
            "Unexpected Part2 sample count:\n"
            f"    expected = {EXPECTED_TEST_SAMPLES}\n"
            f"    actual   = {len(test_rows)}\n"
            f"    file     = {args.test_jsonl}\n"
        )

    validate_test_no_leakage(
        test_rows
    )

    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("=" * 120)
    print(
        "FINAL ACTION EXPERT | "
        "10-SLOT AUTOMATIC PART2 TEST"
    )
    print("=" * 120)

    print(
        "GPU          :",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )

    print(
        "VLM expected :",
        EXPECTED_VLM_PATH,
    )

    print(
        "Cache ver    :",
        EXPECTED_CACHE_VERSION,
    )

    print(
        "Test         :",
        args.test_jsonl,
    )

    print(
        "Samples      :",
        len(test_rows),
    )

    print(
        "Cache root   :",
        args.cache_root,
    )

    print(
        "Model root   :",
        args.model_root,
    )

    print(
        "Result root  :",
        args.result_root,
    )

    print(
        "VLM input    : "
        "3 cameras + speed_mps + route command"
    )

    print(
        "Direct input : "
        "prompt-boundary direct_kv"
    )

    print(
        "Reason input : "
        "direct_kv + reasoning_delta_kv"
    )

    print(
        "DFlash       : NOT USED"
    )

    # =========================================================================
    # CACHE
    # =========================================================================

    print()
    print("=" * 120)
    print(
        "STAGE 1 | VALIDATE EXISTING "
        "REASONING_VLM PART2 KV CACHE"
    )
    print("=" * 120)

    manifest, kv_dim = (
        load_and_validate_test_cache(
            args=args,
            test_rows=test_rows,
        )
    )

    print()
    print(
        f"[CACHE] READY | "
        f"samples={len(manifest)} "
        f"input_dim={kv_dim}"
    )

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    found_specs, missing_specs = (
        discover_models(
            args.model_root
        )
    )

    if args.strict_10 and missing_specs:

        names = "\n".join(
            f"    - {x['name']}"
            for x in missing_specs
        )

        raise RuntimeError(
            "\n"
            "--strict-10 enabled but "
            "some checkpoints are missing:\n"
            f"{names}\n"
        )

    if not found_specs:

        raise RuntimeError(
            "No Action Expert checkpoints found."
        )

    # =========================================================================
    # TEST
    # =========================================================================

    results = []

    print()
    print("=" * 120)
    print(
        "STAGE 2 | ACTION EXPERT EVALUATION"
    )
    print("=" * 120)

    for spec in MODEL_SPECS:

        checkpoint_path = (
            checkpoint_path_for_spec(
                args.model_root,
                spec,
            )
        )

        # ---------------------------------------------------------------------
        # Missing checkpoint -> intentional automatic skip.
        # ---------------------------------------------------------------------

        if not checkpoint_path.is_file():

            print()
            print(
                f"[SKIP] {spec['name']} | "
                f"checkpoint not found"
            )

            continue

        family = str(
            spec["family"]
        )

        if family == "transformer":

            result = test_transformer(
                spec=spec,
                manifest=manifest,
                device=device,
                args=args,
            )

        elif family == "flow":

            result = test_flow(
                spec=spec,
                manifest=manifest,
                device=device,
                args=args,
            )

        else:

            raise ValueError(
                f"Unknown model family: "
                f"{family}"
            )

        if result is not None:

            results.append(
                result
            )

    # =========================================================================
    # SAVE
    # =========================================================================

    (
        run_dir,
        txt_path,
        json_path,
        latest_txt,
        latest_json,
    ) = save_results(
        args=args,
        results=results,
        missing_specs=missing_specs,
        test_samples=len(
            test_rows
        ),
        kv_dim=kv_dim,
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 130)
    print(
        "FINAL PART2 TEST RESULTS"
    )
    print("=" * 130)

    for line in result_table_lines(
        results
    ):
        print(
            line
        )

    print()
    print(
        "Registered models :",
        len(MODEL_SPECS),
    )

    print(
        "Available models  :",
        len(found_specs),
    )

    print(
        "Tested models     :",
        len(results),
    )

    print(
        "Missing models    :",
        len(missing_specs),
    )

    if missing_specs:

        print()

        for spec in missing_specs:

            print(
                "[MISSING]",
                spec["name"],
            )

    print()
    print(
        "Run dir           :",
        run_dir,
    )

    print(
        "TXT               :",
        txt_path,
    )

    print(
        "JSON              :",
        json_path,
    )

    print(
        "Latest TXT        :",
        latest_txt,
    )

    print(
        "Latest JSON       :",
        latest_json,
    )

    # =========================================================================
    # Expected current state:
    #
    # Registered = 10
    # Available  = 8
    # Tested     = 8
    # Missing    = 2
    #
    # Missing:
    #   direct_kv_flow_dit
    #   reasoning_kv_flow_dit
    #
    # Later, once those two checkpoints exist:
    #
    # Registered = 10
    # Available  = 10
    # Tested     = 10
    # Missing    = 0
    # =========================================================================


if __name__ == "__main__":
    main()