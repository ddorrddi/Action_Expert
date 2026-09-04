#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train all 10 Action Expert models using the NEW fixed-split dataset and
the NEW Reasoning_VLM_v2_fixedsplit model.

Prepared Action Expert dataset
------------------------------
Create first with prepare_action_expert_fixedsplit.py:

    /home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/
        train.jsonl
        val.jsonl
        test.jsonl
        manifest.json

The prepare stage preserves the original Action Expert sample policy:
- every valid Part1 frame
- no task/sample cap
- 10 x (x,y,yaw), 0.5 s interval, 5 s, current-ego frame
- valid reasoning trace required

Only the split policy changes:
- exact clip membership from ~/splits_vlm.json
- driving-log-disjoint train/val/test
- no old ActionExpert8 80/10/10 split
- no new random split inside this trainer

10 models
---------
Transformer 2 x 2 x 2 ablation (8):
  direct_encoder_bidirectional
  direct_encoder_causal
  direct_decoder_bidirectional
  direct_decoder_causal
  reasoning_encoder_bidirectional
  reasoning_encoder_causal
  reasoning_decoder_bidirectional
  reasoning_decoder_causal

Flow/DiT v2 (2):
  direct_flow_dit
  reasoning_flow_dit

Important
---------
- Frozen VLM for every branch:
    /home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit
- VLM prompt:
    3 cameras + mission command + speed + acceleration + heading/yaw
    + reasoning-only instruction.
- direct:
    prompt-boundary last-layer KV.
- reasoning:
    same prompt-boundary KV + target-model greedy AR reasoning KV delta.
- DFlash is NOT used to create Action Expert training features.
- One VLM KV cache is generated once; then the VLM is unloaded.
- The 10 Action Experts are trained sequentially, one model at a time.
- Training order: Flow/DiT v2 first, then Transformer ablations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_PROJECT_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_PROJECT_ROOT / "scripts"
VLM_SCRIPT_DIR = Path("/home/lhh/lab/VLM/scripts")
for _p in (SCRIPT_DIR, ACTION_PROJECT_ROOT, ACTION_SCRIPT_DIR, VLM_SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Newly merged fixed-split VLM.
VLM_PATH = Path(
    "/home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit"
)

# Central dataset root + Action Expert fixed-split dataset.
DATASET_ROOT = Path("/home/lhh/lab/Dataset")
ACTION_DATASET_ROOT = (
    DATASET_ROOT / "ActionExpert" / "part1_fixedsplit"
)
TRAIN_JSONL = ACTION_DATASET_ROOT / "train.jsonl"
VAL_JSONL = ACTION_DATASET_ROOT / "val.jsonl"
TEST_JSONL = ACTION_DATASET_ROOT / "test.jsonl"
ACTION_DATASET_MANIFEST_PATH = ACTION_DATASET_ROOT / "manifest.json"
CENTRAL_DATASET_MANIFEST_PATH = DATASET_ROOT / "manifest.json"

# Authoritative senior/base split used by the fixed-split VLM.
SPLIT_JSON_PATH = Path("/home/lhh/splits_vlm.json")

# New cache/model roots: never mix with old ActionExpert8 / old Reasoning_VLM_v2.
CACHE_ROOT = (
    DATASET_ROOT
    / "ActionExpert"
    / "reasoning_vlm_v2_fixedsplit_10model"
    / "action_kv_cache"
)
MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_10model"
)


# =============================================================================
# ACTION MODEL IMPORTS
# =============================================================================

from scripts.stored.action_model_ablation_v2 import (
    ARCHITECTURES,
    build_action_expert as build_direct,
    count_trainable_parameters as count_direct_params,
    trajectory_loss as direct_loss,
    trajectory_metrics_np as direct_metrics_np,
)

from scripts.action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    count_trainable_parameters as count_flow_params,
    euler_sample,
    flow_matching_batch,
    linear_flow_oracle_sanity_check,
    trajectory_metrics_np as flow_metrics_np,
)

import scripts.stored.reasoning_v2_core as vlm_v2_core


# =============================================================================
# CONFIG
# =============================================================================

BRANCHES = ("direct", "reasoning")
FAMILIES = ("transformer", "flow")

# Final Transformer ablation:
#   Input      : direct / reasoning
#   Structure  : encoder / decoder
#   Direction  : bidirectional / causal
#
# encoder = encoder-style fusion:
#           projected VLM memory + trajectory queries in one self-attention stack
# decoder = decoder-style fusion:
#           trajectory-query stream + cross-attention to VLM memory
#
# direction is independent of structure and controls trajectory-query attention.
ABLATION_AXES = {
    "input": ("direct", "reasoning"),
    "structure": ("encoder", "decoder"),
    "attention_direction": ("bidirectional", "causal"),
}

SEED = 20260823
VAL_NOISE_SEED = 20260824
VAL_FLOW_SEED = 20260840

HIDDEN_DIM = 1536
NUM_LAYERS = 13
NUM_HEADS = 12
FF_DIM = 6144
DROPOUT = 0.1
NUM_STEPS = 10

EPOCHS = 50
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-2
WARMUP_RATIO = 0.05
PATIENCE = 10

MAX_NEW_TOKENS = 128
VLM_DTYPE = "bf16"

SOLVER_STEPS = 10
TIMESTEP_SAMPLER = "uniform"

MIN_DELTA_ADE = 1.0e-4
HEADING_LOSS_WEIGHT = 0.5
GRAD_CLIP_NORM = 1.0
NORMALIZER_STD_FLOOR = 1.0e-3

CACHE_VERSION = "reasoning_vlm_v2_fixedsplit_target_ar_kv_ablation_v3"

PROMPT_MODE = "reasoning_v2_fixedsplit_ego_state_reasoning_only"
ATTN_IMPLEMENTATION = "sdpa"
MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
CACHE_STORAGE_DTYPE = torch.bfloat16


# =============================================================================
# GENERIC HELPERS
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


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSON parse failed {path}:{n}") from exc
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


def model_dir_signature(path: Path) -> str:
    """
    Lightweight signature for cache invalidation when a model directory is
    overwritten in place. Hash filenames + sizes + mtimes of model/config files.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)

    patterns = (
        "*.safetensors",
        "*.bin",
        "*.json",
        "*.model",
        "*.txt",
    )
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))

    files = sorted({f.resolve() for f in files if f.is_file()})
    if not files:
        raise RuntimeError(f"No model/config files found under {path}")

    h = hashlib.sha256()
    for file_path in files:
        stat = file_path.stat()
        rel = file_path.relative_to(path)
        h.update(str(rel).encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))

    return h.hexdigest()


def cache_filename(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}_{digest}.pt"


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(name)


# =============================================================================
# ACTION EXPERT FIXED-SPLIT ROWS
# =============================================================================

def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate one row produced by prepare_action_expert_fixedsplit.py.

    Expected GT:
        row["trajectory"] -> [10,3] = x, y, yaw
    """
    row = dict(row)

    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Empty sample id")
    row["id"] = sid

    clip = str(row.get("clip", "")).strip()
    if not clip:
        raise ValueError(f"Missing clip id={sid}")
    row["clip"] = clip

    command = str(
        row.get("command")
        or row.get("mission_command")
        or ""
    ).strip()
    if not command:
        raise ValueError(f"Missing mission command id={sid}")
    row["command"] = command
    row["mission_command"] = command

    images = row.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(
            f"Expected exactly 3 image paths id={sid}: {images}"
        )

    paths = [str(x) for x in images]
    for image_path in paths:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"id={sid}: {image_path}")
    row["images"] = paths

    for key in ("speed_mps", "acceleration_mps2", "heading_rad"):
        value = _finite_float(row.get(key))
        if value is None:
            raise RuntimeError(f"Missing/non-finite {key}: id={sid}")
        row[key] = value

    trajectory = np.asarray(
        row.get("trajectory"),
        dtype=np.float32,
    )
    if trajectory.shape != (NUM_STEPS, 3):
        raise ValueError(
            f"trajectory must be [10,3] id={sid}, got={trajectory.shape}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError(f"Non-finite trajectory id={sid}")
    row["trajectory"] = trajectory.tolist()

    return row


def load_split(
    path: Path,
    split_name: str,
) -> List[Dict[str, Any]]:
    rows = [normalize_row(row) for row in read_jsonl(path)]

    if not rows:
        raise RuntimeError(f"Empty Action Expert split: {path}")

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {path}")

    print(
        f"[DATA {split_name.upper()}] "
        f"rows={len(rows)}"
    )
    return rows


def validate_fixedsplit_dataset(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
) -> None:
    """
    Hard-verify that the prepared Action Expert dataset uses the same
    ~/splits_vlm.json as Reasoning_VLM_v2_fixedsplit.
    """
    if not ACTION_DATASET_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Action Expert fixedsplit manifest missing: "
            f"{ACTION_DATASET_MANIFEST_PATH}\n"
            "Run prepare_action_expert_fixedsplit.py first."
        )
    if not SPLIT_JSON_PATH.is_file():
        raise FileNotFoundError(SPLIT_JSON_PATH)

    manifest = json.loads(
        ACTION_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    split_obj = json.loads(
        SPLIT_JSON_PATH.read_text(encoding="utf-8")
    )

    policy = manifest.get("split_policy") or {}
    if policy.get("type") != "fixed_external_driving_log_split":
        raise RuntimeError(
            "Action Expert dataset is not marked as fixed driving-log split: "
            f"{ACTION_DATASET_MANIFEST_PATH}"
        )

    current_split_sha = sha256_file(SPLIT_JSON_PATH)
    prepared_split_sha = str(policy.get("source_sha256", "")).strip()

    if prepared_split_sha != current_split_sha:
        raise RuntimeError(
            "Action Expert dataset was prepared with a different "
            "splits_vlm.json.\n"
            f"prepared={prepared_split_sha}\n"
            f"current ={current_split_sha}"
        )

    sampling_policy = manifest.get("sampling_policy") or {}
    if sampling_policy.get("task_cap") is not None:
        raise RuntimeError(
            "Prepared Action Expert dataset unexpectedly has a task cap."
        )
    if not bool(sampling_policy.get("use_all_valid_frames", False)):
        print(
            "[WARN] Action Expert dataset was prepared with max_per_clip > 0. "
            "Use the default prepare command for the real experiment."
        )

    splits = split_obj.get("splits")
    clip_to_log = split_obj.get("clip_to_log")
    if not isinstance(splits, dict) or not isinstance(clip_to_log, dict):
        raise RuntimeError("Invalid splits_vlm.json schema")

    split_sets = {
        name: {
            str(x).strip()
            for x in (splits.get(name) or [])
            if str(x).strip()
        }
        for name in ("train", "val", "test")
    }

    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError("train/val clip overlap in splits_vlm.json")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("train/test clip overlap in splits_vlm.json")
    if split_sets["val"] & split_sets["test"]:
        raise RuntimeError("val/test clip overlap in splits_vlm.json")

    missing_log = [
        clip
        for clip in set().union(*split_sets.values())
        if clip not in clip_to_log or not str(clip_to_log[clip]).strip()
    ]
    if missing_log:
        raise RuntimeError(
            f"Missing clip_to_log mappings: {sorted(missing_log)[:5]}"
        )

    split_logs = {
        name: {
            str(clip_to_log[clip]).strip()
            for clip in split_sets[name]
        }
        for name in ("train", "val", "test")
    }

    if split_logs["train"] & split_logs["val"]:
        raise RuntimeError("train/val driving-log overlap")
    if split_logs["train"] & split_logs["test"]:
        raise RuntimeError("train/test driving-log overlap")
    if split_logs["val"] & split_logs["test"]:
        raise RuntimeError("val/test driving-log overlap")

    actual_train_clips = {
        str(row["clip"]).strip() for row in train_rows
    }
    actual_val_clips = {
        str(row["clip"]).strip() for row in val_rows
    }

    invalid_train = actual_train_clips - split_sets["train"]
    invalid_val = actual_val_clips - split_sets["val"]

    if invalid_train:
        raise RuntimeError(
            "Train rows contain wrong-split clips: "
            f"{sorted(invalid_train)[:5]}"
        )
    if invalid_val:
        raise RuntimeError(
            "Val rows contain wrong-split clips: "
            f"{sorted(invalid_val)[:5]}"
        )
    if actual_train_clips & actual_val_clips:
        raise RuntimeError("Train/Val clip leakage")

    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    if train_ids & val_ids:
        raise RuntimeError("Train/Val ID leakage")

    # Optional second guard: central fixed-split VLM manifest should reference
    # the same splits_vlm.json, but the Action Expert rows do not depend on its
    # capped trajectory rows.
    if CENTRAL_DATASET_MANIFEST_PATH.is_file():
        central = json.loads(
            CENTRAL_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        central_policy = central.get("split_policy") or {}
        central_sha = str(
            central_policy.get("source_sha256", "")
        ).strip()
        if central_sha and central_sha != current_split_sha:
            raise RuntimeError(
                "Central VLM Dataset manifest and Action Expert split differ."
            )

    print(
        "[FIXEDSPLIT] PASS | "
        f"split_sha256={current_split_sha[:12]}... | "
        f"train_rows={len(train_rows)} | "
        f"val_rows={len(val_rows)} | "
        f"train_clips={len(actual_train_clips)} | "
        f"val_clips={len(actual_val_clips)}"
    )


# =============================================================================
# Reasoning_VLM_v2_fixedsplit TARGET-ONLY PREFILL / AR GENERATION
# =============================================================================

CAMERAS = ("front_left", "front", "front_right")


def load_target_model(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    # Load ONLY the frozen Reasoning_VLM_v2_fixedsplit target model.
    # No DFlash model/checkpoint/helper is used.
    if not path.is_dir():
        raise FileNotFoundError(path)

    kwargs = {
        "attn_implementation": ATTN_IMPLEMENTATION,
        "low_cpu_mem_usage": True,
    }

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(path),
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(path),
            torch_dtype=dtype,
            **kwargs,
        )

    model = model.to(device)
    model.eval()

    if hasattr(model, "config"):
        model.config.use_cache = True

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        str(path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    return model, processor


def row_image_path(row: Dict[str, Any], cam: str) -> str:
    images = row.get("images")

    if isinstance(images, dict):
        value = images.get(cam)
    elif isinstance(images, list) and len(images) == 3:
        value = images[CAMERAS.index(cam)]
    else:
        value = None

    if not value:
        raise KeyError(f"Missing image {cam}: id={row.get('id')}")

    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(path)

    return str(path)


def reset_multimodal_rope_state(model) -> None:
    """Avoid stale Qwen3-VL multimodal rope_deltas across batch-1 samples."""
    candidates = [model, getattr(model, "model", None)]
    inner = getattr(getattr(model, "model", None), "language_model", None)
    if inner is not None:
        candidates.append(inner)
    for obj in candidates:
        if obj is not None and hasattr(obj, "rope_deltas"):
            obj.rope_deltas = None


def make_reasoning_v2_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt one central fixed-split trajectory row to the VLM reasoning prompt."""
    return {
        "id": str(row["id"]),
        "task": "reasoning",
        "images": list(row["images"]),
        "command": str(row["command"]),
        "speed_mps": float(row["speed_mps"]),
        "acceleration_mps2": float(row["acceleration_mps2"]),
        "heading_rad": float(row["heading_rad"]),
    }


def build_prompt_v2(
    processor,
    row: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Build the SAME user prompt/image layout used to train Reasoning_VLM_v2_fixedsplit.

    Camera labels and image placeholders are intentionally identical to
    reasoning_v2_core.QwenVLCollator.
    """
    rec = make_reasoning_v2_record(row)
    prompt = vlm_v2_core.build_prompt(rec)
    images = vlm_v2_core.open_three_images(rec)

    user_content = [
        {"type": "text", "text": "Front-left camera:"},
        {"type": "image"},
        {"type": "text", "text": "Front camera:"},
        {"type": "image"},
        {"type": "text", "text": "Front-right camera:"},
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": user_content}]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    batch = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
        padding=False,
    )

    batch = dict(batch)
    batch.pop("token_type_ids", None)

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value

    return moved


def get_last_layer_kv(past_key_values):
    if past_key_values is None:
        raise RuntimeError("past_key_values is None")

    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        if layers:
            layer = layers[-1]
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None and value is not None:
                return key, value

    if (
        hasattr(past_key_values, "key_cache")
        and hasattr(past_key_values, "value_cache")
    ):
        if past_key_values.key_cache and past_key_values.value_cache:
            return (
                past_key_values.key_cache[-1],
                past_key_values.value_cache[-1],
            )

    if isinstance(past_key_values, (tuple, list)):
        last = past_key_values[-1]
        if isinstance(last, (tuple, list)) and len(last) >= 2:
            return last[0], last[1]

    raise RuntimeError(
        f"Unsupported past_key_values type: {type(past_key_values)}"
    )


def kv_to_sequence(
    key: torch.Tensor,
    value: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    # [B,H,T,D] K/V -> [B,T,2*H*D]
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected K/V [B,H,T,D], "
            f"got K={tuple(key.shape)}, V={tuple(value.shape)}"
        )

    if max_length is not None:
        key = key[..., :max_length, :]
        value = value[..., :max_length, :]

    key_seq = key.permute(0, 2, 1, 3).contiguous().flatten(2)
    value_seq = value.permute(0, 2, 1, 3).contiguous().flatten(2)

    return torch.cat([key_seq, value_seq], dim=-1)


def normalize_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(x) for x in value if x is not None}
    return {int(value)}


def reasoning_content_ids(
    processor,
    model,
    new_ids: torch.Tensor,
) -> Tuple[List[int], bool]:
    eos_ids: set[int] = set()
    eos_ids |= normalize_token_id_set(
        getattr(processor.tokenizer, "eos_token_id", None)
    )
    eos_ids |= normalize_token_id_set(
        getattr(
            getattr(model, "generation_config", None),
            "eos_token_id",
            None,
        )
    )

    for token_text in ("<|im_end|>", "<|endoftext|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token_text)
        if (
            token_id is not None
            and token_id != getattr(processor.tokenizer, "unk_token_id", None)
            and int(token_id) >= 0
        ):
            eos_ids.add(int(token_id))

    pad_ids: set[int] = set()
    pad_ids |= normalize_token_id_set(
        getattr(processor.tokenizer, "pad_token_id", None)
    )
    pad_ids |= normalize_token_id_set(
        getattr(
            getattr(model, "generation_config", None),
            "pad_token_id",
            None,
        )
    )

    content: List[int] = []
    ended = False

    for token in new_ids.tolist():
        token = int(token)

        if token in eos_ids:
            ended = True
            break

        if token in pad_ids and content:
            ended = True
            break

        content.append(token)

    return content, ended


@torch.inference_mode()
def extract_one(
    model,
    processor,
    row,
    device,
    dtype,
    max_new_tokens,
    allow_truncated,
):
    # Target-only cache extraction.
    reset_multimodal_rope_state(model)

    batch = build_prompt_v2(
        processor=processor,
        row=row,
        device=device,
        dtype=dtype,
    )

    prompt_token_length = int(batch["input_ids"].shape[1])

    # 1) Prompt-boundary KV, immediately before reasoning generation.
    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        prompt_out = model(
            **batch,
            use_cache=True,
            return_dict=True,
        )

    direct_key, direct_value = get_last_layer_kv(
        prompt_out.past_key_values
    )
    prompt_cache_length = int(direct_key.shape[-2])

    direct_kv = (
        kv_to_sequence(
            direct_key,
            direct_value,
            max_length=prompt_cache_length,
        )[0]
        .detach()
        .to(dtype=CACHE_STORAGE_DTYPE)
        .cpu()
        .contiguous()
    )

    del prompt_out, direct_key, direct_value

    # 2) Plain target-model greedy autoregressive reasoning.
    #    No DFlash / speculative decoding.
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        generation = model.generate(
            **batch,
            **generation_kwargs,
        )

    generation_cache = getattr(
        generation,
        "past_key_values",
        None,
    )
    if generation_cache is None:
        raise RuntimeError(
            "Target model generate() did not return past_key_values."
        )

    sequences = getattr(generation, "sequences", None)
    if sequences is None:
        raise RuntimeError("Target model generate() returned no sequences")

    new_ids = sequences[0, prompt_token_length:]
    content_ids, ended_eos = reasoning_content_ids(
        processor,
        model,
        new_ids,
    )

    if not content_ids:
        raise RuntimeError("Generated reasoning is empty")

    if not ended_eos and not allow_truncated:
        raise RuntimeError(
            f"Reasoning did not terminate within "
            f"max_new_tokens={max_new_tokens}. "
            "Increase --max-new-tokens or use --allow-truncated."
        )

    reasoning_text = processor.tokenizer.decode(
        content_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if not reasoning_text:
        raise RuntimeError("Decoded reasoning text is empty")

    gen_key, gen_value = get_last_layer_kv(generation_cache)
    generation_cache_length = int(gen_key.shape[-2])

    desired_reasoning_end = prompt_cache_length + len(content_ids)
    usable_reasoning_end = min(
        generation_cache_length,
        desired_reasoning_end,
    )
    missing = max(
        0,
        desired_reasoning_end - generation_cache_length,
    )

    if ended_eos and missing > 0:
        raise RuntimeError(
            "Target AR cache does not contain all reasoning tokens: "
            f"missing={missing}, "
            f"prompt_cache={prompt_cache_length}, "
            f"reasoning_tokens={len(content_ids)}, "
            f"generation_cache={generation_cache_length}"
        )

    if missing > 1:
        raise RuntimeError(
            f"Target AR cache is too short by {missing} tokens"
        )

    full_kv = (
        kv_to_sequence(
            gen_key,
            gen_value,
            max_length=usable_reasoning_end,
        )[0]
        .detach()
    )

    prefix_len = min(
        prompt_cache_length,
        int(full_kv.shape[0]),
    )
    if prefix_len <= 0:
        raise RuntimeError("Invalid zero-length prompt KV")

    prefix_diff = float(
        (
            full_kv[:prefix_len].float().cpu()
            - direct_kv[:prefix_len].float()
        )
        .abs()
        .max()
        .item()
    )

    reasoning_delta = full_kv[
        prompt_cache_length:usable_reasoning_end
    ]

    if reasoning_delta.shape[0] <= 0:
        raise RuntimeError("No reasoning KV tokens were cached")

    reasoning_delta_kv = (
        reasoning_delta
        .to(dtype=CACHE_STORAGE_DTYPE)
        .cpu()
        .contiguous()
    )

    trajectory = torch.as_tensor(
        row["trajectory"],
        dtype=torch.float32,
    ).cpu()

    if tuple(trajectory.shape) != (NUM_STEPS, 3):
        raise ValueError(
            f"Bad trajectory shape: {tuple(trajectory.shape)}"
        )

    return {
        "cache_version": CACHE_VERSION,
        "id": str(row["id"]),
        "clip": str(row.get("clip", "")),
        "direct_kv": direct_kv,
        "reasoning_delta_kv": reasoning_delta_kv,
        "trajectory": trajectory.contiguous(),
        "prompt_token_length": int(prompt_token_length),
        "prompt_cache_length": int(prompt_cache_length),
        "reasoning_generated_tokens": int(len(content_ids)),
        "reasoning_cached_tokens": int(reasoning_delta_kv.shape[0]),
        "generation_cache_length": int(generation_cache_length),
        "generation_ended": bool(ended_eos),
        "missing_reasoning_cache_tokens": int(missing),
        "prefix_max_abs_diff": float(prefix_diff),
        "reasoning_token_ids": [int(x) for x in content_ids],
        "reasoning_text": reasoning_text,
        "vlm": str(VLM_PATH),
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
    }


# =============================================================================
# CACHE BUILD / REUSE
# =============================================================================

def cache_request(args) -> Dict[str, Any]:
    config_path = args.vlm / "config.json"
    return {
        "version": CACHE_VERSION,
        "vlm": str(args.vlm.resolve()),
        "vlm_config_sha256": (
            sha256_file(config_path) if config_path.is_file() else None
        ),
        "vlm_dir_signature": model_dir_signature(args.vlm),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "train_sha256": sha256_file(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl.resolve()),
        "val_sha256": sha256_file(args.val_jsonl),
        "action_dataset_manifest": str(
            ACTION_DATASET_MANIFEST_PATH.resolve()
        ),
        "action_dataset_manifest_sha256": sha256_file(
            ACTION_DATASET_MANIFEST_PATH
        ),
        "central_dataset_manifest_sha256": (
            sha256_file(CENTRAL_DATASET_MANIFEST_PATH)
            if CENTRAL_DATASET_MANIFEST_PATH.is_file()
            else None
        ),
        "split_json": str(SPLIT_JSON_PATH.resolve()),
        "split_json_sha256": sha256_file(SPLIT_JSON_PATH),
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": args.max_new_tokens,
        "allow_truncated": args.allow_truncated,
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }


def valid_manifest(path: Path, expected_ids: Sequence[str]):
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if [str(x["id"]) for x in rows] != list(expected_ids):
        return None
    if not all(Path(x["cache_file"]).is_file() for x in rows):
        return None
    return rows


def reuse_cache_if_valid(args, train_rows, val_rows):
    meta_path = args.cache_root / "meta.json"
    if not meta_path.is_file():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if meta.get("request") != cache_request(args):
        return None

    train_manifest = valid_manifest(
        args.cache_root / "train" / "manifest.jsonl",
        [x["id"] for x in train_rows],
    )
    val_manifest = valid_manifest(
        args.cache_root / "val" / "manifest.jsonl",
        [x["id"] for x in val_rows],
    )

    if train_manifest is None or val_manifest is None:
        return None

    print(
        f"[CACHE] reuse | train={len(train_manifest)} "
        f"val={len(val_manifest)}"
    )
    return train_manifest, val_manifest


def build_cache(args, train_rows, val_rows, device, vlm_dtype):
    if args.cache_root.exists() and not args.rebuild_cache:
        reused = reuse_cache_if_valid(args, train_rows, val_rows)
        if reused is not None:
            return reused
        raise RuntimeError(
            f"Incompatible existing cache: {args.cache_root}\n"
            "This cache must be rebuilt with target-only AR extraction.\n"
            "Use --rebuild-cache."
        )

    if args.cache_root.exists():
        shutil.rmtree(args.cache_root)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("STAGE 1 | Reasoning_VLM_v2_fixedsplit TARGET-ONLY KV CACHE")
    print("=" * 120)
    print("VLM         :", args.vlm)
    print("Prompt      :", PROMPT_MODE)
    print("Decode      : target autoregressive greedy")
    print("DFlash      : NOT USED")
    print("Attention   :", ATTN_IMPLEMENTATION)
    print("Train / Val :", len(train_rows), "/", len(val_rows))
    print("Cache root  :", args.cache_root)

    model, processor = load_target_model(
        args.vlm,
        device,
        vlm_dtype,
    )

    manifests = {}

    for split, rows in (("train", train_rows), ("val", val_rows)):
        out_dir = args.cache_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []

        split_start = time.perf_counter()

        for index, row in enumerate(rows, 1):
            t0 = time.perf_counter()
            sid = row["id"]
            out_path = out_dir / cache_filename(index, sid)

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
                "kv_dim": int(record["direct_kv"].shape[-1]),
                "prompt_cache_length": int(record["prompt_cache_length"]),
                "reasoning_cached_tokens": int(
                    record["reasoning_cached_tokens"]
                ),
                "reasoning_text": record["reasoning_text"],
                "prefix_max_abs_diff": float(
                    record["prefix_max_abs_diff"]
                ),
            })

            sample_sec = time.perf_counter() - t0
            split_elapsed = time.perf_counter() - split_start
            avg_sec = split_elapsed / index
            split_eta = avg_sec * (len(rows) - index)

            print(
                f"[CACHE {split.upper():5s}] "
                f"{index:04d}/{len(rows):04d} | "
                f"id={sid} | "
                f"directT={record['direct_kv'].shape[0]} "
                f"reasonT={record['reasoning_delta_kv'].shape[0]} "
                f"D={record['direct_kv'].shape[-1]} | "
                f"sample={sample_sec:.2f}s | "
                f"avg={avg_sec:.2f}s/sample | "
                f"elapsed={format_eta(split_elapsed)} | "
                f"eta={format_eta(split_eta)}",
                flush=True,
            )

            del record

        write_jsonl(out_dir / "manifest.jsonl", manifest)
        manifests[split] = manifest

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = int(getattr(text_cfg, "num_hidden_layers"))

    meta = {
        "request": cache_request(args),
        "runtime": "Reasoning_VLM_v2_fixedsplit target-only AR",
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "prompt_mode": PROMPT_MODE,
        "attention_implementation": ATTN_IMPLEMENTATION,
        "last_hidden_layer": n_layers - 1,
        "train_samples": len(manifests["train"]),
        "val_samples": len(manifests["val"]),
    }

    (args.cache_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, processor
    cleanup_cuda()

    return manifests["train"], manifests["val"]


# =============================================================================
# DATASET
# =============================================================================

class BranchDataset(Dataset):
    def __init__(self, manifest, branch):
        if branch not in BRANCHES:
            raise ValueError(branch)
        self.manifest = list(manifest)
        self.branch = branch

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        item = self.manifest[index]
        cache = torch.load(
            item["cache_file"],
            map_location="cpu",
            weights_only=False,
        )

        direct = cache["direct_kv"]
        delta = cache["reasoning_delta_kv"]
        gt = cache["trajectory"].float()

        if direct.ndim != 2 or delta.ndim != 2:
            raise RuntimeError(f"Bad KV rank id={item['id']}")
        if direct.shape[-1] != delta.shape[-1]:
            raise RuntimeError(f"KV dim mismatch id={item['id']}")

        if self.branch == "direct":
            memory = direct
            segment_ids = torch.zeros(direct.shape[0], dtype=torch.long)
        else:
            memory = torch.cat([direct, delta], dim=0)
            segment_ids = torch.cat([
                torch.zeros(direct.shape[0], dtype=torch.long),
                torch.ones(delta.shape[0], dtype=torch.long),
            ])

        return {
            "id": str(item["id"]),
            "memory": memory,
            "segment_ids": segment_ids,
            "trajectory": gt,
        }


def collate_kv(items):
    bs = len(items)
    max_len = max(int(x["memory"].shape[0]) for x in items)
    dim = int(items[0]["memory"].shape[1])
    dtype = items[0]["memory"].dtype

    memory = torch.zeros((bs, max_len, dim), dtype=dtype)
    mask = torch.zeros((bs, max_len), dtype=torch.bool)
    seg = torch.zeros((bs, max_len), dtype=torch.long)

    trajectories = []
    ids = []

    for i, item in enumerate(items):
        x = item["memory"]
        if int(x.shape[1]) != dim:
            raise RuntimeError("KV dim mismatch inside batch")
        n = int(x.shape[0])
        memory[i, :n] = x
        mask[i, :n] = True
        seg[i, :n] = item["segment_ids"]
        trajectories.append(item["trajectory"])
        ids.append(item["id"])

    return {
        "id": ids,
        "memory": memory,
        "memory_mask": mask,
        "segment_ids": seg,
        "trajectory": torch.stack(trajectories),
    }


def build_loader(manifest, branch, batch_size, shuffle, seed):
    ds = BranchDataset(manifest, branch)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
        generator=generator,
    )
    return ds, dl


def move_batch(batch, device):
    inputs = {
        "memory": batch["memory"].to(
            device, dtype=torch.float32, non_blocking=True
        ),
        "memory_mask": batch["memory_mask"].to(
            device, non_blocking=True
        ),
        "segment_ids": batch["segment_ids"].to(
            device, non_blocking=True
        ),
    }
    gt = batch["trajectory"].to(
        device, dtype=torch.float32, non_blocking=True
    )
    return inputs, gt


def infer_input_dim(manifest):
    cache = torch.load(
        manifest[0]["cache_file"],
        map_location="cpu",
        weights_only=False,
    )
    direct = cache["direct_kv"]
    delta = cache["reasoning_delta_kv"]
    if direct.shape[-1] != delta.shape[-1]:
        raise RuntimeError("direct/delta KV dim mismatch")
    return int(direct.shape[-1])


# =============================================================================
# DIRECT TRANSFORMER
# =============================================================================

@torch.inference_mode()
def eval_direct(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    preds, gts = [], []

    for batch in loader:
        inputs, gt = move_batch(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            pred = model(**inputs)

        loss, _, _ = direct_loss(
            pred, gt, heading_weight=HEADING_LOSS_WEIGHT
        )

        bs = int(gt.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
        preds.append(pred.float().cpu().numpy())
        gts.append(gt.float().cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = direct_metrics_np(pred_np, gt_np)
    metrics["loss"] = loss_sum / n
    return metrics


def train_direct(
    architecture,
    branch,
    train_manifest,
    val_manifest,
    input_dim,
    cache_signature,
    device,
    args,
):
    set_seed(args.seed)

    train_ds, train_dl = build_loader(
        train_manifest, branch, args.batch_size, True, args.seed
    )
    val_ds, val_dl = build_loader(
        val_manifest, branch, args.batch_size, False, args.seed
    )

    model = build_direct(
        architecture=architecture,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)

    params = count_direct_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    name = f"{branch}_{architecture}"
    out_dir = args.model_root / "transformer" / name

    if out_dir.exists() and args.overwrite_models:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.jsonl"

    if best_path.exists() and not args.overwrite_models:
        raise RuntimeError(
            f"{best_path} already exists. Use --overwrite-models."
        )
    if history_path.exists():
        history_path.unlink()

    config = {
        "experiment": "reasoning_vlm_v2_fixedsplit_10model",
        "family": "transformer",
        "condition": name,
        "branch": branch,
        "architecture": architecture,
        "structure": architecture.split("_", 1)[0],
        "attention_direction": architecture.split("_", 1)[1],
        "vlm": str(args.vlm),
        "prompt_mode": PROMPT_MODE,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_steps": NUM_STEPS,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "trainable_params": params,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "seed": args.seed,
        "precision": "fp32_master_bf16_autocast",
        "selection_metric": "val_ade_m",
        "cache_signature": cache_signature,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print(f"TRAIN TRANSFORMER | {name}")
    print("=" * 120)
    print("Train / Val :", len(train_ds), "/", len(val_ds))
    print("Architecture:", architecture)
    print("Branch      :", branch)
    print("Params      :", f"{params:,} ({params/1e9:.6f}B)")
    print("Checkpoint  :", best_path)

    best_ade = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0
    model_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_n = 0
        t0 = time.perf_counter()

        for micro_idx, batch in enumerate(train_dl, 1):
            inputs, gt = move_batch(batch, device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                pred = model(**inputs)

            loss, _, _ = direct_loss(
                pred, gt, heading_weight=HEADING_LOSS_WEIGHT
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss {name} e={epoch} micro={micro_idx}"
                )

            (loss / args.grad_accum).backward()

            bs = int(gt.shape[0])
            train_loss_sum += float(loss.detach().item()) * bs
            train_n += bs

            if (
                micro_idx % args.grad_accum == 0
                or micro_idx == len(train_dl)
            ):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(f"Non-finite grad norm {name}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss_value = train_loss_sum / train_n
        val = eval_direct(model, val_dl, device)

        epoch_sec = time.perf_counter() - t0
        elapsed_min = epoch_sec / 60.0
        model_elapsed = time.perf_counter() - model_start
        avg_epoch_sec = model_elapsed / epoch
        model_eta = avg_epoch_sec * (args.epochs - epoch)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        row = {
            "condition": name,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_value,
            "val_loss": float(val["loss"]),
            "val_ade_m": float(val["ade_m"]),
            "val_fde_m": float(val["fde_m"]),
            "val_heading_mae_rad": float(val["heading_mae_rad"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "peak_vram_gb": peak_gb,
            "epoch_minutes": elapsed_min,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[TRAIN {name}] "
            f"epoch={epoch:03d}/{args.epochs:03d} | "
            f"train={train_loss_value:.6f} "
            f"val={val['loss']:.6f} | "
            f"ADE={val['ade_m']:.4f}m "
            f"FDE={val['fde_m']:.4f}m "
            f"Heading={val['heading_mae_rad']:.4f}rad | "
            f"VRAM={peak_gb:.2f}GB | "
            f"epoch_time={format_eta(epoch_sec)} | "
            f"elapsed={format_eta(model_elapsed)} | "
            f"model_eta={format_eta(model_eta)}"
        )

        if val["ade_m"] < best_ade - MIN_DELTA_ADE:
            best_ade = float(val["ade_m"])
            best_epoch = epoch
            no_improve = 0

            torch.save({
                "experiment": "reasoning_vlm_v2_fixedsplit_10model",
                "family": "transformer",
                "condition": name,
                "branch": branch,
                "architecture": architecture,
                "structure": architecture.split("_", 1)[0],
                "attention_direction": architecture.split("_", 1)[1],
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "action_config": {
                    "hidden_dim": args.hidden_dim,
                    "num_steps": NUM_STEPS,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "ff_dim": args.ff_dim,
                    "dropout": args.dropout,
                    "gradient_checkpointing": (
                        not args.no_gradient_checkpointing
                    ),
                },
                "trainable_params": params,
                "cache_signature": cache_signature,
                "vlm": str(args.vlm),
                "prompt_mode": PROMPT_MODE,
                "val_metrics": val,
            }, best_path)

            print(
                f"[{name}] BEST -> epoch={epoch} "
                f"ADE={best_ade:.6f} saved={best_path}"
            )
        else:
            no_improve += 1
            print(
                f"[{name}] no ADE improvement "
                f"{no_improve}/{args.patience} "
                f"(best={best_ade:.6f})"
            )

        if no_improve >= args.patience:
            print(
                f"[{name}] EARLY STOP | "
                f"best_epoch={best_epoch} best_ADE={best_ade:.6f}"
            )
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)

    result = {
        "family": "transformer",
        "condition": name,
        "branch": branch,
        "architecture": architecture,
        "best_epoch": int(best["epoch"]),
        "trainable_params": int(params),
        **{
            f"val_{k}": float(v)
            for k, v in best["val_metrics"].items()
        },
        "checkpoint": str(best_path),
    }

    del (
        best, model, optimizer, scheduler,
        train_dl, val_dl, train_ds, val_ds
    )
    cleanup_cuda()
    return result


# =============================================================================
# FLOW / DiT v2
# =============================================================================

def compute_normalizer(train_manifest):
    sums = torch.zeros(NUM_STEPS, 3, dtype=torch.float64)
    sq_sums = torch.zeros_like(sums)
    n = 0

    for item in train_manifest:
        cache = torch.load(
            item["cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        traj = torch.as_tensor(
            cache["trajectory"], dtype=torch.float64
        )
        sums += traj
        sq_sums += traj * traj
        n += 1

    mean = sums / n
    var = sq_sums / n - mean * mean
    std = torch.sqrt(var.clamp_min(1e-12)).clamp_min(
        NORMALIZER_STD_FLOOR
    )
    return TrajectoryNormalizer(
        mean=mean.float(),
        std=std.float(),
    )


@torch.inference_mode()
def eval_flow(
    model,
    loader,
    device,
    normalizer,
    solver_steps,
    timestep_sampler,
):
    model.eval()
    norm = normalizer.to(device)

    noise_rng = torch.Generator(device="cpu")
    noise_rng.manual_seed(VAL_NOISE_SEED)

    flow_rng = torch.Generator(device="cpu")
    flow_rng.manual_seed(VAL_FLOW_SEED)

    preds, gts = [], []
    loss_sum = 0.0
    n = 0

    for batch in loader:
        inputs, gt = move_batch(batch, device)
        gt_norm = norm.normalize(gt)

        x_t, t, v_target, _ = flow_matching_batch(
            gt_normalized=gt_norm,
            rng=flow_rng,
            timestep_sampler=timestep_sampler,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            v_pred = model(x_t=x_t, t=t, **inputs)

        fm_loss = F.mse_loss(
            v_pred.float(), v_target.float()
        )

        bs = int(gt.shape[0])
        loss_sum += float(fm_loss.item()) * bs
        n += bs

        pred = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=norm,
            solver_steps=solver_steps,
            rng=noise_rng,
        )

        preds.append(pred.float().cpu().numpy())
        gts.append(gt.float().cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = flow_metrics_np(pred_np, gt_np)
    metrics["flow_mse"] = loss_sum / n
    return metrics


def train_flow(
    branch,
    train_manifest,
    val_manifest,
    input_dim,
    cache_signature,
    normalizer,
    device,
    args,
):
    set_seed(args.seed)

    train_ds, train_dl = build_loader(
        train_manifest, branch, args.batch_size, True, args.seed
    )
    val_ds, val_dl = build_loader(
        val_manifest, branch, args.batch_size, False, args.seed
    )

    model = build_flow_dit(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_steps=NUM_STEPS,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)

    params = count_flow_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    out_dir = args.model_root / "flow" / f"{branch}_flow_dit"
    if out_dir.exists() and args.overwrite_models:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.jsonl"

    if best_path.exists() and not args.overwrite_models:
        raise RuntimeError(
            f"{best_path} already exists. Use --overwrite-models."
        )
    if history_path.exists():
        history_path.unlink()

    config = {
        "experiment": "reasoning_vlm_v2_fixedsplit_10model",
        "family": "flow_dit_v2",
        "branch": branch,
        "vlm": str(args.vlm),
        "prompt_mode": PROMPT_MODE,
        "conditioning": "KVMemoryProjector + decoder cross-attention",
        "objective": "linear_flow_matching_velocity_mse",
        "flow_path": "x_t=(1-t)*x0+t*x1; target=x1-x0",
        "action_input": "raw_normalized_xt_linear_plus_fourier_t",
        "normalization": "per_waypoint_xyz_train_only",
        "self_attention": "non_causal",
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_steps": NUM_STEPS,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "trainable_params": params,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "solver_steps": args.solver_steps,
        "timestep_sampler": args.timestep_sampler,
        "seed": args.seed,
        "precision": "fp32_master_bf16_autocast",
        "normalizer": normalizer.to_dict(),
        "cache_signature": cache_signature,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print(f"TRAIN FLOW / DiT v2 | {branch}")
    print("=" * 120)
    print("Train / Val :", len(train_ds), "/", len(val_ds))
    print("Branch      :", branch)
    print("Params      :", f"{params:,} ({params/1e9:.6f}B)")
    print("Solver      :", args.solver_steps)
    print("Checkpoint  :", best_path)

    norm = normalizer.to(device)

    best_ade = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0
    model_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        # Same x0/t RNG stream for both branches at same epoch.
        flow_rng = torch.Generator(device="cpu")
        flow_rng.manual_seed(args.seed + epoch * 1009)

        train_loss_sum = 0.0
        train_n = 0

        for micro_idx, batch in enumerate(train_dl, 1):
            inputs, gt = move_batch(batch, device)
            gt_norm = norm.normalize(gt)

            x_t, t, v_target, _ = flow_matching_batch(
                gt_normalized=gt_norm,
                rng=flow_rng,
                timestep_sampler=args.timestep_sampler,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                v_pred = model(x_t=x_t, t=t, **inputs)

            loss = F.mse_loss(
                v_pred.float(), v_target.float()
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite flow loss branch={branch} "
                    f"e={epoch} micro={micro_idx}"
                )

            (loss / args.grad_accum).backward()

            bs = int(gt.shape[0])
            train_loss_sum += float(loss.detach().item()) * bs
            train_n += bs

            if (
                micro_idx % args.grad_accum == 0
                or micro_idx == len(train_dl)
            ):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP_NORM
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(f"Non-finite flow grad norm {branch}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss_value = train_loss_sum / train_n

        val = eval_flow(
            model=model,
            loader=val_dl,
            device=device,
            normalizer=normalizer,
            solver_steps=args.solver_steps,
            timestep_sampler=args.timestep_sampler,
        )

        epoch_sec = time.perf_counter() - t0
        elapsed_min = epoch_sec / 60.0
        model_elapsed = time.perf_counter() - model_start
        avg_epoch_sec = model_elapsed / epoch
        model_eta = avg_epoch_sec * (args.epochs - epoch)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        row = {
            "branch": branch,
            "epoch": epoch,
            "global_step": global_step,
            "train_flow_mse": train_loss_value,
            "val_flow_mse": float(val["flow_mse"]),
            "val_ade_m": float(val["ade_m"]),
            "val_fde_m": float(val["fde_m"]),
            "val_heading_mae_rad": float(val["heading_mae_rad"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "peak_vram_gb": peak_gb,
            "epoch_minutes": elapsed_min,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[TRAIN flow/{branch}] "
            f"epoch={epoch:03d}/{args.epochs:03d} | "
            f"trainFM={train_loss_value:.6f} "
            f"valFM={val['flow_mse']:.6f} | "
            f"ADE={val['ade_m']:.4f}m "
            f"FDE={val['fde_m']:.4f}m "
            f"Heading={val['heading_mae_rad']:.4f}rad | "
            f"VRAM={peak_gb:.2f}GB | "
            f"epoch_time={format_eta(epoch_sec)} | "
            f"elapsed={format_eta(model_elapsed)} | "
            f"model_eta={format_eta(model_eta)}"
        )

        if val["ade_m"] < best_ade - MIN_DELTA_ADE:
            best_ade = float(val["ade_m"])
            best_epoch = epoch
            no_improve = 0

            torch.save({
                "experiment": "reasoning_vlm_v2_fixedsplit_10model",
                "family": "flow_dit_v2",
                "version": 2,
                "branch": branch,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "trainable_params": params,
                "cache_signature": cache_signature,
                "vlm": str(args.vlm),
                "prompt_mode": PROMPT_MODE,
                "normalizer": normalizer.to_dict(),
                "normalizer_mode": normalizer.mode,
                "action_config": {
                    "hidden_dim": args.hidden_dim,
                    "num_steps": NUM_STEPS,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "ff_dim": args.ff_dim,
                    "dropout": args.dropout,
                    "gradient_checkpointing": (
                        not args.no_gradient_checkpointing
                    ),
                    "solver_steps": args.solver_steps,
                    "timestep_sampler": args.timestep_sampler,
                    "conditioning": "KVMemoryProjector + decoder cross-attention",
                    "self_attention": "non_causal",
                    "flow_path": "x_t=(1-t)*x0+t*x1; target=x1-x0",
                    "action_input": "raw_normalized_xt_linear_plus_fourier_t",
                    "normalization": "per_waypoint_xyz_train_only",
                },
                "val_metrics": {
                    k: float(v) for k, v in val.items()
                },
            }, best_path)

            print(
                f"[flow/{branch}] BEST -> epoch={epoch} "
                f"ADE={best_ade:.6f} saved={best_path}"
            )
        else:
            no_improve += 1
            print(
                f"[flow/{branch}] no ADE improvement "
                f"{no_improve}/{args.patience} "
                f"(best={best_ade:.6f})"
            )

        if no_improve >= args.patience:
            print(
                f"[flow/{branch}] EARLY STOP | "
                f"best_epoch={best_epoch} best_ADE={best_ade:.6f}"
            )
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)

    result = {
        "family": "flow",
        "condition": f"{branch}_flow_dit",
        "branch": branch,
        "architecture": "flow_dit",
        "best_epoch": int(best["epoch"]),
        "trainable_params": int(params),
        **{
            f"val_{k}": float(v)
            for k, v in best["val_metrics"].items()
        },
        "checkpoint": str(best_path),
    }

    del (
        best, model, optimizer, scheduler,
        train_dl, val_dl, train_ds, val_ds
    )
    cleanup_cuda()
    return result


# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(args, results):
    args.model_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "reasoning_vlm_v2_fixedsplit_10model",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "vlm": str(args.vlm),
        "dataset_root": str(DATASET_ROOT),
        "split_json": str(SPLIT_JSON_PATH),
        "prompt_mode": PROMPT_MODE,
        "cache_root": str(args.cache_root),
        "results": results,
    }

    (args.model_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 130,
        "Reasoning_VLM_v2_fixedsplit | ACTION EXPERT 2x2x2 ABLATION SUMMARY",
        "=" * 130,
        f"VLM   : {args.vlm}",
        f"Cache : {args.cache_root}",
        "",
        f"{'FAMILY':<13}{'CONDITION':<38}{'PARAMS(B)':>12}"
        f"{'BEST_E':>9}{'ADE':>11}{'FDE':>11}{'HEAD':>11}",
        "-" * 130,
    ]

    for r in results:
        lines.append(
            f"{r['family']:<13}"
            f"{r['condition']:<38}"
            f"{r['trainable_params']/1e9:>12.3f}"
            f"{r['best_epoch']:>9d}"
            f"{r['val_ade_m']:>11.4f}"
            f"{r['val_fde_m']:>11.4f}"
            f"{r['val_heading_mae_rad']:>11.4f}"
        )

    (args.model_root / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vlm", type=Path, default=VLM_PATH)
    p.add_argument("--train-jsonl", type=Path, default=TRAIN_JSONL)
    p.add_argument("--val-jsonl", type=Path, default=VAL_JSONL)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--model-root", type=Path, default=MODEL_ROOT)

    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--overwrite-models", action="store_true")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the prepared Action Expert fixed-split rows and split "
            "integrity, then exit before loading the VLM."
        ),
    )

    p.add_argument(
        "--vlm-dtype",
        choices=("fp16", "bf16"),
        default=VLM_DTYPE,
    )
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--allow-truncated", action="store_true")

    p.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    p.add_argument(
        "--branches",
        nargs="+",
        choices=BRANCHES,
        default=list(BRANCHES),
    )
    p.add_argument(
        "--transformer-architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )

    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS)
    p.add_argument("--ff-dim", type=int, default=FF_DIM)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--no-gradient-checkpointing", action="store_true")

    p.add_argument("--solver-steps", type=int, default=SOLVER_STEPS)
    p.add_argument(
        "--timestep-sampler",
        choices=("uniform", "beta"),
        default=TIMESTEP_SAMPLER,
    )

    return p.parse_args()


def main():
    args = parse_args()

    for p in (
        args.vlm,
        args.train_jsonl,
        args.val_jsonl,
        ACTION_DATASET_MANIFEST_PATH,
        SPLIT_JSON_PATH,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch-size and grad-accum must be >= 1")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs/patience must be >= 1")
    if args.solver_steps < 1:
        raise ValueError("solver-steps must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be >= 1")

    # Dataset was already built from raw nuReasoning by the fixed-split
    # Action Expert prepare script. No new split is created here.
    train_rows = load_split(args.train_jsonl, "train")
    val_rows = load_split(args.val_jsonl, "val")
    validate_fixedsplit_dataset(train_rows, val_rows)

    if args.validate_only:
        print("=" * 120)
        print("FIXEDSPLIT ACTION EXPERT DATA VALIDATION: PASS")
        print("=" * 120)
        print("VLM           :", args.vlm)
        print("Dataset root  :", ACTION_DATASET_ROOT)
        print("Split JSON    :", SPLIT_JSON_PATH)
        print(
            "Train / Val   :",
            len(train_rows),
            "/",
            len(val_rows),
            "(all valid AE rows)",
        )
        print("Trajectory GT : 10 x (x,y,yaw), prepared from raw Part1")
        print("Training      : NOT STARTED (--validate-only)")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 Action Expert training requires RTX 3080 Ti / BF16-capable GPU."
        )

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)

    print("=" * 120)
    print("Reasoning_VLM_v2_fixedsplit | ALL 10 ACTION EXPERTS")
    print("=" * 120)
    print("GPU           :", torch.cuda.get_device_name(args.gpu_id))
    print("VLM           :", args.vlm)
    print("Dataset root  :", ACTION_DATASET_ROOT)
    print("Split JSON    :", SPLIT_JSON_PATH)
    print("Prompt mode   :", PROMPT_MODE)
    print("Train / Val   :", len(train_rows), "/", len(val_rows), "(all valid AE rows)")
    print("Cache root    :", args.cache_root)
    print("Model root    :", args.model_root)
    print("Families      :", ", ".join(args.families))
    print("Branches      :", ", ".join(args.branches))
    print("Architectures :", ", ".join(args.transformer_architectures))
    print("Ablation axes  : Input=direct/reasoning | Structure=encoder/decoder | Direction=bidirectional/causal")
    print("Naming axes   : input(direct/reasoning) / structure(encoder/decoder) / direction(bidirectional/causal)")
    print(
        "Capacity      :",
        f"H={args.hidden_dim} L={args.num_layers} "
        f"heads={args.num_heads} FF={args.ff_dim}",
    )
    print(
        "Batch         :",
        f"{args.batch_size} x accum {args.grad_accum} "
        f"= {args.batch_size * args.grad_accum}",
    )

    # 1) One unified Reasoning_VLM_v2_fixedsplit cache.
    train_manifest, val_manifest = build_cache(
        args,
        train_rows,
        val_rows,
        device,
        resolve_dtype(args.vlm_dtype),
    )

    input_dim = infer_input_dim(train_manifest)
    if infer_input_dim(val_manifest) != input_dim:
        raise RuntimeError("Train/Val KV input_dim mismatch")

    cache_signature = sha256_file(args.cache_root / "meta.json")

    # 2) VLM is already unloaded inside build_cache(). Train exactly one Action
    #    Expert at a time from here.
    args.model_root.mkdir(parents=True, exist_ok=True)
    results = []

    total_requested_models = (
        len(args.transformer_architectures) * len(args.branches)
        if "transformer" in args.families
        else 0
    ) + (
        len(args.branches)
        if "flow" in args.families
        else 0
    )

    all_models_start = time.perf_counter()
    completed_model_times: List[float] = []

    def print_overall_eta(last_model_name: str, last_model_seconds: float) -> None:
        completed_model_times.append(float(last_model_seconds))
        completed = len(completed_model_times)
        avg_model_sec = sum(completed_model_times) / completed
        remaining = max(0, total_requested_models - completed)
        total_eta = avg_model_sec * remaining
        total_elapsed = time.perf_counter() - all_models_start

        print(
            f"[OVERALL] completed={completed}/{total_requested_models} | "
            f"last={last_model_name} "
            f"({format_eta(last_model_seconds)}) | "
            f"elapsed={format_eta(total_elapsed)} | "
            f"eta~={format_eta(total_eta)}",
            flush=True,
        )

    # Training order: Flow/DiT v2 first, then Transformer ablations.
    if "flow" in args.families:
        oracle_error = linear_flow_oracle_sanity_check(
            seed=args.seed,
            solver_steps=10,
        )
        if oracle_error > 1e-5:
            raise RuntimeError(
                f"Flow oracle check failed: {oracle_error:.8e}"
            )

        normalizer = compute_normalizer(train_manifest)

        print("\n" + "=" * 120)
        print("FLOW / DiT v2 STARTUP")
        print("=" * 120)
        print("Oracle check :", f"PASS max_error={oracle_error:.3e}")
        print("Normalizer   :", normalizer.mode)

        # Branch order follows --branches (default: direct -> reasoning).
        for branch in args.branches:
            _model_t0 = time.perf_counter()
            r = train_flow(
                branch=branch,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                input_dim=input_dim,
                cache_signature=cache_signature,
                normalizer=normalizer,
                device=device,
                args=args,
            )
            _model_sec = time.perf_counter() - _model_t0
            results.append(r)
            save_summary(args, results)
            print_overall_eta(r["condition"], _model_sec)

    if "transformer" in args.families:
        # Naming order: input / structure / attention direction.
        # direct_* four models first, then reasoning_* four models.
        for branch in args.branches:
            for architecture in args.transformer_architectures:
                _model_t0 = time.perf_counter()
                r = train_direct(
                    architecture=architecture,
                    branch=branch,
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                    input_dim=input_dim,
                    cache_signature=cache_signature,
                    device=device,
                    args=args,
                )
                _model_sec = time.perf_counter() - _model_t0
                results.append(r)
                save_summary(args, results)
                print_overall_eta(r["condition"], _model_sec)

    save_summary(args, results)

    expected = (
        len(args.transformer_architectures) * len(args.branches)
        if "transformer" in args.families
        else 0
    ) + (
        len(args.branches)
        if "flow" in args.families
        else 0
    )

    print("\n" + "=" * 120)
    print("TRAINING COMPLETE")
    print("=" * 120)
    print(
        f"{'FAMILY':<13}{'CONDITION':<38}{'PARAMS(B)':>12}"
        f"{'BEST_E':>9}{'ADE':>11}{'FDE':>11}{'HEAD':>11}"
    )
    print("-" * 120)

    for r in results:
        print(
            f"{r['family']:<13}"
            f"{r['condition']:<38}"
            f"{r['trainable_params']/1e9:>12.3f}"
            f"{r['best_epoch']:>9d}"
            f"{r['val_ade_m']:>11.4f}"
            f"{r['val_fde_m']:>11.4f}"
            f"{r['val_heading_mae_rad']:>11.4f}"
        )

    print()
    print("Requested models :", expected)
    print("Completed models :", len(results))
    print("Summary          :", args.model_root / "summary.txt")
    print("Model root       :", args.model_root)


if __name__ == "__main__":
    main()
