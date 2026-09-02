#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning_VLM_v2 + 10 Action Experts | PART2 AR-ONLY TEST
=========================================================

This test script is matched to the training code that produced:

  /home/lhh/lab/models/action_expert/reasoning_vlm_v2_10model

Models
------
Transformer 8:
  direct_encoder_bidirectional
  direct_encoder_causal
  direct_decoder_bidirectional
  direct_decoder_causal
  reasoning_encoder_bidirectional
  reasoning_encoder_causal
  reasoning_decoder_bidirectional
  reasoning_decoder_causal

Flow/DiT 2:
  direct_flow_dit
  reasoning_flow_dit

Evaluation principle
--------------------
- SAME frozen VLM:
    /home/lhh/lab/models/vlm/Reasoning_VLM_v2
- SAME Reasoning_VLM_v2 prompt:
    3 cameras + mission command + speed + acceleration + heading/yaw
    + reasoning-only instruction
- SAME cache definition as training:
    direct    = prompt-boundary last-layer KV
    reasoning = direct KV + generated-reasoning KV delta
- TARGET greedy autoregressive generation only.
- DFlash is intentionally NOT used here.
- Held-out Part2 300 is used by default.
- The VLM test cache is generated once and then the VLM is unloaded.
- Action Experts are loaded/evaluated one at a time for VRAM safety.
- Flow models use the same Gaussian RNG seed/order for fair comparison.

Outputs
-------
<result-root>/<timestamp>/
  summary.txt
  summary.json
  generation_summary.json
  predictions/
    <condition>.jsonl

Default dataset
---------------
/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

ACTION_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_ROOT / "scripts"
VLM_SCRIPT_DIR = Path("/home/lhh/lab/VLM/scripts")

for _p in (SCRIPT_DIR, ACTION_ROOT, ACTION_SCRIPT_DIR, VLM_SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VLM_PATH = Path("/home/lhh/lab/models/vlm/Reasoning_VLM_v2")
MODEL_ROOT = Path("/home/lhh/lab/models/action_expert/reasoning_vlm_v2_10model")

DEFAULT_DATASET = Path("/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl")
RAW_PART2_ROOT = Path("/media/HDD/nuReasoning/train/part_2")

DEFAULT_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert10_v2/part2_test_kv_cache"
)
DEFAULT_RESULT_ROOT = Path("/home/lhh/lab/E2E/Result/reasoning_vlm_v2_10model")


# =============================================================================
# ACTION MODEL IMPORTS
# =============================================================================

from scripts.stored.action_model_ablation_v2 import (
    ARCHITECTURES,
    build_action_expert,
    trajectory_metrics_np,
)

try:
    from scripts.stored.action_model_flow_dit import (
        TrajectoryNormalizer,
        build_flow_dit,
        euler_sample,
    )
except ImportError:
    from scripts.stored.action_model_flow_dit import (
        TrajectoryNormalizer,
        build_flow_dit,
        euler_sample,
    )

import reasoning_v2_core as vlm_v2_core


# =============================================================================
# CONFIG
# =============================================================================

BRANCHES = ("direct", "reasoning")

EXPECTED_TRANSFORMER_ARCHITECTURES = (
    "encoder_bidirectional",
    "encoder_causal",
    "decoder_bidirectional",
    "decoder_causal",
)

NUM_STEPS = 10

PROMPT_MODE = "reasoning_v2_ego_state_reasoning_only"
ATTN_IMPLEMENTATION = "sdpa"
MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
MAX_NEW_TOKENS = 128
CACHE_STORAGE_DTYPE = torch.bfloat16

CACHE_VERSION = "reasoning_vlm_v2_part2_target_ar_test_v1"

SEED = 20260823
FLOW_TEST_SEED = 20260841

CAMERAS = ("front_left", "front", "front_right")


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


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []
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


def cache_filename(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}_{digest}.pt"


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return float(np.mean(vals)) if vals else float("nan")


def percentile(xs: Iterable[float], q: float) -> float:
    vals = [float(x) for x in xs]
    return float(np.percentile(vals, q)) if vals else float("nan")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


# =============================================================================
# PART2 ROW NORMALIZATION / EGO CONTEXT
# =============================================================================

_LEGACY_PICKLE_CLASS_CACHE: Dict[Tuple[str, str], type] = {}
_RAW_METADATA_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _legacy_pickle_class(module: str, name: str) -> type:
    key = (str(module), str(name))
    if key not in _LEGACY_PICKLE_CLASS_CACHE:
        cls = type(str(name), (), {})
        cls.__module__ = str(module)
        _LEGACY_PICKLE_CLASS_CACHE[key] = cls
    return _LEGACY_PICKLE_CLASS_CACHE[key]


class NuReasoningCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in {"data_schema", "data_schema_v0"}:
            return _legacy_pickle_class(module, name)
        return super().find_class(module, name)


def load_nureasoning_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return NuReasoningCompatUnpickler(f).load()


def _raw_clip_metadata(raw_root: Path, clip: str) -> Dict[str, Any]:
    clip = str(clip).strip()
    if not clip:
        raise ValueError("Missing clip name")

    key = (str(raw_root.resolve()), clip)
    if key not in _RAW_METADATA_CACHE:
        path = raw_root / clip / "metadata.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        _RAW_METADATA_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
    return _RAW_METADATA_CACHE[key]


def _resolve_raw_ego_path(row: Dict[str, Any], raw_root: Path) -> Path:
    clip = str(row.get("clip", "")).strip()
    sid = str(row.get("id", "")).strip()

    try:
        frame_index = int(row.get("frame_index", -1))
    except Exception:
        frame_index = -1

    metadata = _raw_clip_metadata(raw_root, clip)
    frames = metadata.get("frames", [])
    if not isinstance(frames, list):
        raise RuntimeError(f"metadata.frames is not a list: clip={clip}")

    chosen = None

    # 1) Prefer exact token/id match.
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        token = str(frame.get("token", "")).strip()
        if token and token == sid:
            chosen = frame
            break

    # 2) Fall back to frame_index.
    if chosen is None and frame_index >= 0:
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            try:
                idx = int(frame.get("frame_index", -999999))
            except Exception:
                continue
            if idx == frame_index:
                chosen = frame
                break

    if chosen is None:
        raise RuntimeError(
            f"Could not match raw Part2 frame: id={sid}, "
            f"clip={clip}, frame_index={frame_index}"
        )

    ego_value = chosen.get("ego_state")
    if not ego_value:
        raise KeyError(f"Missing ego_state in raw metadata: id={sid}")

    ego_path = Path(str(ego_value)).expanduser()
    if not ego_path.is_absolute():
        ego_path = (raw_root / clip / ego_path).resolve()

    if not ego_path.is_file():
        raise FileNotFoundError(ego_path)

    return ego_path


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def restore_v2_ego_context(
    row: Dict[str, Any],
    raw_root: Path,
) -> Dict[str, Any]:
    """
    Use fields already stored in part2_e2e_300.jsonl when all three exist.
    Otherwise restore them from original Part2 ego_state.pkl exactly through
    reasoning_v2_core helpers.
    """
    row = dict(row)

    speed = _finite_float(row.get("speed_mps"))
    acceleration = _finite_float(row.get("acceleration_mps2"))
    heading = _finite_float(
        row.get("heading_rad", row.get("yaw_rad"))
    )

    if speed is not None and acceleration is not None and heading is not None:
        row["speed_mps"] = speed
        row["acceleration_mps2"] = acceleration
        row["heading_rad"] = heading
        return row

    ego_path = _resolve_raw_ego_path(row, raw_root)
    ego_state = load_nureasoning_pickle(ego_path)

    _, _, restored_heading = vlm_v2_core.extract_pose(ego_state)
    restored_speed = vlm_v2_core.extract_speed_mps(ego_state)
    restored_accel = vlm_v2_core.extract_acceleration_mps2(
        ego_state,
        restored_heading,
    )

    if restored_speed is None or not math.isfinite(float(restored_speed)):
        raise RuntimeError(f"Missing/non-finite speed: id={row.get('id')}")
    if restored_accel is None or not math.isfinite(float(restored_accel)):
        raise RuntimeError(f"Missing/non-finite acceleration: id={row.get('id')}")
    if not math.isfinite(float(restored_heading)):
        raise RuntimeError(f"Missing/non-finite heading: id={row.get('id')}")

    row["speed_mps"] = float(restored_speed)
    row["acceleration_mps2"] = float(restored_accel)
    row["heading_rad"] = float(restored_heading)
    return row


def normalize_test_row(
    row: Dict[str, Any],
    raw_root: Path,
) -> Dict[str, Any]:
    row = dict(row)

    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Empty sample id")
    row["id"] = sid

    command = str(
        row.get("mission_command")
        or row.get("command")
        or ""
    ).strip()
    if not command:
        raise ValueError(f"Missing mission command id={sid}")
    row["mission_command"] = command
    row["command"] = command

    images = row.get("images")
    if isinstance(images, dict):
        required = ("front_left", "front", "front_right")
        missing = [k for k in required if not images.get(k)]
        if missing:
            raise ValueError(f"Missing images id={sid}: {missing}")
        paths = [str(images[k]) for k in required]
    elif isinstance(images, list) and len(images) == 3:
        paths = [str(x) for x in images]
    else:
        raise ValueError(f"Expected exactly 3 images id={sid}: {images}")

    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"id={sid}: {p}")
    row["images"] = paths

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (NUM_STEPS, 3):
        raise ValueError(f"trajectory must be [10,3] id={sid}, got={gt.shape}")
    if not np.isfinite(gt).all():
        raise ValueError(f"Non-finite trajectory id={sid}")
    row["trajectory"] = gt.tolist()

    row = restore_v2_ego_context(row, raw_root)
    return row


def load_test_rows(
    path: Path,
    raw_root: Path,
    limit: int,
) -> List[Dict[str, Any]]:
    raw = read_jsonl(path)
    if limit > 0:
        raw = raw[:limit]

    rows = [normalize_test_row(x, raw_root) for x in raw]

    ids = [x["id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate IDs in test dataset")

    if not rows:
        raise RuntimeError("No test rows selected")

    return rows


# =============================================================================
# Reasoning_VLM_v2 PROMPT + TARGET AR KV
# =============================================================================

def load_target_model(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
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

    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        str(path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    return model, processor


def reset_multimodal_rope_state(model) -> None:
    candidates = [model, getattr(model, "model", None)]
    inner = getattr(getattr(model, "model", None), "language_model", None)
    if inner is not None:
        candidates.append(inner)

    for obj in candidates:
        if obj is not None and hasattr(obj, "rope_deltas"):
            obj.rope_deltas = None


def make_reasoning_v2_record(row: Dict[str, Any]) -> Dict[str, Any]:
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
    rec = make_reasoning_v2_record(row)
    prompt = vlm_v2_core.build_prompt(rec)
    images = vlm_v2_core.open_three_images(rec)

    # Identical layout to the training script.
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
    max_length: Optional[int] = None,
) -> torch.Tensor:
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
    row: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    allow_truncated: bool,
) -> Dict[str, Any]:
    reset_multimodal_rope_state(model)

    batch = build_prompt_v2(
        processor=processor,
        row=row,
        device=device,
        dtype=dtype,
    )
    prompt_token_length = int(batch["input_ids"].shape[1])

    # 1) Direct prompt-boundary KV.
    sync_cuda(device)
    t0 = time.perf_counter()

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

    sync_cuda(device)
    prompt_forward_ms = (time.perf_counter() - t0) * 1000.0

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

    # 2) Same target-model greedy AR used by training cache extraction.
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)

    sync_cuda(device)
    t1 = time.perf_counter()

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        generation = model.generate(
            **batch,
            **generation_kwargs,
        )

    sync_cuda(device)
    generate_call_ms = (time.perf_counter() - t1) * 1000.0

    generation_cache = getattr(generation, "past_key_values", None)
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
        "prompt_forward_ms": float(prompt_forward_ms),
        "generate_call_ms": float(generate_call_ms),
        "vlm": str(VLM_PATH),
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
    }


# =============================================================================
# TEST CACHE
# =============================================================================

def cache_request(args, selected_ids: Sequence[str]) -> Dict[str, Any]:
    config_path = args.vlm / "config.json"

    return {
        "version": CACHE_VERSION,
        "vlm": str(args.vlm.resolve()),
        "vlm_config_sha256": (
            sha256_file(config_path) if config_path.is_file() else None
        ),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "samples": len(selected_ids),
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": args.max_new_tokens,
        "allow_truncated": args.allow_truncated,
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }


def valid_manifest(
    path: Path,
    expected_ids: Sequence[str],
) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None

    rows = read_jsonl(path)

    if [str(x["id"]) for x in rows] != list(expected_ids):
        return None

    if not all(Path(x["cache_file"]).is_file() for x in rows):
        return None

    return rows


def reuse_cache_if_valid(
    args,
    rows: Sequence[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    meta_path = args.cache_root / "meta.json"
    manifest_path = args.cache_root / "manifest.jsonl"

    if not meta_path.is_file():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    ids = [x["id"] for x in rows]

    if meta.get("request") != cache_request(args, ids):
        return None

    manifest = valid_manifest(manifest_path, ids)
    if manifest is None:
        return None

    print(f"[CACHE] reuse | samples={len(manifest)}")
    return manifest


def build_test_cache(
    args,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    vlm_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    if args.cache_root.exists() and not args.rebuild_cache:
        reused = reuse_cache_if_valid(args, rows)
        if reused is not None:
            return reused

        raise RuntimeError(
            f"Incompatible existing test cache: {args.cache_root}\n"
            "Use --rebuild-cache."
        )

    if args.cache_root.exists():
        shutil.rmtree(args.cache_root)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("STAGE 1 | PART2 Reasoning_VLM_v2 TARGET-AR TEST CACHE")
    print("=" * 120)
    print("VLM        :", args.vlm)
    print("Dataset    :", args.dataset)
    print("Samples    :", len(rows))
    print("Prompt     :", PROMPT_MODE)
    print("Decode     : target autoregressive greedy")
    print("DFlash     : NOT USED")
    print("Cache root :", args.cache_root)

    model, processor = load_target_model(
        args.vlm,
        device,
        vlm_dtype,
    )

    manifest: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for index, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        sid = row["id"]
        out_path = args.cache_root / cache_filename(index, sid)

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
            "direct_tokens": int(record["direct_kv"].shape[0]),
            "reasoning_tokens": int(record["reasoning_delta_kv"].shape[0]),
            "reasoning_text": record["reasoning_text"],
            "prefix_max_abs_diff": float(record["prefix_max_abs_diff"]),
            "prompt_forward_ms": float(record["prompt_forward_ms"]),
            "generate_call_ms": float(record["generate_call_ms"]),
        })

        sample_sec = time.perf_counter() - t0
        elapsed = time.perf_counter() - start
        avg_sec = elapsed / index
        eta = avg_sec * (len(rows) - index)

        print(
            f"[CACHE] {index:04d}/{len(rows):04d} | "
            f"id={sid} | "
            f"directT={record['direct_kv'].shape[0]} "
            f"reasonT={record['reasoning_delta_kv'].shape[0]} "
            f"D={record['direct_kv'].shape[-1]} | "
            f"reason='{record['reasoning_text'][:70]}' | "
            f"sample={sample_sec:.2f}s | "
            f"eta={format_eta(eta)}",
            flush=True,
        )

        del record

    write_jsonl(args.cache_root / "manifest.jsonl", manifest)

    meta = {
        "request": cache_request(args, [x["id"] for x in rows]),
        "runtime": "Reasoning_VLM_v2 target-only AR",
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "prompt_mode": PROMPT_MODE,
        "attention_implementation": ATTN_IMPLEMENTATION,
        "samples": len(manifest),
    }

    (args.cache_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, processor
    cleanup_cuda()

    return manifest


def infer_input_dim(manifest: Sequence[Dict[str, Any]]) -> int:
    cache = torch.load(
        manifest[0]["cache_file"],
        map_location="cpu",
        weights_only=False,
    )

    direct = cache["direct_kv"]
    delta = cache["reasoning_delta_kv"]

    if direct.ndim != 2 or delta.ndim != 2:
        raise RuntimeError("KV tensors must be rank-2")
    if direct.shape[-1] != delta.shape[-1]:
        raise RuntimeError("direct/reasoning KV dim mismatch")

    return int(direct.shape[-1])


# =============================================================================
# CHECKPOINT DISCOVERY / VALIDATION
# =============================================================================

def expected_transformer_paths(model_root: Path) -> Dict[str, Path]:
    return {
        f"{branch}_{arch}":
            model_root / "transformer" / f"{branch}_{arch}" / "best.pt"
        for branch in BRANCHES
        for arch in EXPECTED_TRANSFORMER_ARCHITECTURES
    }


def expected_flow_paths(model_root: Path) -> Dict[str, Path]:
    return {
        f"{branch}_flow_dit":
            model_root / "flow" / f"{branch}_flow_dit" / "best.pt"
        for branch in BRANCHES
    }


def validate_checkpoint_common(
    ckpt: Dict[str, Any],
    path: Path,
    expected_branch: str,
    expected_input_dim: int,
    expected_vlm: Path,
) -> None:
    if "model_state_dict" not in ckpt:
        raise KeyError(f"model_state_dict missing: {path}")
    if "action_config" not in ckpt:
        raise KeyError(f"action_config missing: {path}")
    if "input_dim" not in ckpt:
        raise KeyError(f"input_dim missing: {path}")

    saved_branch = str(ckpt.get("branch", ""))
    if saved_branch and saved_branch != expected_branch:
        raise RuntimeError(
            f"Branch mismatch: expected={expected_branch}, "
            f"saved={saved_branch}, path={path}"
        )

    saved_input_dim = int(ckpt["input_dim"])
    if saved_input_dim != expected_input_dim:
        raise RuntimeError(
            f"input_dim mismatch: cache={expected_input_dim}, "
            f"checkpoint={saved_input_dim}, path={path}"
        )

    saved_prompt = ckpt.get("prompt_mode")
    if saved_prompt is not None and str(saved_prompt) != PROMPT_MODE:
        raise RuntimeError(
            f"prompt_mode mismatch: expected={PROMPT_MODE}, "
            f"saved={saved_prompt}, path={path}"
        )

    saved_vlm = ckpt.get("vlm")
    if saved_vlm is not None:
        saved_vlm_path = Path(str(saved_vlm)).expanduser().resolve()
        if saved_vlm_path != expected_vlm.expanduser().resolve():
            raise RuntimeError(
                f"VLM mismatch: expected={expected_vlm.resolve()}, "
                f"saved={saved_vlm_path}, path={path}"
            )


def checkpoint_precheck(
    args,
    input_dim: int,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    transformer_paths = expected_transformer_paths(args.model_root)
    flow_paths = expected_flow_paths(args.model_root)

    print("\n" + "=" * 120)
    print("STAGE 2 | ACTION EXPERT CHECKPOINT PRECHECK")
    print("=" * 120)

    # Also show what the imported implementation actually exposes.
    missing_arches = [
        x for x in EXPECTED_TRANSFORMER_ARCHITECTURES
        if x not in ARCHITECTURES
    ]
    if missing_arches:
        raise RuntimeError(
            "Current action_model_ablation_v2.ARCHITECTURES does not contain "
            f"the architectures used by this training run: {missing_arches}\n"
            f"Current ARCHITECTURES={tuple(ARCHITECTURES)}"
        )

    for name, path in transformer_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        branch = str(ckpt.get("branch", name.split("_", 1)[0]))
        architecture = str(ckpt.get("architecture", name[len(branch) + 1:]))

        validate_checkpoint_common(
            ckpt,
            path,
            expected_branch=branch,
            expected_input_dim=input_dim,
            expected_vlm=args.vlm,
        )

        if branch not in BRANCHES:
            raise RuntimeError(f"Unknown branch in checkpoint: {branch}")

        expected_name = f"{branch}_{architecture}"
        if expected_name != name:
            raise RuntimeError(
                f"Transformer checkpoint naming mismatch: "
                f"folder={name}, checkpoint={expected_name}"
            )

        print(
            f"[OK] Transformer {name:<34s} "
            f"epoch={int(ckpt.get('epoch', -1)):02d} "
            f"D={int(ckpt['input_dim'])}"
        )

        del ckpt

    for name, path in flow_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        branch = str(ckpt.get("branch", name.replace("_flow_dit", "")))

        validate_checkpoint_common(
            ckpt,
            path,
            expected_branch=branch,
            expected_input_dim=input_dim,
            expected_vlm=args.vlm,
        )

        if "normalizer" not in ckpt:
            raise KeyError(f"normalizer missing: {path}")

        print(
            f"[OK] Flow        {name:<34s} "
            f"epoch={int(ckpt.get('epoch', -1)):02d} "
            f"D={int(ckpt['input_dim'])}"
        )

        del ckpt

    print(
        f"[PRECHECK] PASS | Transformer={len(transformer_paths)} "
        f"Flow={len(flow_paths)} Total={len(transformer_paths)+len(flow_paths)}"
    )

    return transformer_paths, flow_paths


# =============================================================================
# CACHE -> ACTION MODEL INPUT
# =============================================================================

def load_action_input(
    item: Dict[str, Any],
    branch: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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

    if branch == "direct":
        memory = direct
        segment_ids = torch.zeros(
            direct.shape[0],
            dtype=torch.long,
        )
    elif branch == "reasoning":
        memory = torch.cat([direct, delta], dim=0)
        segment_ids = torch.cat([
            torch.zeros(direct.shape[0], dtype=torch.long),
            torch.ones(delta.shape[0], dtype=torch.long),
        ])
    else:
        raise ValueError(branch)

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
    gt = gt.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    inputs = {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }
    return inputs, gt


# =============================================================================
# MODEL LOADERS
# =============================================================================

def load_transformer(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["action_config"]
    architecture = str(ckpt["architecture"])

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
    ).to(
        device=device,
        dtype=torch.float32,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, ckpt


def load_flow(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["action_config"]

    model = build_flow_dit(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    normalizer = TrajectoryNormalizer.from_dict(
        ckpt["normalizer"]
    ).to(device)

    solver_steps = int(
        cfg.get("solver_steps", 10)
    )

    return model, ckpt, normalizer, solver_steps


# =============================================================================
# ACTION EXPERT EVALUATION
# =============================================================================

@torch.inference_mode()
def warmup_transformer(
    model,
    item: Dict[str, Any],
    branch: str,
    device: torch.device,
    runs: int,
) -> None:
    if runs <= 0:
        return

    inputs, _ = load_action_input(item, branch, device)

    for _ in range(runs):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            _ = model(**inputs)

    sync_cuda(device)


@torch.inference_mode()
def evaluate_transformer(
    condition: str,
    checkpoint_path: Path,
    manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    prediction_dir: Path,
    warmup_runs: int,
) -> Dict[str, Any]:
    model, ckpt = load_transformer(
        checkpoint_path,
        device,
    )

    branch = str(ckpt["branch"])
    architecture = str(ckpt["architecture"])

    warmup_transformer(
        model,
        manifest[0],
        branch,
        device,
        warmup_runs,
    )

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    latencies: List[float] = []
    pred_rows: List[Dict[str, Any]] = []

    start = time.perf_counter()

    for index, item in enumerate(manifest, 1):
        inputs, gt = load_action_input(
            item,
            branch,
            device,
        )

        sync_cuda(device)
        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            pred = model(**inputs)

        sync_cuda(device)
        action_ms = (time.perf_counter() - t0) * 1000.0

        pred_np = pred.float().cpu().numpy()
        gt_np = gt.float().cpu().numpy()

        preds.append(pred_np)
        gts.append(gt_np)
        latencies.append(action_ms)

        pred_rows.append({
            "id": str(item["id"]),
            "condition": condition,
            "branch": branch,
            "architecture": architecture,
            "prediction": pred_np[0].tolist(),
            "ground_truth": gt_np[0].tolist(),
            "action_ms": float(action_ms),
        })

        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(manifest) - index)

        if (
            index == 1
            or index % 25 == 0
            or index == len(manifest)
        ):
            print(
                f"[TEST {condition}] "
                f"{index:04d}/{len(manifest):04d} | "
                f"AE={action_ms:.2f}ms | "
                f"eta={format_eta(eta)}",
                flush=True,
            )

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)

    out_path = prediction_dir / f"{condition}.jsonl"
    write_jsonl(out_path, pred_rows)

    result = {
        "family": "transformer",
        "condition": condition,
        "branch": branch,
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "samples": len(manifest),
        "ade_m": float(metrics["ade_m"]),
        "fde_m": float(metrics["fde_m"]),
        "heading_mae_rad": float(metrics["heading_mae_rad"]),
        "action_ms_mean": mean(latencies),
        "action_ms_p50": percentile(latencies, 50),
        "action_ms_p95": percentile(latencies, 95),
        "prediction_file": str(out_path),
    }

    del model, ckpt, preds, gts, pred_rows
    cleanup_cuda()

    return result


@torch.inference_mode()
def warmup_flow(
    model,
    normalizer,
    solver_steps: int,
    item: Dict[str, Any],
    branch: str,
    device: torch.device,
    runs: int,
) -> None:
    if runs <= 0:
        return

    inputs, _ = load_action_input(item, branch, device)

    for i in range(runs):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(FLOW_TEST_SEED + 100000 + i)

        _ = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=normalizer,
            solver_steps=solver_steps,
            rng=rng,
        )

    sync_cuda(device)


@torch.inference_mode()
def evaluate_flow(
    condition: str,
    checkpoint_path: Path,
    manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    prediction_dir: Path,
    warmup_runs: int,
    solver_steps_override: Optional[int],
) -> Dict[str, Any]:
    model, ckpt, normalizer, checkpoint_solver_steps = load_flow(
        checkpoint_path,
        device,
    )

    branch = str(ckpt["branch"])
    solver_steps = (
        int(solver_steps_override)
        if solver_steps_override is not None
        else int(checkpoint_solver_steps)
    )

    warmup_flow(
        model,
        normalizer,
        solver_steps,
        manifest[0],
        branch,
        device,
        warmup_runs,
    )

    # Reset the exact same RNG stream for every Flow condition.
    flow_rng = torch.Generator(device="cpu")
    flow_rng.manual_seed(FLOW_TEST_SEED)

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    latencies: List[float] = []
    pred_rows: List[Dict[str, Any]] = []

    start = time.perf_counter()

    for index, item in enumerate(manifest, 1):
        inputs, gt = load_action_input(
            item,
            branch,
            device,
        )

        sync_cuda(device)
        t0 = time.perf_counter()

        pred = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=normalizer,
            solver_steps=solver_steps,
            rng=flow_rng,
        )

        sync_cuda(device)
        action_ms = (time.perf_counter() - t0) * 1000.0

        pred_np = pred.float().cpu().numpy()
        gt_np = gt.float().cpu().numpy()

        preds.append(pred_np)
        gts.append(gt_np)
        latencies.append(action_ms)

        pred_rows.append({
            "id": str(item["id"]),
            "condition": condition,
            "branch": branch,
            "architecture": "flow_dit",
            "solver_steps": int(solver_steps),
            "prediction": pred_np[0].tolist(),
            "ground_truth": gt_np[0].tolist(),
            "action_ms": float(action_ms),
        })

        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(manifest) - index)

        if (
            index == 1
            or index % 25 == 0
            or index == len(manifest)
        ):
            print(
                f"[TEST {condition}] "
                f"{index:04d}/{len(manifest):04d} | "
                f"AE={action_ms:.2f}ms | "
                f"eta={format_eta(eta)}",
                flush=True,
            )

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)

    out_path = prediction_dir / f"{condition}.jsonl"
    write_jsonl(out_path, pred_rows)

    result = {
        "family": "flow",
        "condition": condition,
        "branch": branch,
        "architecture": "flow_dit",
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "samples": len(manifest),
        "solver_steps": int(solver_steps),
        "ade_m": float(metrics["ade_m"]),
        "fde_m": float(metrics["fde_m"]),
        "heading_mae_rad": float(metrics["heading_mae_rad"]),
        "action_ms_mean": mean(latencies),
        "action_ms_p50": percentile(latencies, 50),
        "action_ms_p95": percentile(latencies, 95),
        "prediction_file": str(out_path),
    }

    del model, ckpt, preds, gts, pred_rows
    cleanup_cuda()

    return result


# =============================================================================
# GENERATION / RESULT SUMMARY
# =============================================================================

def summarize_generation(
    manifest: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    direct_tokens = [int(x["direct_tokens"]) for x in manifest]
    reasoning_tokens = [int(x["reasoning_tokens"]) for x in manifest]
    prefix_diff = [float(x["prefix_max_abs_diff"]) for x in manifest]
    prompt_ms = [float(x["prompt_forward_ms"]) for x in manifest]
    generate_ms = [float(x["generate_call_ms"]) for x in manifest]

    return {
        "samples": len(manifest),
        "direct_tokens_mean": mean(direct_tokens),
        "direct_tokens_min": int(min(direct_tokens)),
        "direct_tokens_max": int(max(direct_tokens)),
        "reasoning_tokens_mean": mean(reasoning_tokens),
        "reasoning_tokens_min": int(min(reasoning_tokens)),
        "reasoning_tokens_max": int(max(reasoning_tokens)),
        "prefix_max_abs_diff_max": float(max(prefix_diff)),
        "prompt_forward_ms_mean": mean(prompt_ms),
        "generate_call_ms_mean": mean(generate_ms),
        "note": (
            "generate_call_ms is the full model.generate() call used to reproduce "
            "the training cache and includes its own prompt processing. "
            "Do not add it to prompt_forward_ms as a physical E2E latency."
        ),
    }


def delta_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

    for r in results:
        key = (str(r["family"]), str(r["architecture"]))
        by_key.setdefault(key, {})[str(r["branch"])] = r

    out: List[Dict[str, Any]] = []

    for (family, arch), branches in by_key.items():
        if "direct" not in branches or "reasoning" not in branches:
            continue

        direct = branches["direct"]
        reasoning = branches["reasoning"]

        out.append({
            "family": family,
            "architecture": arch,
            "direct_ade_m": direct["ade_m"],
            "reasoning_ade_m": reasoning["ade_m"],
            "delta_ade_m": reasoning["ade_m"] - direct["ade_m"],
            "delta_ade_pct": (
                (reasoning["ade_m"] - direct["ade_m"])
                / direct["ade_m"] * 100.0
                if direct["ade_m"] != 0
                else float("nan")
            ),
            "direct_fde_m": direct["fde_m"],
            "reasoning_fde_m": reasoning["fde_m"],
            "delta_fde_m": reasoning["fde_m"] - direct["fde_m"],
            "direct_heading_mae_rad": direct["heading_mae_rad"],
            "reasoning_heading_mae_rad": reasoning["heading_mae_rad"],
            "delta_heading_mae_rad": (
                reasoning["heading_mae_rad"]
                - direct["heading_mae_rad"]
            ),
        })

    return out


def format_summary(
    args,
    generation_summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    deltas: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "=" * 150,
        "Reasoning_VLM_v2 | PART2 10 ACTION EXPERT TEST | TARGET AR ONLY",
        "=" * 150,
        f"Dataset       : {args.dataset}",
        f"Samples       : {generation_summary['samples']}",
        f"VLM           : {args.vlm}",
        f"Prompt mode   : {PROMPT_MODE}",
        f"Model root    : {args.model_root}",
        f"Test cache    : {args.cache_root}",
        f"DFlash        : NOT USED",
        "",
        "VLM CACHE STATS",
        "-" * 150,
        (
            f"Direct tokens    : mean={generation_summary['direct_tokens_mean']:.2f} "
            f"min={generation_summary['direct_tokens_min']} "
            f"max={generation_summary['direct_tokens_max']}"
        ),
        (
            f"Reasoning tokens : mean={generation_summary['reasoning_tokens_mean']:.2f} "
            f"min={generation_summary['reasoning_tokens_min']} "
            f"max={generation_summary['reasoning_tokens_max']}"
        ),
        (
            f"Prefix max diff  : "
            f"{generation_summary['prefix_max_abs_diff_max']:.6e}"
        ),
        (
            f"Prompt forward   : "
            f"{generation_summary['prompt_forward_ms_mean']:.2f} ms mean"
        ),
        (
            f"generate() call  : "
            f"{generation_summary['generate_call_ms_mean']:.2f} ms mean"
        ),
        "",
        "ACTION EXPERT RESULTS",
        "-" * 150,
        (
            f"{'FAMILY':<13}{'CONDITION':<38}{'BEST_E':>8}"
            f"{'ADE(m)':>11}{'FDE(m)':>11}{'HEAD(rad)':>12}"
            f"{'AEmean':>11}{'AEp50':>11}{'AEp95':>11}"
        ),
        "-" * 150,
    ]

    for r in results:
        lines.append(
            f"{r['family']:<13}"
            f"{r['condition']:<38}"
            f"{r['best_epoch']:>8d}"
            f"{r['ade_m']:>11.4f}"
            f"{r['fde_m']:>11.4f}"
            f"{r['heading_mae_rad']:>12.4f}"
            f"{r['action_ms_mean']:>11.2f}"
            f"{r['action_ms_p50']:>11.2f}"
            f"{r['action_ms_p95']:>11.2f}"
        )

    lines += [
        "",
        "REASONING - DIRECT DELTA",
        "-" * 150,
        "Negative ADE/FDE delta = reasoning KV improved trajectory error.",
        (
            f"{'FAMILY':<13}{'ARCH':<24}"
            f"{'DIR_ADE':>11}{'REA_ADE':>11}{'ΔADE':>11}{'ΔADE%':>10}"
            f"{'DIR_FDE':>11}{'REA_FDE':>11}{'ΔFDE':>11}"
            f"{'ΔHEAD':>11}"
        ),
        "-" * 150,
    ]

    for d in deltas:
        lines.append(
            f"{d['family']:<13}"
            f"{d['architecture']:<24}"
            f"{d['direct_ade_m']:>11.4f}"
            f"{d['reasoning_ade_m']:>11.4f}"
            f"{d['delta_ade_m']:>11.4f}"
            f"{d['delta_ade_pct']:>10.2f}"
            f"{d['direct_fde_m']:>11.4f}"
            f"{d['reasoning_fde_m']:>11.4f}"
            f"{d['delta_fde_m']:>11.4f}"
            f"{d['delta_heading_mae_rad']:>11.4f}"
        )

    best = min(results, key=lambda x: x["ade_m"])
    fastest = min(results, key=lambda x: x["action_ms_mean"])

    lines += [
        "",
        "BEST",
        "-" * 150,
        (
            f"Lowest ADE : {best['condition']} "
            f"-> ADE={best['ade_m']:.4f} m, "
            f"FDE={best['fde_m']:.4f} m, "
            f"HEAD={best['heading_mae_rad']:.4f} rad"
        ),
        (
            f"Fastest AE : {fastest['condition']} "
            f"-> {fastest['action_ms_mean']:.2f} ms mean"
        ),
        "",
        "Latency note:",
        "  AEmean/AEp50/AEp95 are Action Expert-only latency.",
        "  VLM generate() timing is reported separately because this test cache extraction",
        "  intentionally mirrors the training code's second full model.generate() call.",
        "",
        "=" * 150,
    ]

    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--raw-part2-root", type=Path, default=RAW_PART2_ROOT)
    p.add_argument("--vlm", type=Path, default=VLM_PATH)
    p.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)

    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--gpu-id", type=int, default=0)

    p.add_argument(
        "--vlm-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    p.add_argument("--allow-truncated", action="store_true")

    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Untimed warmup Action Expert runs per model.",
    )
    p.add_argument(
        "--flow-solver-steps",
        type=int,
        default=None,
        help="Override Flow solver steps. Default: checkpoint value (normally 10).",
    )
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument(
        "--families",
        nargs="+",
        choices=("transformer", "flow"),
        default=["transformer", "flow"],
    )
    p.add_argument(
        "--branches",
        nargs="+",
        choices=BRANCHES,
        default=list(BRANCHES),
    )

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.vlm_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 requested but current GPU does not report BF16 support."
        )

    if args.limit == 0:
        raise ValueError("--limit must be > 0 or < 0 for all samples")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if (
        args.flow_solver_steps is not None
        and args.flow_solver_steps < 1
    ):
        raise ValueError("--flow-solver-steps must be >= 1")

    for p in (args.dataset, args.vlm, args.model_root):
        if not p.exists():
            raise FileNotFoundError(p)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed)

    print("=" * 120)
    print("Reasoning_VLM_v2 | PART2 10 ACTION EXPERT TEST")
    print("=" * 120)
    print("GPU           :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset       :", args.dataset)
    print("Raw Part2     :", args.raw_part2_root)
    print("VLM           :", args.vlm)
    print("Prompt mode   :", PROMPT_MODE)
    print("Model root    :", args.model_root)
    print("Cache root    :", args.cache_root)
    print("Families      :", ", ".join(args.families))
    print("Branches      :", ", ".join(args.branches))
    print("DFlash        : NOT USED")

    rows = load_test_rows(
        path=args.dataset,
        raw_root=args.raw_part2_root,
        limit=args.limit,
    )

    print("Selected rows :", len(rows))

    # 1) Build/reuse one unified Part2 Reasoning_VLM_v2 cache.
    manifest = build_test_cache(
        args=args,
        rows=rows,
        device=device,
        vlm_dtype=resolve_dtype(args.vlm_dtype),
    )

    input_dim = infer_input_dim(manifest)

    # 2) Validate all 10 checkpoints before expensive evaluation.
    transformer_paths, flow_paths = checkpoint_precheck(
        args,
        input_dim,
    )

    # Result folder is created after cache/precheck pass.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.result_root / timestamp
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    generation_summary = summarize_generation(manifest)
    (run_dir / "generation_summary.json").write_text(
        json.dumps(
            generation_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 120)
    print("STAGE 3 | ACTION EXPERT TEST")
    print("=" * 120)

    # Same ordering style as training: Flow first, then Transformer.
    if "flow" in args.families:
        for branch in args.branches:
            condition = f"{branch}_flow_dit"
            path = flow_paths[condition]

            print("\n" + "-" * 120)
            print(f"FLOW TEST | {condition}")
            print("-" * 120)

            result = evaluate_flow(
                condition=condition,
                checkpoint_path=path,
                manifest=manifest,
                device=device,
                prediction_dir=prediction_dir,
                warmup_runs=args.warmup_runs,
                solver_steps_override=args.flow_solver_steps,
            )
            results.append(result)

            print(
                f"[RESULT] {condition} | "
                f"ADE={result['ade_m']:.4f}m "
                f"FDE={result['fde_m']:.4f}m "
                f"HEAD={result['heading_mae_rad']:.4f}rad "
                f"AE={result['action_ms_mean']:.2f}ms"
            )

    if "transformer" in args.families:
        for branch in args.branches:
            for arch in EXPECTED_TRANSFORMER_ARCHITECTURES:
                condition = f"{branch}_{arch}"
                path = transformer_paths[condition]

                print("\n" + "-" * 120)
                print(f"TRANSFORMER TEST | {condition}")
                print("-" * 120)

                result = evaluate_transformer(
                    condition=condition,
                    checkpoint_path=path,
                    manifest=manifest,
                    device=device,
                    prediction_dir=prediction_dir,
                    warmup_runs=args.warmup_runs,
                )
                results.append(result)

                print(
                    f"[RESULT] {condition} | "
                    f"ADE={result['ade_m']:.4f}m "
                    f"FDE={result['fde_m']:.4f}m "
                    f"HEAD={result['heading_mae_rad']:.4f}rad "
                    f"AE={result['action_ms_mean']:.2f}ms"
                )

    if not results:
        raise RuntimeError("No model was evaluated")

    deltas = delta_rows(results)

    payload = {
        "experiment": "reasoning_vlm_v2_part2_10model_ar_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "samples": len(manifest),
        "vlm": str(args.vlm),
        "prompt_mode": PROMPT_MODE,
        "decode_backend": "target_autoregressive",
        "dflash_used": False,
        "model_root": str(args.model_root),
        "cache_root": str(args.cache_root),
        "generation": generation_summary,
        "results": results,
        "reasoning_minus_direct": deltas,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_text = format_summary(
        args=args,
        generation_summary=generation_summary,
        results=results,
        deltas=deltas,
    )

    (run_dir / "summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print("\n" + summary_text)
    print("Result dir :", run_dir)
    print("Summary    :", run_dir / "summary.txt")
    print("JSON       :", run_dir / "summary.json")


if __name__ == "__main__":
    main()
