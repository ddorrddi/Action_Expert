#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract controlled RAW last-layer K/V caches for a 2x2 ablation.

VLM
---
/home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit

Dataset
-------
/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl

Ablation axes
-------------
Axis A: direct vs reuse
  direct : prompt-boundary RAW K/V only
  reuse  : prompt-boundary RAW K/V + generated trace RAW K/V

Axis B: trajectory vs reasoning
  trajectory : exact trajectory prompt used by Reasoning_VLM_v2 training
  reasoning  : exact reasoning prompt used by Reasoning_VLM_v2 training

The image paths, mission command, speed, acceleration, heading, processor,
image pixel budget, chat template, VLM weights, decoding policy, and sample
order are identical between trajectory/reasoning conditions. Only the final
TASK INSTRUCTION differs.

Stored per sample
-----------------
reasoning_direct_key/value
reasoning_delta_key/value
trajectory_direct_key/value
trajectory_delta_key/value
trajectory_gt                  [10,3] (x,y,yaw) for Action Expert evaluation

No learned projection is applied to VLM K/V.
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
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# DEFAULTS
# =============================================================================

VLM_PATH = Path("/home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit")
TEST_JSONL = Path("/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl")
OUTPUT_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit"
)

SEED = 20260823
NUM_STEPS = 10
MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
CACHE_DTYPE = torch.bfloat16
CACHE_VERSION = "fixedsplit_trace_rawkv_ablation_v1"

CAMERAS = ("front_left", "front", "front_right")
TASKS = ("trajectory", "reasoning")


# =============================================================================
# EXACT PROMPTS FROM Reasoning_VLM_v2 TRAINING
# =============================================================================

def build_context(rec: Dict[str, Any]) -> str:
    return (
        "You are an autonomous-driving assistant.\n"
        "Three synchronized camera views are provided in this order: "
        "front-left, front, front-right.\n\n"
        "Driving context:\n"
        f"- Mission command: {rec['command']}\n"
        f"- Current speed: {float(rec['speed_mps']):.3f} m/s\n"
        f"- Current signed longitudinal acceleration: "
        f"{float(rec['acceleration_mps2']):.3f} m/s^2 "
        "(positive means accelerating forward, negative means decelerating)\n"
        f"- Current heading/yaw from the ego state: "
        f"{float(rec['heading_rad']):.6f} rad\n\n"
        "Use the camera observations together with the mission command and "
        "ego-state dynamics when they are relevant. "
        "Treat heading/yaw as the current orientation value, not by itself "
        "as evidence that the vehicle is turning.\n\n"
    )


def build_task_prompt(rec: Dict[str, Any], task: str) -> str:
    if task == "reasoning":
        return (
            build_context(rec)
            + "Generate the driving reasoning for the current situation. "
            "Use the three camera views, mission command, current speed, "
            "signed longitudinal acceleration, and heading/yaw when relevant. "
            "Explain the important scene evidence, safety constraints, route "
            "intention, and ego-motion context that justify the appropriate "
            "driving behavior. "
            "Do not output a separate final action label, JSON, or trajectory. "
            "Return only the natural-language reasoning trace."
        )

    if task == "trajectory":
        return (
            build_context(rec)
            + "Predict the vehicle's trajectory for the next 5 seconds "
            "as 10 waypoints at 0.5 s intervals, "
            "in the ego frame "
            "(x forward, y left, meters).\n"
            "Answer with exactly 10 comma-separated pairs like: "
            "(x1,y1), (x2,y2), ... "
            "with one decimal place."
        )

    raise ValueError(f"Unknown task: {task}")


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSONL parse error: {path}:{line_no}") from exc
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
    patterns = ("*.safetensors", "*.bin", "*.json", "*.model", "*.txt")
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise RuntimeError(f"No model/config files found under {path}")

    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(str(p.relative_to(path)).encode("utf-8"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(st.st_mtime_ns).encode("ascii"))
    return h.hexdigest()


def cache_filename(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}_{digest}.pt"


def finite_float(value: Any, name: str, sample_id: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Bad {name}: id={sample_id} value={value}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {name}: id={sample_id}")
    return out


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Empty sample id")
    row["id"] = sid

    command = str(row.get("command") or row.get("mission_command") or "").strip()
    if not command:
        raise ValueError(f"Missing mission command: id={sid}")
    row["command"] = command

    images = row.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"Expected exactly 3 images: id={sid}")
    images = [str(x) for x in images]
    for p in images:
        if not Path(p).is_file():
            raise FileNotFoundError(f"id={sid}: {p}")
    row["images"] = images

    row["speed_mps"] = finite_float(row.get("speed_mps"), "speed_mps", sid)
    row["acceleration_mps2"] = finite_float(
        row.get("acceleration_mps2"), "acceleration_mps2", sid
    )
    row["heading_rad"] = finite_float(row.get("heading_rad"), "heading_rad", sid)

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (NUM_STEPS, 3):
        raise ValueError(f"trajectory must be [10,3]: id={sid}, got={gt.shape}")
    if not np.isfinite(gt).all():
        raise ValueError(f"Non-finite trajectory: id={sid}")
    row["trajectory"] = gt.tolist()
    return row


def row_fingerprint(row: Dict[str, Any]) -> str:
    """Fingerprint all controlled non-task inputs."""
    payload = {
        "id": row["id"],
        "images": row["images"],
        "command": row["command"],
        "speed_mps": float(row["speed_mps"]),
        "acceleration_mps2": float(row["acceleration_mps2"]),
        "heading_rad": float(row["heading_rad"]),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


# =============================================================================
# MODEL / PROCESSOR
# =============================================================================

def load_target_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
):
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    kwargs = {
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), torch_dtype=dtype, **kwargs
        )

    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
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


def open_three_images(row: Dict[str, Any]) -> List[Image.Image]:
    result: List[Image.Image] = []
    for p in row["images"]:
        with Image.open(p) as im:
            result.append(im.convert("RGB").copy())
    return result


def build_batch(
    processor,
    row: Dict[str, Any],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], str]:
    prompt = build_task_prompt(row, task)
    images = open_three_images(row)

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
        truncation=False,
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
    return moved, prompt_text


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

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if past_key_values.key_cache and past_key_values.value_cache:
            return past_key_values.key_cache[-1], past_key_values.value_cache[-1]

    if isinstance(past_key_values, (tuple, list)):
        last = past_key_values[-1]
        if isinstance(last, (tuple, list)) and len(last) >= 2:
            return last[0], last[1]

    raise RuntimeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def raw_kv_to_cpu(
    key: torch.Tensor,
    value: torch.Tensor,
    max_length: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"Expected K/V [B,H,T,D], got {key.shape}, {value.shape}")
    if key.shape != value.shape or key.shape[0] != 1:
        raise ValueError("K/V shape or batch mismatch")
    if max_length is not None:
        key = key[..., :max_length, :]
        value = value[..., :max_length, :]
    return (
        key[0].detach().to(dtype=CACHE_DTYPE).cpu().contiguous(),
        value[0].detach().to(dtype=CACHE_DTYPE).cpu().contiguous(),
    )


def normalize_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(x) for x in value if x is not None}
    return {int(value)}


def content_token_ids(processor, model, new_ids: torch.Tensor) -> Tuple[List[int], bool]:
    eos_ids: set[int] = set()
    eos_ids |= normalize_token_id_set(getattr(processor.tokenizer, "eos_token_id", None))
    eos_ids |= normalize_token_id_set(
        getattr(getattr(model, "generation_config", None), "eos_token_id", None)
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
    pad_ids |= normalize_token_id_set(getattr(processor.tokenizer, "pad_token_id", None))
    pad_ids |= normalize_token_id_set(
        getattr(getattr(model, "generation_config", None), "pad_token_id", None)
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


# =============================================================================
# ONE TASK EXTRACTION
# =============================================================================

@torch.inference_mode()
def extract_task_cache(
    model,
    processor,
    row: Dict[str, Any],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    allow_truncated: bool,
    max_prefix_diff: float,
) -> Dict[str, Any]:
    reset_multimodal_rope_state(model)
    batch, prompt_text = build_batch(processor, row, task, device, dtype)
    prompt_token_length = int(batch["input_ids"].shape[1])

    # Prompt-only prefill = DIRECT condition.
    with torch.autocast(
        device_type="cuda", dtype=dtype, enabled=device.type == "cuda"
    ):
        prompt_out = model(**batch, use_cache=True, return_dict=True)

    prompt_key, prompt_value = get_last_layer_kv(prompt_out.past_key_values)
    prompt_cache_length = int(prompt_key.shape[-2])
    direct_key, direct_value = raw_kv_to_cpu(
        prompt_key, prompt_value, max_length=prompt_cache_length
    )

    text_cfg = getattr(model.config, "text_config", model.config)
    hq = int(getattr(text_cfg, "num_attention_heads", direct_key.shape[0]))
    hkv = int(direct_key.shape[0])
    head_dim = int(direct_key.shape[-1])
    if hq % hkv != 0:
        raise RuntimeError(f"Invalid GQA geometry Hq={hq} Hkv={hkv}")

    del prompt_out, prompt_key, prompt_value

    # Greedy self-generated task trace.
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
        device_type="cuda", dtype=dtype, enabled=device.type == "cuda"
    ):
        generation = model.generate(**batch, **generation_kwargs)

    sequences = getattr(generation, "sequences", None)
    generation_cache = getattr(generation, "past_key_values", None)
    if sequences is None or generation_cache is None:
        raise RuntimeError("generate() did not return sequences/past_key_values")

    new_ids = sequences[0, prompt_token_length:]
    content_ids, ended = content_token_ids(processor, model, new_ids)
    if not content_ids:
        raise RuntimeError(f"Empty generated {task} trace: id={row['id']}")
    if not ended and not allow_truncated:
        raise RuntimeError(
            f"{task} generation did not terminate within {max_new_tokens} tokens: "
            f"id={row['id']}"
        )

    text = processor.tokenizer.decode(
        content_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not text:
        raise RuntimeError(f"Decoded {task} trace is empty: id={row['id']}")

    gen_key, gen_value = get_last_layer_kv(generation_cache)
    generation_cache_length = int(gen_key.shape[-2])
    if int(gen_key.shape[1]) != hkv or int(gen_key.shape[-1]) != head_dim:
        raise RuntimeError("Prompt/generation K/V geometry mismatch")

    # Same extraction rule as the RAW-KV DiT training script.
    desired_end = prompt_cache_length + len(content_ids)
    usable_end = min(generation_cache_length, desired_end)
    missing = max(0, desired_end - generation_cache_length)
    if ended and missing > 0:
        raise RuntimeError(
            f"AR cache does not contain all {task} tokens: id={row['id']} missing={missing}"
        )
    if missing > 1:
        raise RuntimeError(
            f"AR cache is too short for {task}: id={row['id']} missing={missing}"
        )

    prefix_len = min(prompt_cache_length, generation_cache_length)
    if prefix_len <= 0:
        raise RuntimeError("Invalid zero-length prompt KV")

    gen_prefix_key = gen_key[0, :, :prefix_len, :].detach().float().cpu()
    gen_prefix_value = gen_value[0, :, :prefix_len, :].detach().float().cpu()
    key_diff = (gen_prefix_key - direct_key[:, :prefix_len, :].float()).abs().max().item()
    val_diff = (gen_prefix_value - direct_value[:, :prefix_len, :].float()).abs().max().item()
    prefix_diff = float(max(key_diff, val_diff))
    if prefix_diff > max_prefix_diff:
        raise RuntimeError(
            f"Prompt prefill/generate prefix mismatch too large: id={row['id']} "
            f"task={task} diff={prefix_diff:.6g} > {max_prefix_diff:.6g}"
        )

    delta_key = (
        gen_key[0, :, prompt_cache_length:usable_end, :]
        .detach().to(dtype=CACHE_DTYPE).cpu().contiguous()
    )
    delta_value = (
        gen_value[0, :, prompt_cache_length:usable_end, :]
        .detach().to(dtype=CACHE_DTYPE).cpu().contiguous()
    )
    if delta_key.ndim != 3 or delta_key.shape != delta_value.shape:
        raise RuntimeError(f"Bad {task} delta K/V shape")
    if int(delta_key.shape[1]) <= 0:
        raise RuntimeError(f"No cached {task} generated tokens: id={row['id']}")

    return {
        "task": task,
        "prompt_text": prompt_text,
        "direct_key": direct_key,
        "direct_value": direct_value,
        "delta_key": delta_key,
        "delta_value": delta_value,
        "generated_text": text,
        "generated_token_ids": [int(x) for x in content_ids],
        "prompt_token_length": prompt_token_length,
        "prompt_cache_length": prompt_cache_length,
        "generated_tokens": len(content_ids),
        "cached_generated_tokens": int(delta_key.shape[1]),
        "generation_cache_length": generation_cache_length,
        "generation_ended": bool(ended),
        "missing_generated_cache_tokens": int(missing),
        "prefix_max_abs_diff": prefix_diff,
        "vlm_num_attention_heads": hq,
        "vlm_num_kv_heads": hkv,
        "vlm_head_dim": head_dim,
    }


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract controlled trajectory/reasoning RAW-KV caches for 2x2 ablation."
    )
    p.add_argument("--vlm", type=Path, default=VLM_PATH)
    p.add_argument("--jsonl", type=Path, default=TEST_JSONL)
    p.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    p.add_argument("--reasoning-max-new-tokens", type=int, default=128)
    p.add_argument("--trajectory-max-new-tokens", type=int, default=128)
    p.add_argument("--allow-truncated", action="store_true")
    p.add_argument("--max-prefix-diff", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit", type=int, default=None, help="Debug only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.vlm = args.vlm.expanduser().resolve()
    args.jsonl = args.jsonl.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.vlm.is_dir():
        raise FileNotFoundError(args.vlm)
    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")

    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists: {args.output}\nUse --overwrite")
        shutil.rmtree(args.output)
    cache_dir = args.output / "test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(r) for r in read_jsonl(args.jsonl)]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError("No rows")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate test IDs")

    set_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = resolve_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 120)
    print("CONTROLLED TRACE RAW-KV ABLATION CACHE")
    print("=" * 120)
    print("GPU       :", torch.cuda.get_device_name(args.gpu_id))
    print("VLM       :", args.vlm)
    print("Dataset   :", args.jsonl)
    print("Samples   :", len(rows))
    print("Output    :", args.output)
    print("Attention :", args.attn)
    print("Precision :", args.dtype)
    print("Control   : same images/mission/speed/accel/heading; only task instruction changes")

    model, processor = load_target_model(args.vlm, device, dtype, args.attn)

    manifest: List[Dict[str, Any]] = []
    all_start = time.perf_counter()

    for index, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        fingerprint_before = row_fingerprint(row)

        reasoning = extract_task_cache(
            model=model,
            processor=processor,
            row=row,
            task="reasoning",
            device=device,
            dtype=dtype,
            max_new_tokens=args.reasoning_max_new_tokens,
            allow_truncated=args.allow_truncated,
            max_prefix_diff=args.max_prefix_diff,
        )

        # Re-check controlled inputs before second condition.
        if row_fingerprint(row) != fingerprint_before:
            raise RuntimeError(f"Controlled input mutated after reasoning: id={row['id']}")

        trajectory = extract_task_cache(
            model=model,
            processor=processor,
            row=row,
            task="trajectory",
            device=device,
            dtype=dtype,
            max_new_tokens=args.trajectory_max_new_tokens,
            allow_truncated=args.allow_truncated,
            max_prefix_diff=args.max_prefix_diff,
        )

        if row_fingerprint(row) != fingerprint_before:
            raise RuntimeError(f"Controlled input mutated after trajectory: id={row['id']}")

        # Geometry must be identical; token length may differ because the task instruction differs.
        geom_r = (
            reasoning["vlm_num_attention_heads"],
            reasoning["vlm_num_kv_heads"],
            reasoning["vlm_head_dim"],
        )
        geom_t = (
            trajectory["vlm_num_attention_heads"],
            trajectory["vlm_num_kv_heads"],
            trajectory["vlm_head_dim"],
        )
        if geom_r != geom_t:
            raise RuntimeError(f"Reasoning/trajectory KV geometry mismatch: id={row['id']}")

        out_path = cache_dir / cache_filename(index, row["id"])
        payload = {
            "cache_version": CACHE_VERSION,
            "id": row["id"],
            "clip": str(row.get("clip", "")),
            "input_fingerprint": fingerprint_before,
            "images": list(row["images"]),
            "command": row["command"],
            "speed_mps": float(row["speed_mps"]),
            "acceleration_mps2": float(row["acceleration_mps2"]),
            "heading_rad": float(row["heading_rad"]),
            "trajectory_gt": torch.as_tensor(row["trajectory"], dtype=torch.float32),
            "vlm": str(args.vlm),
            "vlm_num_attention_heads": geom_r[0],
            "vlm_num_kv_heads": geom_r[1],
            "vlm_head_dim": geom_r[2],
            "memory_mode": "raw_last_layer_kv_separate",
            "decode_backend": "target_autoregressive_greedy",
            # Reasoning condition
            "reasoning_direct_key": reasoning["direct_key"],
            "reasoning_direct_value": reasoning["direct_value"],
            "reasoning_delta_key": reasoning["delta_key"],
            "reasoning_delta_value": reasoning["delta_value"],
            "reasoning_text": reasoning["generated_text"],
            "reasoning_token_ids": reasoning["generated_token_ids"],
            "reasoning_prompt_text": reasoning["prompt_text"],
            "reasoning_prompt_cache_length": reasoning["prompt_cache_length"],
            "reasoning_cached_tokens": reasoning["cached_generated_tokens"],
            "reasoning_prefix_max_abs_diff": reasoning["prefix_max_abs_diff"],
            # Trajectory condition
            "trajectory_direct_key": trajectory["direct_key"],
            "trajectory_direct_value": trajectory["direct_value"],
            "trajectory_delta_key": trajectory["delta_key"],
            "trajectory_delta_value": trajectory["delta_value"],
            "trajectory_text": trajectory["generated_text"],
            "trajectory_token_ids": trajectory["generated_token_ids"],
            "trajectory_prompt_text": trajectory["prompt_text"],
            "trajectory_prompt_cache_length": trajectory["prompt_cache_length"],
            "trajectory_cached_tokens": trajectory["cached_generated_tokens"],
            "trajectory_prefix_max_abs_diff": trajectory["prefix_max_abs_diff"],
        }
        torch.save(payload, out_path)

        manifest.append({
            "id": row["id"],
            "clip": str(row.get("clip", "")),
            "cache_file": str(out_path),
            "input_fingerprint": fingerprint_before,
            "reasoning_prompt_tokens": int(reasoning["prompt_cache_length"]),
            "reasoning_trace_tokens": int(reasoning["cached_generated_tokens"]),
            "trajectory_prompt_tokens": int(trajectory["prompt_cache_length"]),
            "trajectory_trace_tokens": int(trajectory["cached_generated_tokens"]),
            "reasoning_text": reasoning["generated_text"],
            "trajectory_text": trajectory["generated_text"],
        })

        sample_sec = time.perf_counter() - t0
        elapsed = time.perf_counter() - all_start
        avg = elapsed / index
        eta = avg * (len(rows) - index)
        print(
            f"[{index:04d}/{len(rows):04d}] id={row['id']} | "
            f"R prompt/trace={reasoning['prompt_cache_length']}/{reasoning['cached_generated_tokens']} | "
            f"T prompt/trace={trajectory['prompt_cache_length']}/{trajectory['cached_generated_tokens']} | "
            f"sample={sample_sec:.2f}s elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
            flush=True,
        )

        del reasoning, trajectory, payload
        if index % 25 == 0:
            cleanup_cuda()

    write_jsonl(cache_dir / "manifest.jsonl", manifest)
    meta = {
        "cache_version": CACHE_VERSION,
        "vlm": str(args.vlm),
        "vlm_dir_signature": model_dir_signature(args.vlm),
        "jsonl": str(args.jsonl),
        "jsonl_sha256": sha256_file(args.jsonl),
        "num_samples": len(manifest),
        "attn_implementation": args.attn,
        "dtype": args.dtype,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "decode": "greedy, do_sample=False",
        "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
        "trajectory_max_new_tokens": args.trajectory_max_new_tokens,
        "allow_truncated": args.allow_truncated,
        "control": {
            "same_images": True,
            "same_mission_command": True,
            "same_speed": True,
            "same_acceleration": True,
            "same_heading": True,
            "same_processor": True,
            "same_vlm": True,
            "only_final_task_instruction_differs": True,
        },
        "conditions": {
            "direct_reasoning": "reasoning prompt RAW K/V only",
            "reuse_reasoning": "reasoning prompt RAW K/V + generated reasoning RAW K/V",
            "direct_trajectory": "trajectory prompt RAW K/V only",
            "reuse_trajectory": "trajectory prompt RAW K/V + generated trajectory RAW K/V",
        },
    }
    (args.output / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 120)
    print("CACHE EXTRACTION COMPLETE")
    print("=" * 120)
    print("Samples  :", len(manifest))
    print("Manifest :", cache_dir / "manifest.jsonl")
    print("Meta     :", args.output / "meta.json")

    del model, processor
    cleanup_cuda()


if __name__ == "__main__":
    main()
