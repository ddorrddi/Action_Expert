#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the TRAJECTORY-ONLY observation KV cache for the Alpamayo-style ablation.

This script is intentionally paired to the already-built Reasoning_VLM cache:
  /home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache

For each split, it caches ONLY the sample IDs that already exist in the reasoning
cache manifest. Therefore the two branches are trained/evaluated on exactly the
same samples.

TRAJ_ONLY branch:
    VLM_Baseline + [3 images, speed, route command]
        -> prompt-boundary last-layer Qwen K/V
        -> Action Expert
        -> trajectory

Important:
  - No reasoning generation is performed here.
  - Prompt text is kept identical to the old reasoning-cache script so the
    observation-side token sequence is matched as closely as possible.
  - The baseline VLM is frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# PATHS / CONFIG
# =============================================================================

BASELINE_VLM_PATH = Path("/home/lhh/lab/models/vlm/VLM_Baseline")
DATASET_DIR = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpert8/part1")
REASONING_CACHE_ROOT = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache")
OUTPUT_CACHE_ROOT = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpertAlpamayo/traj_only_kv_cache")

MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
MAX_SEQ_LEN = 2048
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_DTYPE = "bf16"

CACHE_VERSION = "alpamayo_ablation_traj_only_prompt_kv_v1"


# =============================================================================
# IO / SIGNATURE
# =============================================================================

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


def directory_signature(path: Path) -> str:
    pieces = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = file_path.stat()
        pieces.append(
            (
                str(file_path.relative_to(path)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        )
    payload = json.dumps(pieces, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_cache_name(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:07d}_{digest}.pt"


# =============================================================================
# MODEL / PROCESSOR
# =============================================================================

def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("--dtype must be bf16 or fp16")


def load_model(path: Path, device: torch.device, dtype: torch.dtype, attn_impl: str):
    kwargs = dict(
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(path), dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(path), torch_dtype=dtype, **kwargs
        )
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_processor(path: Path):
    return AutoProcessor.from_pretrained(
        str(path),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )


# =============================================================================
# PROMPT
# =============================================================================

def build_prompt(row: Dict[str, Any]) -> str:
    """Must stay identical to the old cache_action_kv.py prompt."""
    speed = row.get("speed_mps", None)
    if speed is None:
        raise ValueError("speed_mps is missing")

    command = row.get("command", "UNKNOWN")
    return (
        "You are the driving assistant of an autonomous vehicle. "
        "The three images are the vehicle's front-left, front, and front-right camera views.\n"
        f"Current speed: {float(speed):.1f} m/s. Route command: {command}.\n"
        "Generate the driving reasoning only."
    )


def open_three_images(row: Dict[str, Any]) -> List[Image.Image]:
    paths = row.get("images")
    if not isinstance(paths, list) or len(paths) != 3:
        raise ValueError(f"Exactly 3 image paths required: id={row.get('id')}")

    images: List[Image.Image] = []
    for value in paths:
        p = Path(str(value))
        if not p.is_file():
            raise FileNotFoundError(p)
        with Image.open(p) as img:
            images.append(img.convert("RGB").copy())
    return images


def build_prompt_batch(
    processor,
    row: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    prompt = build_prompt(row)
    images = open_three_images(row)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    batch.pop("token_type_ids", None)

    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        value = value.to(device)
        if value.is_floating_point():
            value = value.to(dtype)
        moved[key] = value
    return moved


# =============================================================================
# KV ACCESS
# =============================================================================

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

    raise RuntimeError(f"Unsupported cache type: {type(past_key_values)}")


def kv_to_sequence(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """[B,H,T,D] K/V -> [B,T,2*H*D]."""
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected K/V [B,H,T,D], got K={tuple(key.shape)}, V={tuple(value.shape)}"
        )
    key_seq = key.permute(0, 2, 1, 3).contiguous().flatten(2)
    value_seq = value.permute(0, 2, 1, 3).contiguous().flatten(2)
    return torch.cat([key_seq, value_seq], dim=-1)


# =============================================================================
# ONE SAMPLE
# =============================================================================

@torch.inference_mode()
def extract_one(
    model,
    processor,
    row: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    batch = build_prompt_batch(processor, row, device, dtype)
    prompt_token_length = int(batch["input_ids"].shape[1])

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        out = model(
            **batch,
            use_cache=True,
            return_dict=True,
        )

    key, value = get_last_layer_kv(out.past_key_values)
    prompt_cache_length = int(key.shape[-2])
    observation_kv = (
        kv_to_sequence(key, value)[0]
        .detach()
        .to(dtype=dtype)
        .cpu()
        .contiguous()
    )

    trajectory = torch.as_tensor(row["trajectory"], dtype=torch.float32)
    if tuple(trajectory.shape) != (10, 3):
        raise ValueError(f"Bad trajectory shape: {tuple(trajectory.shape)}")

    result = {
        "cache_version": CACHE_VERSION,
        "id": str(row.get("id", "")),
        "clip": str(row.get("clip", "")),
        "observation_kv": observation_kv,
        "trajectory": trajectory.contiguous(),
        "prompt_token_length": int(prompt_token_length),
        "prompt_cache_length": int(prompt_cache_length),
        "speed_mps": float(row["speed_mps"]),
        "command": str(row.get("command", "UNKNOWN")),
    }

    del out, key, value, batch
    return result


# =============================================================================
# PAIRED SPLIT BUILD
# =============================================================================

def load_paired_rows(
    dataset_path: Path,
    reasoning_manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_rows = read_jsonl(dataset_path)
    reasoning_manifest = read_jsonl(reasoning_manifest_path)

    by_id = {str(row["id"]): row for row in dataset_rows}
    paired_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for entry in reasoning_manifest:
        sample_id = str(entry["id"])
        row = by_id.get(sample_id)
        if row is None:
            missing.append(sample_id)
            continue
        paired_rows.append(row)

    if missing:
        raise RuntimeError(
            f"{len(missing)} reasoning-cache IDs are missing from {dataset_path}. "
            f"Examples: {missing[:5]}"
        )
    if len(paired_rows) != len(reasoning_manifest):
        raise RuntimeError("Pairing count mismatch")

    return paired_rows, reasoning_manifest


def build_split(
    split_name: str,
    rows: Sequence[Dict[str, Any]],
    reasoning_manifest: Sequence[Dict[str, Any]],
    split_dir: Path,
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    if len(rows) != len(reasoning_manifest):
        raise ValueError("rows/reasoning_manifest length mismatch")

    split_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    for index, (row, reason_entry) in enumerate(zip(rows, reasoning_manifest), 1):
        sample_id = str(row["id"])
        if sample_id != str(reason_entry["id"]):
            raise RuntimeError(
                f"Paired order mismatch at index={index}: {sample_id} vs {reason_entry['id']}"
            )

        cache_path = split_dir / sample_cache_name(index, sample_id)
        t0 = time.perf_counter()
        record = extract_one(
            model=model,
            processor=processor,
            row=row,
            device=device,
            dtype=dtype,
        )
        torch.save(record, cache_path)

        # Ensure the baseline VLM and Reasoning_VLM expose compatible KV dimensions.
        reason_cache = torch.load(
            reason_entry["cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        reason_direct = reason_cache["direct_kv"]
        if int(reason_direct.shape[-1]) != int(record["observation_kv"].shape[-1]):
            raise RuntimeError(
                f"KV dimension mismatch for id={sample_id}: "
                f"baseline={record['observation_kv'].shape[-1]} "
                f"reasoning={reason_direct.shape[-1]}"
            )
        reason_traj = reason_cache["trajectory"].float()
        if not torch.equal(reason_traj, record["trajectory"]):
            max_diff = float((reason_traj - record["trajectory"]).abs().max().item())
            if max_diff > 1e-6:
                raise RuntimeError(
                    f"Trajectory target mismatch for id={sample_id}: max_diff={max_diff}"
                )

        elapsed = time.perf_counter() - t0
        manifest.append(
            {
                "id": sample_id,
                "clip": str(row.get("clip", "")),
                "cache_file": str(cache_path),
                "paired_reasoning_cache_file": str(reason_entry["cache_file"]),
                "prompt_cache_length": int(record["prompt_cache_length"]),
                "kv_dim": int(record["observation_kv"].shape[-1]),
            }
        )

        print(
            f"[traj-only:{split_name}] {index:06d}/{len(rows):06d} "
            f"id={sample_id} promptKV={record['prompt_cache_length']} "
            f"dim={record['observation_kv'].shape[-1]} {elapsed:.2f}s",
            flush=True,
        )

        del reason_cache, reason_direct, reason_traj
        if index % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    write_jsonl(split_dir / "manifest.jsonl", manifest)
    return manifest


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=BASELINE_VLM_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--reasoning-cache-root", type=Path, default=REASONING_CACHE_ROOT)
    parser.add_argument("--output-cache-root", type=Path, default=OUTPUT_CACHE_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default=DEFAULT_DTYPE)
    parser.add_argument("--attn", default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = resolve_dtype(args.dtype)

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Selected GPU does not support BF16. Use --dtype fp16 "
            "(required for Quadro P5000)."
        )

    required_paths = [args.model, args.dataset_dir, args.reasoning_cache_root]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_cache_root.mkdir(parents=True, exist_ok=True)
    model_sig = directory_signature(args.model)

    print("=" * 100)
    print("ALPAMAYO-STYLE ABLATION: BUILD TRAJECTORY-ONLY OBSERVATION KV")
    print("=" * 100)
    print("GPU              :", torch.cuda.get_device_name(args.gpu_id))
    print("Baseline VLM     :", args.model)
    print("Dataset          :", args.dataset_dir)
    print("Reasoning cache  :", args.reasoning_cache_root)
    print("Output cache     :", args.output_cache_root)
    print("Splits           :", ", ".join(args.splits))
    print("Precision        :", args.dtype)
    print("Attention        :", args.attn)
    print("Pairing          : exact IDs from existing reasoning cache")
    print("Generation       : NONE")
    print()

    processor = load_processor(args.model)
    model = load_model(args.model, device, dtype, args.attn)

    split_summaries: Dict[str, Any] = {}

    for split_name in args.splits:
        dataset_path = args.dataset_dir / f"{split_name}.jsonl"
        reasoning_manifest_path = args.reasoning_cache_root / split_name / "manifest.jsonl"
        reasoning_meta_path = args.reasoning_cache_root / split_name / "meta.json"
        split_dir = args.output_cache_root / split_name
        split_meta_path = split_dir / "meta.json"

        for p in (dataset_path, reasoning_manifest_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        rows, reasoning_manifest = load_paired_rows(
            dataset_path=dataset_path,
            reasoning_manifest_path=reasoning_manifest_path,
        )

        expected_meta = {
            "cache_version": CACHE_VERSION,
            "model_path": str(args.model),
            "model_signature": model_sig,
            "dataset_path": str(dataset_path),
            "dataset_signature": sha256_file(dataset_path),
            "reasoning_manifest_path": str(reasoning_manifest_path),
            "reasoning_manifest_signature": sha256_file(reasoning_manifest_path),
            "reasoning_meta_signature": (
                sha256_file(reasoning_meta_path) if reasoning_meta_path.is_file() else None
            ),
            "dtype": args.dtype,
            "attn_implementation": args.attn,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "pair_count": len(rows),
            "prompt": "identical to old cache_action_kv.py reasoning prompt",
        }

        if split_dir.exists():
            if args.rebuild:
                shutil.rmtree(split_dir)
            else:
                if split_meta_path.is_file() and (split_dir / "manifest.jsonl").is_file():
                    existing_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
                    if existing_meta == expected_meta:
                        count = len(read_jsonl(split_dir / "manifest.jsonl"))
                        print(f"[traj-only:{split_name}] valid paired cache exists -> reuse ({count})")
                        split_summaries[split_name] = {"reused": True, "records": count}
                        continue
                raise RuntimeError(
                    f"Existing incompatible cache: {split_dir}\n"
                    "Use --rebuild for a clean rebuild."
                )

        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_split(
            split_name=split_name,
            rows=rows,
            reasoning_manifest=reasoning_manifest,
            split_dir=split_dir,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )
        split_meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        split_summaries[split_name] = {"reused": False, "records": len(manifest)}

    root_meta = {
        "cache_version": CACHE_VERSION,
        "model_path": str(args.model),
        "model_signature": model_sig,
        "reasoning_cache_root": str(args.reasoning_cache_root),
        "splits": split_summaries,
        "representation": {
            "traj_only": "VLM_Baseline prompt-boundary last Qwen text-layer K/V",
            "reasoning_branch_reused": "Reasoning_VLM direct_kv + reasoning_delta_kv",
            "kv_sequence_format": "[T, concat(flatten(K_heads), flatten(V_heads))]",
        },
    }
    (args.output_cache_root / "meta.json").write_text(
        json.dumps(root_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDONE")
    print("Trajectory-only cache:", args.output_cache_root)
    print("Reasoning cache reused:", args.reasoning_cache_root)


if __name__ == "__main__":
    main()
