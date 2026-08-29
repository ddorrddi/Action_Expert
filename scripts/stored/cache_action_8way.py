#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build generation-time KV caches from the merged Reasoning_VLM.

This script intentionally does NOT decode -> retokenize -> full Qwen re-forward.

For each sample it stores only:
  direct_kv          : last Qwen text layer K/V at the prompt boundary
  reasoning_delta_kv : additional K/V created while generating Reasoning

Thus:
  Direct memory    = direct_kv
  Reasoning memory = concat(direct_kv, reasoning_delta_kv)

The VLM is frozen and the Action Expert later trains only from these cached features.
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


VLM_MODEL_PATH = Path("/home/lhh/lab/models/vlm/Reasoning_VLM")
DATASET_DIR = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpert8/part1")
CACHE_ROOT = Path("/home/lhh/lab/Action_Expert/dataset/ActionExpert8/action_kv_cache")

MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
MAX_SEQ_LEN = 2048
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_DTYPE = "bf16"

CACHE_VERSION = "action8_generation_kv_v1"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def file_signature(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sample_cache_name(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:07d}_{digest}.pt"


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


def build_prompt(row: Dict[str, Any]) -> str:
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


def kv_to_sequence(
    key: torch.Tensor,
    value: torch.Tensor,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Expected K/V [B,H,T,D], got K={tuple(key.shape)}, V={tuple(value.shape)}"
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


def reasoning_content_ids(processor, model, new_ids: torch.Tensor) -> Tuple[List[int], bool]:
    eos_ids = set()
    eos_ids |= normalize_token_id_set(getattr(processor.tokenizer, "eos_token_id", None))
    eos_ids |= normalize_token_id_set(getattr(model.generation_config, "eos_token_id", None))

    pad_ids = set()
    pad_ids |= normalize_token_id_set(getattr(processor.tokenizer, "pad_token_id", None))
    pad_ids |= normalize_token_id_set(getattr(model.generation_config, "pad_token_id", None))

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
    batch = build_prompt_batch(processor, row, device, dtype)
    prompt_token_length = int(batch["input_ids"].shape[1])

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

    direct_key, direct_value = get_last_layer_kv(prompt_out.past_key_values)
    prompt_cache_length = int(direct_key.shape[-2])
    direct_kv = (
        kv_to_sequence(direct_key, direct_value)[0]
        .detach()
        .to(dtype=dtype)
        .cpu()
        .contiguous()
    )

    del prompt_out, direct_key, direct_value

    generation_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )
    if getattr(processor.tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = int(processor.tokenizer.pad_token_id)

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        generation = model.generate(
            **batch,
            **generation_kwargs,
        )

    generation_cache = getattr(generation, "past_key_values", None)
    if generation_cache is None:
        raise RuntimeError(
            "generate() did not return past_key_values; requires use_cache=True and return_dict_in_generate=True."
        )

    sequences = getattr(generation, "sequences", None)
    if sequences is None:
        raise RuntimeError("generate() did not return sequences")

    new_ids = sequences[0, prompt_token_length:]
    reason_ids, ended = reasoning_content_ids(processor, model, new_ids)

    if not reason_ids:
        raise RuntimeError("Generated reasoning is empty")
    if not ended and not allow_truncated:
        raise RuntimeError(
            f"Reasoning did not terminate within max_new_tokens={max_new_tokens}; "
            "increase --max-new-tokens or pass --allow-truncated."
        )

    reasoning_text = processor.tokenizer.decode(
        reason_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not reasoning_text:
        raise RuntimeError("Decoded reasoning text is empty")

    gen_key, gen_value = get_last_layer_kv(generation_cache)
    generation_cache_length = int(gen_key.shape[-2])

    desired_reasoning_end = prompt_cache_length + len(reason_ids)
    usable_reasoning_end = min(generation_cache_length, desired_reasoning_end)
    missing = max(0, desired_reasoning_end - generation_cache_length)

    if ended and missing > 0:
        raise RuntimeError(
            "Returned generation KV does not contain the full reasoning content: "
            f"missing={missing}, prompt_cache={prompt_cache_length}, "
            f"reasoning_tokens={len(reason_ids)}, generation_cache={generation_cache_length}"
        )

    if missing > 1:
        raise RuntimeError(
            f"Generation cache is too short by {missing} tokens; refusing inconsistent sample."
        )

    generation_seq = kv_to_sequence(
        gen_key,
        gen_value,
        max_length=usable_reasoning_end,
    )[0].detach()

    prefix_len = min(prompt_cache_length, int(generation_seq.shape[0]))
    if prefix_len <= 0:
        raise RuntimeError("Invalid zero-length prompt KV")

    prefix_diff = float(
        (
            generation_seq[:prefix_len].float().cpu()
            - direct_kv[:prefix_len].float()
        )
        .abs()
        .max()
        .item()
    )

    delta_start = prompt_cache_length
    delta_end = usable_reasoning_end
    reasoning_delta = generation_seq[delta_start:delta_end]

    if reasoning_delta.shape[0] <= 0:
        raise RuntimeError(
            "No reasoning KV tokens were cached. If generation hit the limit, increase --max-new-tokens."
        )

    reasoning_delta_kv = (
        reasoning_delta.to(dtype=dtype).cpu().contiguous()
    )

    trajectory = torch.as_tensor(row["trajectory"], dtype=torch.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] < 10 or trajectory.shape[1] < 3:
        raise ValueError(f"Bad trajectory shape: {tuple(trajectory.shape)}")
    trajectory = trajectory[:10, :3].contiguous()

    result = {
        "cache_version": CACHE_VERSION,
        "id": str(row.get("id", "")),
        "clip": str(row.get("clip", "")),
        "direct_kv": direct_kv,
        "reasoning_delta_kv": reasoning_delta_kv,
        "trajectory": trajectory,
        "prompt_token_length": int(prompt_token_length),
        "prompt_cache_length": int(prompt_cache_length),
        "reasoning_generated_tokens": int(len(reason_ids)),
        "reasoning_cached_tokens": int(reasoning_delta_kv.shape[0]),
        "generation_cache_length": int(generation_cache_length),
        "generation_ended": bool(ended),
        "missing_reasoning_cache_tokens": int(missing),
        "prefix_max_abs_diff": float(prefix_diff),
        "reasoning_text": reasoning_text,
        "gt_reasoning": str(row.get("gt_reasoning", "")),
        "speed_mps": float(row["speed_mps"]),
        "command": str(row.get("command", "UNKNOWN")),
    }

    del generation, generation_cache, gen_key, gen_value, generation_seq, batch
    return result


def build_split(
    split_name: str,
    rows: Sequence[Dict[str, Any]],
    split_dir: Path,
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    allow_truncated: bool,
) -> List[Dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    skipped = 0

    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("id", f"row_{index}"))

        if row.get("speed_mps") is None:
            print(
                f"[cache:{split_name}] {index:06d}/{len(rows):06d} "
                f"SKIP id={sample_id} reason=missing_speed",
                flush=True,
            )
            skipped += 1
            continue

        cache_path = split_dir / sample_cache_name(index, sample_id)
        t0 = time.perf_counter()
        try:
            record = extract_one(
                model=model,
                processor=processor,
                row=row,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                allow_truncated=allow_truncated,
            )
            torch.save(record, cache_path)

            elapsed = time.perf_counter() - t0
            manifest.append(
                {
                    "id": sample_id,
                    "clip": str(row.get("clip", "")),
                    "cache_file": str(cache_path),
                    "prompt_cache_length": int(record["prompt_cache_length"]),
                    "reasoning_cached_tokens": int(record["reasoning_cached_tokens"]),
                    "kv_dim": int(record["direct_kv"].shape[-1]),
                    "prefix_max_abs_diff": float(record["prefix_max_abs_diff"]),
                }
            )
            print(
                f"[cache:{split_name}] {index:06d}/{len(rows):06d} "
                f"id={sample_id} promptKV={record['prompt_cache_length']} "
                f"reasonKV={record['reasoning_cached_tokens']} "
                f"dim={record['direct_kv'].shape[-1]} "
                f"prefix_diff={record['prefix_max_abs_diff']:.3e} "
                f"{elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            skipped += 1
            print(
                f"[cache:{split_name}] {index:06d}/{len(rows):06d} "
                f"SKIP id={sample_id} reason={type(exc).__name__}: {exc}",
                flush=True,
            )

        if index % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not manifest:
        raise RuntimeError(f"No usable cache records for split={split_name}")

    write_jsonl(split_dir / "manifest.jsonl", manifest)
    print(
        f"[cache:{split_name}] complete: kept={len(manifest)} skipped={skipped} dir={split_dir}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=VLM_MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default=DEFAULT_DTYPE)
    parser.add_argument("--attn", default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--allow-truncated", action="store_true")
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
            "Selected GPU does not support BF16. Re-run with --dtype fp16 "
            "(e.g. for Quadro P5000)."
        )

    for path in (args.model, args.dataset_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    args.cache_root.mkdir(parents=True, exist_ok=True)

    model_sig = directory_signature(args.model)

    print("=" * 100)
    print("BUILD ACTION EXPERT 8-WAY GENERATION-TIME KV CACHE")
    print("=" * 100)
    print("GPU        :", torch.cuda.get_device_name(args.gpu_id))
    print("Model      :", args.model)
    print("Dataset    :", args.dataset_dir)
    print("Cache root :", args.cache_root)
    print("Splits     :", ", ".join(args.splits))
    print("Precision  :", args.dtype)
    print("Attention  :", args.attn)
    print("Pixels     :", MIN_PIXELS)
    print("Max new    :", args.max_new_tokens)
    print("KV source  : generate().past_key_values (no replay)")
    print()

    processor = load_processor(args.model)
    model = load_model(args.model, device, dtype, args.attn)

    split_summaries = {}
    for split_name in args.splits:
        dataset_path = args.dataset_dir / f"{split_name}.jsonl"
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        rows = read_jsonl(dataset_path)
        split_dir = args.cache_root / split_name

        split_meta_path = split_dir / "meta.json"
        expected_meta = {
            "cache_version": CACHE_VERSION,
            "model_path": str(args.model),
            "model_signature": model_sig,
            "dataset_path": str(dataset_path),
            "dataset_signature": file_signature(dataset_path),
            "dtype": args.dtype,
            "attn_implementation": args.attn,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "max_new_tokens": int(args.max_new_tokens),
        }

        if split_dir.exists():
            if args.rebuild:
                shutil.rmtree(split_dir)
            else:
                if split_meta_path.is_file() and (split_dir / "manifest.jsonl").is_file():
                    existing_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
                    if existing_meta == expected_meta:
                        print(f"[cache:{split_name}] valid cache already exists -> reuse")
                        split_summaries[split_name] = {
                            "reused": True,
                            "records": len(read_jsonl(split_dir / "manifest.jsonl")),
                        }
                        continue
                raise RuntimeError(
                    f"Existing incompatible cache: {split_dir}\n"
                    "Use --rebuild for a clean rebuild."
                )

        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_split(
            split_name=split_name,
            rows=rows,
            split_dir=split_dir,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            allow_truncated=args.allow_truncated,
        )
        split_meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        split_summaries[split_name] = {
            "reused": False,
            "records": len(manifest),
        }

    root_meta = {
        "cache_version": CACHE_VERSION,
        "model_path": str(args.model),
        "model_signature": model_sig,
        "splits": split_summaries,
        "representation": {
            "direct": "last Qwen text-layer full K/V at prompt boundary",
            "reasoning": "direct K/V + generation-time reasoning K/V delta",
            "kv_sequence_format": "[T, concat(flatten(K_heads), flatten(V_heads))]",
        },
    }
    (args.cache_root / "meta.json").write_text(
        json.dumps(root_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDONE")
    print("Cache root:", args.cache_root)


if __name__ == "__main__":
    main()
