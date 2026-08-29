#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Reasoning_VLM_1200 reasoning test.

This script uses EXACTLY the same VLM input construction as
final_train_all.py:

    3 camera images:
        front_left
        front
        front_right

    + mission_command only

No DFlash.
No Action Expert.
No KV cache extraction.

It simply runs Reasoning_VLM_1200 autoregressively and prints
the generated reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)


# =============================================================================
# CONFIG
# =============================================================================

VLM_PATH = Path(
    "/home/lhh/lab/models/vlm/Reasoning_VLM_1200"
)

DEFAULT_JSONL = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert8/part1/val.jsonl"
)

CAMERAS = (
    "front_left",
    "front",
    "front_right",
)

MIN_PIXELS = 200_704
MAX_PIXELS = 200_704

ATTN_IMPLEMENTATION = "eager"
MAX_NEW_TOKENS = 128


# =============================================================================
# DATA
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"JSON parse failed: {path}:{line_no}"
                ) from exc

    if not rows:
        raise RuntimeError(f"No samples found: {path}")

    return rows


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
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
        raise ValueError(
            f"Missing mission command id={sid}"
        )

    row["mission_command"] = command
    row["command"] = command

    images = row.get("images")

    if isinstance(images, dict):
        for cam in CAMERAS:
            if not images.get(cam):
                raise ValueError(
                    f"Missing image '{cam}' id={sid}"
                )

            path = Path(str(images[cam]))

            if not path.is_file():
                raise FileNotFoundError(
                    f"id={sid} cam={cam}: {path}"
                )

    elif isinstance(images, list) and len(images) == 3:
        for path_str in images:
            path = Path(str(path_str))

            if not path.is_file():
                raise FileNotFoundError(
                    f"id={sid}: {path}"
                )

    else:
        raise ValueError(
            f"Expected exactly 3 images id={sid}: {images}"
        )

    return row


def row_image_path(
    row: Dict[str, Any],
    cam: str,
) -> str:

    images = row["images"]

    if isinstance(images, dict):
        value = images.get(cam)

    elif isinstance(images, list) and len(images) == 3:
        value = images[CAMERAS.index(cam)]

    else:
        value = None

    if not value:
        raise KeyError(
            f"Missing image {cam}: id={row['id']}"
        )

    path = Path(str(value))

    if not path.is_file():
        raise FileNotFoundError(path)

    return str(path)


# =============================================================================
# EXACT SAME PROMPT AS TRAINING CACHE SCRIPT
# =============================================================================

def make_reasoning1200_messages(
    row: Dict[str, Any],
) -> List[Dict[str, Any]]:

    command = str(
        row.get("mission_command")
        or row.get("command")
        or ""
    ).strip()

    if not command:
        raise ValueError(
            f"Empty mission command: id={row.get('id')}"
        )

    content: List[Dict[str, Any]] = []

    for cam in CAMERAS:
        content.append(
            {
                "type": "image",
                "path": row_image_path(row, cam),
            }
        )

    content.append(
        {
            "type": "text",
            "text": command,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def build_prompt(
    processor,
    row: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
):

    messages = make_reasoning1200_messages(row)

    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    batch = dict(batch)

    # Same as training script.
    batch.pop("token_type_ids", None)

    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(
                device,
                non_blocking=True,
            )

            if value.is_floating_point():
                value = value.to(dtype=dtype)

        moved[key] = value

    return messages, moved


# =============================================================================
# MODEL
# =============================================================================

def load_model(
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
        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                str(path),
                dtype=dtype,
                **kwargs,
            )
        )

    except TypeError:
        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                str(path),
                torch_dtype=dtype,
                **kwargs,
            )
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


# =============================================================================
# GENERATION
# =============================================================================

def normalize_token_id_set(value):
    if value is None:
        return set()

    if isinstance(value, int):
        return {int(value)}

    if isinstance(value, (list, tuple, set)):
        return {
            int(x)
            for x in value
            if x is not None
        }

    return {int(value)}


def extract_content_ids(
    processor,
    model,
    new_ids: torch.Tensor,
):

    eos_ids = set()

    eos_ids |= normalize_token_id_set(
        getattr(
            processor.tokenizer,
            "eos_token_id",
            None,
        )
    )

    eos_ids |= normalize_token_id_set(
        getattr(
            getattr(
                model,
                "generation_config",
                None,
            ),
            "eos_token_id",
            None,
        )
    )

    for token_text in (
        "<|im_end|>",
        "<|endoftext|>",
    ):
        token_id = (
            processor.tokenizer
            .convert_tokens_to_ids(token_text)
        )

        if (
            token_id is not None
            and token_id
            != getattr(
                processor.tokenizer,
                "unk_token_id",
                None,
            )
            and int(token_id) >= 0
        ):
            eos_ids.add(int(token_id))

    pad_ids = set()

    pad_ids |= normalize_token_id_set(
        getattr(
            processor.tokenizer,
            "pad_token_id",
            None,
        )
    )

    content_ids = []
    ended = False

    for token in new_ids.tolist():
        token = int(token)

        if token in eos_ids:
            ended = True
            break

        if token in pad_ids and content_ids:
            ended = True
            break

        content_ids.append(token)

    return content_ids, ended


@torch.inference_mode()
def run_one(
    model,
    processor,
    row,
    device,
    dtype,
    max_new_tokens,
):

    messages, batch = build_prompt(
        processor=processor,
        row=row,
        device=device,
        dtype=dtype,
    )

    prompt_token_length = int(
        batch["input_ids"].shape[1]
    )

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    pad_token_id = getattr(
        processor.tokenizer,
        "pad_token_id",
        None,
    )

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(
            pad_token_id
        )

    torch.cuda.synchronize()
    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        generation = model.generate(
            **batch,
            **generation_kwargs,
        )

    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)

    sequences = generation.sequences

    new_ids = sequences[
        0,
        prompt_token_length:
    ]

    content_ids, ended = extract_content_ids(
        processor,
        model,
        new_ids,
    )

    reasoning_text = (
        processor.tokenizer.decode(
            content_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        .strip()
    )

    return {
        "messages": messages,
        "prompt_tokens": prompt_token_length,
        "generated_tokens": len(content_ids),
        "ended_eos": ended,
        "reasoning_text": reasoning_text,
        "generation_ms": float(elapsed_ms),
        "token_ids": content_ids,
    }


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
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
    )

    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based sample index",
    )

    p.add_argument(
        "--id",
        type=str,
        default=None,
        help="Sample ID. Overrides --index.",
    )

    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    p.add_argument(
        "--show-token-ids",
        action="store_true",
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

    dtype = torch.float16

    rows = [
        normalize_row(x)
        for x in read_jsonl(args.jsonl)
    ]

    if args.id is not None:

        matches = [
            row
            for row in rows
            if row["id"] == args.id
        ]

        if not matches:
            raise KeyError(
                f"Sample id not found: {args.id}"
            )

        row = matches[0]

    else:
        if (
            args.index < 0
            or args.index >= len(rows)
        ):
            raise IndexError(
                f"--index must be 0 ~ {len(rows)-1}"
            )

        row = rows[args.index]

    print("=" * 120)
    print("Reasoning_VLM_1200 | DIRECT REASONING TEST")
    print("=" * 120)

    print("GPU             :", torch.cuda.get_device_name(args.gpu_id))
    print("VLM             :", args.vlm)
    print("Dataset         :", args.jsonl)
    print("Sample ID       :", row["id"])
    print("Attention       :", ATTN_IMPLEMENTATION)
    print("Precision       : FP16")
    print("Max new tokens  :", args.max_new_tokens)

    print()
    print("[INPUT]")
    print("front_left      :", row_image_path(row, "front_left"))
    print("front           :", row_image_path(row, "front"))
    print("front_right     :", row_image_path(row, "front_right"))
    print("mission_command :", row["mission_command"])

    print()
    print("[LOAD MODEL]")

    model, processor = load_model(
        path=args.vlm,
        device=device,
        dtype=dtype,
    )

    result = run_one(
        model=model,
        processor=processor,
        row=row,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
    )

    print()
    print("=" * 120)
    print("GENERATED REASONING")
    print("=" * 120)
    print(result["reasoning_text"])

    print()
    print("=" * 120)
    print("GENERATION INFO")
    print("=" * 120)
    print("Prompt tokens    :", result["prompt_tokens"])
    print("Generated tokens :", result["generated_tokens"])
    print("EOS terminated   :", result["ended_eos"])
    print("Generation time  :", f"{result['generation_ms']:.2f} ms")

    if args.show_token_ids:
        print("Token IDs        :", result["token_ids"])


if __name__ == "__main__":
    main()