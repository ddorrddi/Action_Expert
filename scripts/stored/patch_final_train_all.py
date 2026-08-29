#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import ast
import re
import shutil

TARGET = Path("/home/lhh/lab/Action_Expert/scripts/final_train_all.py")
BACKUP = TARGET.with_name("final_train_all.before_nodflash.py")

NEW_RUNTIME = '# =============================================================================\n# Reasoning_VLM_1200 TARGET-ONLY PREFILL / AR GENERATION\n# =============================================================================\n\nCAMERAS = ("front_left", "front", "front_right")\n\n\ndef load_target_model(\n    path: Path,\n    device: torch.device,\n    dtype: torch.dtype,\n):\n    # Load ONLY the frozen Reasoning_VLM_1200 target model.\n    # No DFlash model/checkpoint/helper is used.\n    if not path.is_dir():\n        raise FileNotFoundError(path)\n\n    kwargs = {\n        "attn_implementation": ATTN_IMPLEMENTATION,\n        "low_cpu_mem_usage": True,\n    }\n\n    try:\n        model = AutoModelForImageTextToText.from_pretrained(\n            str(path),\n            dtype=dtype,\n            **kwargs,\n        )\n    except TypeError:\n        model = AutoModelForImageTextToText.from_pretrained(\n            str(path),\n            torch_dtype=dtype,\n            **kwargs,\n        )\n\n    model = model.to(device)\n    model.eval()\n\n    if hasattr(model, "config"):\n        model.config.use_cache = True\n\n    for parameter in model.parameters():\n        parameter.requires_grad_(False)\n\n    processor = AutoProcessor.from_pretrained(\n        str(path),\n        min_pixels=MIN_PIXELS,\n        max_pixels=MAX_PIXELS,\n    )\n\n    return model, processor\n\n\ndef row_image_path(row: Dict[str, Any], cam: str) -> str:\n    images = row.get("images")\n\n    if isinstance(images, dict):\n        value = images.get(cam)\n    elif isinstance(images, list) and len(images) == 3:\n        value = images[CAMERAS.index(cam)]\n    else:\n        value = None\n\n    if not value:\n        raise KeyError(f"Missing image {cam}: id={row.get(\'id\')}")\n\n    path = Path(str(value))\n    if not path.is_file():\n        raise FileNotFoundError(path)\n\n    return str(path)\n\n\ndef make_reasoning1200_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:\n    # Exact Reasoning_VLM_1200 prompt:\n    # 3 camera images + mission command only.\n    command = str(\n        row.get("mission_command")\n        or row.get("command")\n        or ""\n    ).strip()\n\n    if not command:\n        raise ValueError(f"Empty mission command: id={row.get(\'id\')}")\n\n    content: List[Dict[str, Any]] = []\n\n    for cam in CAMERAS:\n        content.append(\n            {\n                "type": "image",\n                "path": row_image_path(row, cam),\n            }\n        )\n\n    content.append(\n        {\n            "type": "text",\n            "text": command,\n        }\n    )\n\n    return [{"role": "user", "content": content}]\n\n\ndef build_prompt_1200(\n    processor,\n    row: Dict[str, Any],\n    device: torch.device,\n    dtype: torch.dtype,\n) -> Dict[str, Any]:\n    batch = processor.apply_chat_template(\n        make_reasoning1200_messages(row),\n        tokenize=True,\n        add_generation_prompt=True,\n        return_dict=True,\n        return_tensors="pt",\n    )\n\n    batch = dict(batch)\n    batch.pop("token_type_ids", None)\n\n    moved: Dict[str, Any] = {}\n\n    for key, value in batch.items():\n        if torch.is_tensor(value):\n            value = value.to(device, non_blocking=True)\n            if value.is_floating_point():\n                value = value.to(dtype=dtype)\n        moved[key] = value\n\n    return moved\n\n\ndef get_last_layer_kv(past_key_values):\n    if past_key_values is None:\n        raise RuntimeError("past_key_values is None")\n\n    if hasattr(past_key_values, "layers"):\n        layers = past_key_values.layers\n        if layers:\n            layer = layers[-1]\n            key = getattr(layer, "keys", None)\n            value = getattr(layer, "values", None)\n            if key is not None and value is not None:\n                return key, value\n\n    if (\n        hasattr(past_key_values, "key_cache")\n        and hasattr(past_key_values, "value_cache")\n    ):\n        if past_key_values.key_cache and past_key_values.value_cache:\n            return (\n                past_key_values.key_cache[-1],\n                past_key_values.value_cache[-1],\n            )\n\n    if isinstance(past_key_values, (tuple, list)):\n        last = past_key_values[-1]\n        if isinstance(last, (tuple, list)) and len(last) >= 2:\n            return last[0], last[1]\n\n    raise RuntimeError(\n        f"Unsupported past_key_values type: {type(past_key_values)}"\n    )\n\n\ndef kv_to_sequence(\n    key: torch.Tensor,\n    value: torch.Tensor,\n    max_length: int | None = None,\n) -> torch.Tensor:\n    # [B,H,T,D] K/V -> [B,T,2*H*D]\n    if key.ndim != 4 or value.ndim != 4:\n        raise ValueError(\n            f"Expected K/V [B,H,T,D], "\n            f"got K={tuple(key.shape)}, V={tuple(value.shape)}"\n        )\n\n    if max_length is not None:\n        key = key[..., :max_length, :]\n        value = value[..., :max_length, :]\n\n    key_seq = key.permute(0, 2, 1, 3).contiguous().flatten(2)\n    value_seq = value.permute(0, 2, 1, 3).contiguous().flatten(2)\n\n    return torch.cat([key_seq, value_seq], dim=-1)\n\n\ndef normalize_token_id_set(value: Any) -> set[int]:\n    if value is None:\n        return set()\n    if isinstance(value, int):\n        return {int(value)}\n    if isinstance(value, (list, tuple, set)):\n        return {int(x) for x in value if x is not None}\n    return {int(value)}\n\n\ndef reasoning_content_ids(\n    processor,\n    model,\n    new_ids: torch.Tensor,\n) -> Tuple[List[int], bool]:\n    eos_ids: set[int] = set()\n    eos_ids |= normalize_token_id_set(\n        getattr(processor.tokenizer, "eos_token_id", None)\n    )\n    eos_ids |= normalize_token_id_set(\n        getattr(\n            getattr(model, "generation_config", None),\n            "eos_token_id",\n            None,\n        )\n    )\n\n    for token_text in ("<|im_end|>", "<|endoftext|>"):\n        token_id = processor.tokenizer.convert_tokens_to_ids(token_text)\n        if (\n            token_id is not None\n            and token_id != getattr(processor.tokenizer, "unk_token_id", None)\n            and int(token_id) >= 0\n        ):\n            eos_ids.add(int(token_id))\n\n    pad_ids: set[int] = set()\n    pad_ids |= normalize_token_id_set(\n        getattr(processor.tokenizer, "pad_token_id", None)\n    )\n    pad_ids |= normalize_token_id_set(\n        getattr(\n            getattr(model, "generation_config", None),\n            "pad_token_id",\n            None,\n        )\n    )\n\n    content: List[int] = []\n    ended = False\n\n    for token in new_ids.tolist():\n        token = int(token)\n\n        if token in eos_ids:\n            ended = True\n            break\n\n        if token in pad_ids and content:\n            ended = True\n            break\n\n        content.append(token)\n\n    return content, ended\n\n\n@torch.inference_mode()\ndef extract_one(\n    model,\n    processor,\n    row,\n    device,\n    dtype,\n    max_new_tokens,\n    allow_truncated,\n):\n    # Target-only cache extraction.\n    batch = build_prompt_1200(\n        processor=processor,\n        row=row,\n        device=device,\n        dtype=dtype,\n    )\n\n    prompt_token_length = int(batch["input_ids"].shape[1])\n\n    # 1) Prompt-boundary KV, immediately before reasoning generation.\n    with torch.autocast(\n        device_type="cuda",\n        dtype=dtype,\n        enabled=device.type == "cuda",\n    ):\n        prompt_out = model(\n            **batch,\n            use_cache=True,\n            return_dict=True,\n        )\n\n    direct_key, direct_value = get_last_layer_kv(\n        prompt_out.past_key_values\n    )\n    prompt_cache_length = int(direct_key.shape[-2])\n\n    direct_kv = (\n        kv_to_sequence(\n            direct_key,\n            direct_value,\n            max_length=prompt_cache_length,\n        )[0]\n        .detach()\n        .to(dtype=CACHE_STORAGE_DTYPE)\n        .cpu()\n        .contiguous()\n    )\n\n    del prompt_out, direct_key, direct_value\n\n    # 2) Plain target-model greedy autoregressive reasoning.\n    #    No DFlash / speculative decoding.\n    generation_kwargs = {\n        "max_new_tokens": int(max_new_tokens),\n        "do_sample": False,\n        "use_cache": True,\n        "return_dict_in_generate": True,\n    }\n\n    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)\n    if pad_token_id is not None:\n        generation_kwargs["pad_token_id"] = int(pad_token_id)\n\n    with torch.autocast(\n        device_type="cuda",\n        dtype=dtype,\n        enabled=device.type == "cuda",\n    ):\n        generation = model.generate(\n            **batch,\n            **generation_kwargs,\n        )\n\n    generation_cache = getattr(\n        generation,\n        "past_key_values",\n        None,\n    )\n    if generation_cache is None:\n        raise RuntimeError(\n            "Target model generate() did not return past_key_values."\n        )\n\n    sequences = getattr(generation, "sequences", None)\n    if sequences is None:\n        raise RuntimeError("Target model generate() returned no sequences")\n\n    new_ids = sequences[0, prompt_token_length:]\n    content_ids, ended_eos = reasoning_content_ids(\n        processor,\n        model,\n        new_ids,\n    )\n\n    if not content_ids:\n        raise RuntimeError("Generated reasoning is empty")\n\n    if not ended_eos and not allow_truncated:\n        raise RuntimeError(\n            f"Reasoning did not terminate within "\n            f"max_new_tokens={max_new_tokens}. "\n            "Increase --max-new-tokens or use --allow-truncated."\n        )\n\n    reasoning_text = processor.tokenizer.decode(\n        content_ids,\n        skip_special_tokens=True,\n        clean_up_tokenization_spaces=False,\n    ).strip()\n\n    if not reasoning_text:\n        raise RuntimeError("Decoded reasoning text is empty")\n\n    gen_key, gen_value = get_last_layer_kv(generation_cache)\n    generation_cache_length = int(gen_key.shape[-2])\n\n    desired_reasoning_end = prompt_cache_length + len(content_ids)\n    usable_reasoning_end = min(\n        generation_cache_length,\n        desired_reasoning_end,\n    )\n    missing = max(\n        0,\n        desired_reasoning_end - generation_cache_length,\n    )\n\n    if ended_eos and missing > 0:\n        raise RuntimeError(\n            "Target AR cache does not contain all reasoning tokens: "\n            f"missing={missing}, "\n            f"prompt_cache={prompt_cache_length}, "\n            f"reasoning_tokens={len(content_ids)}, "\n            f"generation_cache={generation_cache_length}"\n        )\n\n    if missing > 1:\n        raise RuntimeError(\n            f"Target AR cache is too short by {missing} tokens"\n        )\n\n    full_kv = (\n        kv_to_sequence(\n            gen_key,\n            gen_value,\n            max_length=usable_reasoning_end,\n        )[0]\n        .detach()\n    )\n\n    prefix_len = min(\n        prompt_cache_length,\n        int(full_kv.shape[0]),\n    )\n    if prefix_len <= 0:\n        raise RuntimeError("Invalid zero-length prompt KV")\n\n    prefix_diff = float(\n        (\n            full_kv[:prefix_len].float().cpu()\n            - direct_kv[:prefix_len].float()\n        )\n        .abs()\n        .max()\n        .item()\n    )\n\n    reasoning_delta = full_kv[\n        prompt_cache_length:usable_reasoning_end\n    ]\n\n    if reasoning_delta.shape[0] <= 0:\n        raise RuntimeError("No reasoning KV tokens were cached")\n\n    reasoning_delta_kv = (\n        reasoning_delta\n        .to(dtype=CACHE_STORAGE_DTYPE)\n        .cpu()\n        .contiguous()\n    )\n\n    trajectory = torch.as_tensor(\n        row["trajectory"],\n        dtype=torch.float32,\n    ).cpu()\n\n    if tuple(trajectory.shape) != (NUM_STEPS, 3):\n        raise ValueError(\n            f"Bad trajectory shape: {tuple(trajectory.shape)}"\n        )\n\n    return {\n        "cache_version": CACHE_VERSION,\n        "id": str(row["id"]),\n        "clip": str(row.get("clip", "")),\n        "direct_kv": direct_kv,\n        "reasoning_delta_kv": reasoning_delta_kv,\n        "trajectory": trajectory.contiguous(),\n        "prompt_token_length": int(prompt_token_length),\n        "prompt_cache_length": int(prompt_cache_length),\n        "reasoning_generated_tokens": int(len(content_ids)),\n        "reasoning_cached_tokens": int(reasoning_delta_kv.shape[0]),\n        "generation_cache_length": int(generation_cache_length),\n        "generation_ended": bool(ended_eos),\n        "missing_reasoning_cache_tokens": int(missing),\n        "prefix_max_abs_diff": float(prefix_diff),\n        "reasoning_token_ids": [int(x) for x in content_ids],\n        "reasoning_text": reasoning_text,\n        "vlm": str(VLM_PATH),\n        "prompt_mode": PROMPT_MODE,\n        "decode_backend": "target_autoregressive",\n        "dflash_used": False,\n    }\n\n\n'
NEW_BUILD_CACHE = 'def build_cache(args, train_rows, val_rows, device, vlm_dtype):\n    if args.cache_root.exists() and not args.rebuild_cache:\n        reused = reuse_cache_if_valid(args, train_rows, val_rows)\n        if reused is not None:\n            return reused\n        raise RuntimeError(\n            f"Incompatible existing cache: {args.cache_root}\\n"\n            "This cache must be rebuilt with target-only AR extraction.\\n"\n            "Use --rebuild-cache."\n        )\n\n    if args.cache_root.exists():\n        shutil.rmtree(args.cache_root)\n    args.cache_root.mkdir(parents=True, exist_ok=True)\n\n    print("\\n" + "=" * 120)\n    print("STAGE 1 | Reasoning_VLM_1200 TARGET-ONLY KV CACHE")\n    print("=" * 120)\n    print("VLM         :", args.vlm)\n    print("Prompt      :", PROMPT_MODE)\n    print("Decode      : target autoregressive greedy")\n    print("DFlash      : NOT USED")\n    print("Attention   :", ATTN_IMPLEMENTATION)\n    print("Train / Val :", len(train_rows), "/", len(val_rows))\n    print("Cache root  :", args.cache_root)\n\n    model, processor = load_target_model(\n        args.vlm,\n        device,\n        vlm_dtype,\n    )\n\n    manifests = {}\n\n    for split, rows in (("train", train_rows), ("val", val_rows)):\n        out_dir = args.cache_root / split\n        out_dir.mkdir(parents=True, exist_ok=True)\n        manifest = []\n\n        for index, row in enumerate(rows, 1):\n            t0 = time.perf_counter()\n            sid = row["id"]\n            out_path = out_dir / cache_filename(index, sid)\n\n            record = extract_one(\n                model=model,\n                processor=processor,\n                row=row,\n                device=device,\n                dtype=vlm_dtype,\n                max_new_tokens=args.max_new_tokens,\n                allow_truncated=args.allow_truncated,\n            )\n\n            torch.save(record, out_path)\n\n            manifest.append({\n                "id": sid,\n                "clip": str(row.get("clip", "")),\n                "cache_file": str(out_path),\n                "kv_dim": int(record["direct_kv"].shape[-1]),\n                "prompt_cache_length": int(record["prompt_cache_length"]),\n                "reasoning_cached_tokens": int(\n                    record["reasoning_cached_tokens"]\n                ),\n                "reasoning_text": record["reasoning_text"],\n                "prefix_max_abs_diff": float(\n                    record["prefix_max_abs_diff"]\n                ),\n            })\n\n            print(\n                f"[cache {split} {index:04d}/{len(rows):04d}] "\n                f"id={sid} "\n                f"directT={record[\'direct_kv\'].shape[0]} "\n                f"reasonT={record[\'reasoning_delta_kv\'].shape[0]} "\n                f"D={record[\'direct_kv\'].shape[-1]} "\n                f"prefix_diff={record[\'prefix_max_abs_diff\']:.3e} "\n                f"{time.perf_counter()-t0:.2f}s",\n                flush=True,\n            )\n\n            del record\n\n        write_jsonl(out_dir / "manifest.jsonl", manifest)\n        manifests[split] = manifest\n\n    cfg = model.config\n    text_cfg = getattr(cfg, "text_config", cfg)\n    n_layers = int(getattr(text_cfg, "num_hidden_layers"))\n\n    meta = {\n        "request": cache_request(args),\n        "runtime": "Reasoning_VLM_1200 target-only AR",\n        "decode_backend": "target_autoregressive",\n        "dflash_used": False,\n        "prompt_mode": PROMPT_MODE,\n        "attention_implementation": ATTN_IMPLEMENTATION,\n        "last_hidden_layer": n_layers - 1,\n        "train_samples": len(manifests["train"]),\n        "val_samples": len(manifests["val"]),\n    }\n\n    (args.cache_root / "meta.json").write_text(\n        json.dumps(meta, ensure_ascii=False, indent=2),\n        encoding="utf-8",\n    )\n\n    del model, processor\n    cleanup_cuda()\n\n    return manifests["train"], manifests["val"]\n\n\n'

if not TARGET.is_file():
    raise FileNotFoundError(TARGET)

src = TARGET.read_text(encoding="utf-8")
shutil.copy2(TARGET, BACKUP)

src = src.replace("import importlib.util\n", "")
src = src.replace("import inspect\n", "")
src = src.replace("import types\n", "")

old_transformers = "from transformers import get_cosine_schedule_with_warmup\n"
new_transformers = (
    "from transformers import (\n"
    "    AutoModelForImageTextToText,\n"
    "    AutoProcessor,\n"
    "    get_cosine_schedule_with_warmup,\n"
    ")\n"
)
if old_transformers not in src:
    raise RuntimeError("Could not find transformers import line")
src = src.replace(old_transformers, new_transformers, 1)

src = re.sub(
    r"\nLEGACY_E2E_CANDIDATES = \(\n.*?\n\)\n",
    "\n",
    src,
    flags=re.S,
)

src = src.replace(
    'CACHE_VERSION = "reasoning_vlm1200_unified_kv_v1"',
    'CACHE_VERSION = "reasoning_vlm1200_target_ar_kv_v2"',
)

anchor = 'CACHE_VERSION = "reasoning_vlm1200_target_ar_kv_v2"\n'
extra = (
    '\n'
    'PROMPT_MODE = "reasoning1200_mission_only"\n'
    'ATTN_IMPLEMENTATION = "eager"\n'
    'MIN_PIXELS = 200_704\n'
    'MAX_PIXELS = 200_704\n'
    'CACHE_STORAGE_DTYPE = torch.bfloat16\n'
)
if anchor not in src:
    raise RuntimeError("Could not find CACHE_VERSION")
src = src.replace(anchor, anchor + extra, 1)

start_marker = (
    "# =============================================================================\n"
    "# TESTED Reasoning_VLM_1200 PREFILL / GENERATION RUNTIME\n"
    "# =============================================================================\n"
)
end_marker = (
    "# =============================================================================\n"
    "# CACHE BUILD / REUSE\n"
    "# =============================================================================\n"
)
start = src.find(start_marker)
end = src.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise RuntimeError("Could not locate old VLM runtime section")
src = src[:start] + NEW_RUNTIME + src[end:]

build_start = src.find(
    "def build_cache(args, train_rows, val_rows, device, vlm_dtype):"
)
dataset_marker = (
    "# =============================================================================\n"
    "# DATASET\n"
    "# =============================================================================\n"
)
build_end = src.find(dataset_marker, build_start)
if build_start < 0 or build_end < 0:
    raise RuntimeError("Could not locate build_cache function")
src = src[:build_start] + NEW_BUILD_CACHE + src[build_end:]

# Rename misleading DFlash prompt label everywhere in this training script.
src = src.replace('"prompt_mode": "dflash1200"', '"prompt_mode": PROMPT_MODE')
src = src.replace(
    'print("Prompt mode   : dflash1200")',
    'print("Prompt mode   :", PROMPT_MODE)',
)

# Strengthen cache identity so stale cache cannot be reused.
needle = (
    '"allow_truncated": args.allow_truncated,\n'
    '        "prompt_mode": PROMPT_MODE,\n'
)
replacement = (
    '"allow_truncated": args.allow_truncated,\n'
    '        "prompt_mode": PROMPT_MODE,\n'
    '        "decode_backend": "target_autoregressive",\n'
    '        "dflash_used": False,\n'
    '        "attn_implementation": ATTN_IMPLEMENTATION,\n'
)
if needle in src:
    src = src.replace(needle, replacement, 1)

for forbidden in (
    "load_vlm_runtime(",
    "load_target_adaptive(",
    "LEGACY_E2E_CANDIDATES",
    "mod.target_prefill",
    "mod.target_step",
):
    if forbidden in src:
        raise RuntimeError(
            f"Old DFlash/E2E dependency remains: {forbidden}"
        )

if "dflash1200" in src.lower():
    raise RuntimeError(
        "Literal dflash1200 still remains; refusing partial patch."
    )

ast.parse(src)
TARGET.write_text(src, encoding="utf-8")

print("=" * 88)
print("PATCH COMPLETE")
print("=" * 88)
print("Modified :", TARGET)
print("Backup   :", BACKUP)
print("Cache    : Reasoning_VLM_1200 target-only AR")
print("DFlash   : NOT USED")
print()
print("Re-run from clean cache/models:")
print(
    "python3 /home/lhh/lab/Action_Expert/scripts/"
    "final_train_all.py "
    "--gpu-id 0 --rebuild-cache --overwrite-models"
)
