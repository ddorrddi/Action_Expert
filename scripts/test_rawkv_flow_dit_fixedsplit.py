#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test RAW-KV Flow/DiT Action Expert trained by:

train_rawkv_flow_dit_fixedsplit.py

Evaluates:
    direct_flow_dit
    reasoning_flow_dit

Expected checkpoint:
    /home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit/

Expected cache:
    /home/lhh/lab/Dataset/ActionExpert/reasoning_vlm_v2_fixedsplit_rawkv_dit/action_kv_cache

This is NOT the old 10-model tester.
RAW K/V remains separated:
    direct:
        direct_key + direct_value

    reasoning:
        [direct_key + reasoning_delta_key]
        [direct_value + reasoning_delta_value]
"""

from pathlib import Path
import json
import gc
import time
import argparse

import numpy as np
import torch

from action_model_flow_dit import (
    TrajectoryNormalizer,
    trajectory_metrics_np,
)

from train_rawkv_flow_dit_fixedsplit import (
    build_flow_dit_rawkv,
    euler_sample_rawkv,
)


NUM_STEPS = 10
CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/action_kv_cache"
)

MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit"
)

RESULT_ROOT = Path(
    "/home/lhh/lab/E2E/Result/rawkv_dit_test"
)


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def load_manifest(split="val"):
    path = CACHE_ROOT / split / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def load_cache(item, branch):
    cache = torch.load(
        item["cache_file"],
        map_location="cpu",
        weights_only=False,
    )

    dk = cache["direct_key"]
    dv = cache["direct_value"]

    if branch == "direct":
        key = dk
        value = dv
    else:
        rk = cache["reasoning_delta_key"]
        rv = cache["reasoning_delta_value"]

        key = torch.cat([dk, rk], dim=1)
        value = torch.cat([dv, rv], dim=1)

    return {
        "key": key,
        "value": value,
        "gt": cache["trajectory"].float(),
    }


def collate_one(item, branch, device):
    x = load_cache(item, branch)

    key = x["key"].unsqueeze(0).to(
        device,
        dtype=torch.bfloat16
    )
    value = x["value"].unsqueeze(0).to(
        device,
        dtype=torch.bfloat16
    )

    mask = torch.ones(
        (1, key.shape[2]),
        device=device,
        dtype=torch.bool,
    )

    gt = x["gt"].unsqueeze(0).to(device)

    return key, value, mask, gt


def load_model(branch, device):

    ckpt_path = (
        MODEL_ROOT
        / "flow"
        / f"{branch}_flow_dit"
        / "best.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["action_config"]

    model = build_flow_dit_rawkv(
        hidden_dim=cfg["hidden_dim"],
        num_steps=cfg["num_steps"],
        num_layers=cfg["num_layers"],
        vlm_num_attention_heads=ckpt["vlm_num_attention_heads"],
        vlm_num_kv_heads=ckpt["vlm_num_kv_heads"],
        vlm_head_dim=ckpt["vlm_head_dim"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        gradient_checkpointing=False,
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    normalizer = TrajectoryNormalizer.from_dict(
        ckpt["normalizer"]
    ).to(device)

    return model, normalizer, ckpt


@torch.inference_mode()
def evaluate(branch, device, split):

    manifest = load_manifest(split)

    model, normalizer, ckpt = load_model(
        branch,
        device,
    )

    preds = []
    gts = []
    latency = []

    rng = torch.Generator(device="cpu")
    rng.manual_seed(20260841)

    print("=" * 100)
    print(f"TEST {branch}_flow_dit")
    print("checkpoint epoch :", ckpt["epoch"])
    print("samples          :", len(manifest))
    print("=" * 100)

    for i, item in enumerate(manifest, 1):

        key, value, mask, gt = collate_one(
            item,
            branch,
            device,
        )

        torch.cuda.synchronize()
        t0 = time.time()

        pred = euler_sample_rawkv(
            model=model,
            vlm_key=key,
            vlm_value=value,
            memory_mask=mask,
            normalizer=normalizer,
            solver_steps=10,
            rng=rng,
        )

        torch.cuda.synchronize()

        latency.append(
            (time.time()-t0)*1000
        )

        preds.append(
            pred.cpu().numpy()[0]
        )
        gts.append(
            gt.cpu().numpy()[0]
        )

        if i % 25 == 0:
            print(
                f"{i}/{len(manifest)}"
            )

    preds = np.asarray(preds)
    gts = np.asarray(gts)

    metrics = trajectory_metrics_np(
        preds,
        gts
    )

    result = {
        "condition": f"{branch}_flow_dit",
        "epoch": int(ckpt["epoch"]),
        "samples": len(manifest),
        "ADE": float(metrics["ade_m"]),
        "FDE": float(metrics["fde_m"]),
        "Heading": float(metrics["heading_mae_rad"]),
        "AE_ms": float(np.mean(latency)),
    }

    del model
    cleanup()

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="val",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    results = []

    for branch in [
        "direct",
        "reasoning",
    ]:
        results.append(
            evaluate(
                branch,
                device,
                args.split,
            )
        )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_ROOT/"summary.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nFINAL")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
