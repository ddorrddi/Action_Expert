#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final paired test: 0.499B direct Transformer vs ~0.500B controlled Flow Matching.

Evaluates four checkpoints on the exact same paired test IDs:
  1) Transformer / traj_only
  2) Transformer / coc_reasoning
  3) Flow / traj_only
  4) Flow / coc_reasoning

Key outputs:
  - CoC effect within Transformer
  - CoC effect within Flow
  - Flow-vs-Transformer effect for traj_only
  - Flow-vs-Transformer effect for coc_reasoning

Flow branches use the exact same Gaussian x0 stream.
Models are loaded one at a time to avoid holding four ~0.5B models in VRAM.

Outputs:
  - result.txt       : full terminal-style test report
  - summary.json     : summarized machine-readable results
  - per_sample.jsonl : per-sample ADE results
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_model_compare import (
    build_action_expert,
    per_sample_metrics_np,
    trajectory_metrics_np,
)
from action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    euler_sample,
)
from train_compare_ablation import (
    BRANCHES,
    REASONING_CACHE_ROOT,
    TRAJ_ONLY_CACHE_ROOT,
    PairEntry,
    PairedBranchDataset,
    collate_kv,
    load_paired_manifest,
)


TRANSFORMER_MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_capacity"
)
FLOW_MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_flow_dit_05b_controlled_v2"
)
RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/Result/transformer_vs_flow_05b_v2"
)

DEFAULT_BATCH_SIZE = 1
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260823
TEST_NOISE_SEED = 20260825


# =============================================================================
# TXT LOGGING
# =============================================================================

class Tee:
    """
    Write stdout simultaneously to terminal and a text file.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_test_loader(
    pairs: Sequence[PairEntry],
    branch: str,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        PairedBranchDataset(pairs, branch=branch),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
    )


def move_batch(batch: Dict[str, Any], device: torch.device):
    inputs = {
        "memory": batch["memory"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "memory_mask": batch["memory_mask"].to(
            device,
            non_blocking=True,
        ),
        "segment_ids": batch["segment_ids"].to(
            device,
            non_blocking=True,
        ),
    }

    gt = batch["trajectory"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    return inputs, gt


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_transformer(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["action_config"]

    model = build_action_expert(
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

    return model, ckpt


def load_flow(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[
    torch.nn.Module,
    Dict[str, Any],
    TrajectoryNormalizer,
]:

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

    normalizer = TrajectoryNormalizer.from_dict(
        ckpt["normalizer"]
    ).to(device)

    return model, ckpt, normalizer


@torch.inference_mode()
def evaluate_transformer(
    branch: str,
    pairs: Sequence[PairEntry],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
) -> Dict[str, Any]:

    loader = build_test_loader(
        pairs,
        branch,
        batch_size,
    )

    model, ckpt = load_transformer(
        checkpoint_path,
        device,
    )

    first_batch = next(iter(loader))
    warm_inputs, _ = move_batch(
        first_batch,
        device,
    )

    for _ in range(max(0, warmup_batches)):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            _ = model(**warm_inputs)

    sync_cuda(device)

    preds = []
    gts = []
    ids = []
    latency_ms = []

    for batch in loader:

        inputs, gt = move_batch(
            batch,
            device,
        )

        bs = int(gt.shape[0])

        sync_cuda(device)

        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            pred = model(**inputs)

        sync_cuda(device)

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        latency_ms.extend(
            [elapsed_ms / bs] * bs
        )

        preds.append(
            pred.float().cpu().numpy()
        )

        gts.append(
            gt.cpu().numpy()
        )

        ids.extend(
            [str(x) for x in batch["id"]]
        )

    pred_np = np.concatenate(
        preds,
        axis=0,
    )

    gt_np = np.concatenate(
        gts,
        axis=0,
    )

    metrics = trajectory_metrics_np(
        pred_np,
        gt_np,
    )

    per_sample = per_sample_metrics_np(
        pred_np,
        gt_np,
    )

    result = {
        "family": "transformer_direct",
        "branch": branch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "trainable_params": int(
            ckpt["trainable_params"]
        ),
        "samples": len(ids),
        "metrics": metrics,
        "ae_latency_ms_mean": float(
            np.mean(latency_ms)
        ),
        "ae_latency_ms_median": float(
            np.median(latency_ms)
        ),
        "ids": ids,
        "per_sample": {
            k: v.astype(
                np.float64
            ).tolist()
            for k, v in per_sample.items()
        },
    }

    del model
    del loader
    del ckpt

    cleanup_cuda()

    return result


@torch.inference_mode()
def evaluate_flow(
    branch: str,
    pairs: Sequence[PairEntry],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
    solver_steps_override: int | None,
) -> Dict[str, Any]:

    loader = build_test_loader(
        pairs,
        branch,
        batch_size,
    )

    model, ckpt, normalizer = load_flow(
        checkpoint_path,
        device,
    )

    solver_steps = (
        int(solver_steps_override)
        if solver_steps_override is not None
        else int(
            ckpt["action_config"]["solver_steps"]
        )
    )

    first_batch = next(iter(loader))

    warm_inputs, _ = move_batch(
        first_batch,
        device,
    )

    warm_noise = torch.zeros(
        (
            int(
                warm_inputs["memory"].shape[0]
            ),
            model.num_steps,
            3,
        ),
        device=device,
        dtype=torch.float32,
    )

    for _ in range(max(0, warmup_batches)):
        _ = euler_sample(
            model=model,
            memory=warm_inputs["memory"],
            memory_mask=warm_inputs["memory_mask"],
            segment_ids=warm_inputs["segment_ids"],
            normalizer=normalizer,
            solver_steps=solver_steps,
            noise=warm_noise,
        )

    sync_cuda(device)

    preds = []
    gts = []
    ids = []
    latency_ms = []

    noise_rng = torch.Generator(
        device="cpu"
    )

    noise_rng.manual_seed(
        TEST_NOISE_SEED
    )

    for batch in loader:

        inputs, gt = move_batch(
            batch,
            device,
        )

        bs = int(gt.shape[0])

        noise = torch.randn(
            (
                bs,
                model.num_steps,
                3,
            ),
            generator=noise_rng,
            dtype=torch.float32,
            device="cpu",
        ).to(device)

        sync_cuda(device)

        t0 = time.perf_counter()

        pred = euler_sample(
            model=model,
            memory=inputs["memory"],
            memory_mask=inputs["memory_mask"],
            segment_ids=inputs["segment_ids"],
            normalizer=normalizer,
            solver_steps=solver_steps,
            noise=noise,
        )

        sync_cuda(device)

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        latency_ms.extend(
            [elapsed_ms / bs] * bs
        )

        preds.append(
            pred.cpu().numpy()
        )

        gts.append(
            gt.cpu().numpy()
        )

        ids.extend(
            [str(x) for x in batch["id"]]
        )

    pred_np = np.concatenate(
        preds,
        axis=0,
    )

    gt_np = np.concatenate(
        gts,
        axis=0,
    )

    metrics = trajectory_metrics_np(
        pred_np,
        gt_np,
    )

    per_sample = per_sample_metrics_np(
        pred_np,
        gt_np,
    )

    result = {
        "family": "flow_matching",
        "branch": branch,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            ckpt["epoch"]
        ),
        "trainable_params": int(
            ckpt["trainable_params"]
        ),
        "samples": len(ids),
        "solver_steps": solver_steps,
        "noise_seed": TEST_NOISE_SEED,
        "metrics": metrics,
        "ae_latency_ms_mean": float(
            np.mean(latency_ms)
        ),
        "ae_latency_ms_median": float(
            np.median(latency_ms)
        ),
        "ids": ids,
        "per_sample": {
            k: v.astype(
                np.float64
            ).tolist()
            for k, v in per_sample.items()
        },
    }

    del model
    del loader
    del ckpt
    del normalizer

    cleanup_cuda()

    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
):
    if n_bootstrap <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    n = values.size

    means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(n_bootstrap):

        means[i] = values[
            rng.integers(
                0,
                n,
                size=n,
            )
        ].mean()

    return tuple(
        float(x)
        for x in np.percentile(
            means,
            [2.5, 97.5],
        )
    )


def paired_comparison(
    a: Dict[str, Any],
    b: Dict[str, Any],
    label: str,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:

    if a["ids"] != b["ids"]:
        raise RuntimeError(
            f"ID ordering mismatch for {label}"
        )

    a_ade = np.asarray(
        a["per_sample"]["ade_m"],
        dtype=np.float64,
    )

    b_ade = np.asarray(
        b["per_sample"]["ade_m"],
        dtype=np.float64,
    )

    delta = b_ade - a_ade

    ci = bootstrap_mean_ci(
        delta,
        bootstrap,
        seed,
    )

    rel_improvement_b_vs_a = (
        100.0
        * (
            a_ade.mean()
            - b_ade.mean()
        )
        / a_ade.mean()
    )

    return {
        "label": label,
        "a_mean_ade_m": float(
            a_ade.mean()
        ),
        "b_mean_ade_m": float(
            b_ade.mean()
        ),
        "mean_delta_b_minus_a_m": float(
            delta.mean()
        ),
        "relative_improvement_b_vs_a_percent": float(
            rel_improvement_b_vs_a
        ),
        "b_win_rate": float(
            (delta < 0).mean()
        ),
        "bootstrap_95ci_delta_m": [
            float(ci[0]),
            float(ci[1]),
        ],
    }


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--traj-only-cache-root",
        type=Path,
        default=TRAJ_ONLY_CACHE_ROOT,
    )

    p.add_argument(
        "--reasoning-cache-root",
        type=Path,
        default=REASONING_CACHE_ROOT,
    )

    p.add_argument(
        "--transformer-model-root",
        type=Path,
        default=TRANSFORMER_MODEL_ROOT,
    )

    p.add_argument(
        "--flow-model-root",
        type=Path,
        default=FLOW_MODEL_ROOT,
    )

    p.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
    )

    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    p.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    p.add_argument(
        "--solver-steps",
        type=int,
        default=None,
    )

    return p.parse_args()


def run_test(args):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    torch.cuda.set_device(
        args.gpu_id
    )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Use BF16-capable RTX 3080 Ti / GPU 0"
        )

    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    test_pairs = load_paired_manifest(
        "test",
        args.traj_only_cache_root,
        args.reasoning_cache_root,
    )

    checkpoints = {
        (
            "transformer",
            "traj_only",
        ):
            args.transformer_model_root
            / "traj_only"
            / "best.pt",

        (
            "transformer",
            "coc_reasoning",
        ):
            args.transformer_model_root
            / "coc_reasoning"
            / "best.pt",

        (
            "flow",
            "traj_only",
        ):
            args.flow_model_root
            / "traj_only"
            / "best.pt",

        (
            "flow",
            "coc_reasoning",
        ):
            args.flow_model_root
            / "coc_reasoning"
            / "best.pt",
    }

    for key, path in checkpoints.items():

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint {key}: {path}"
            )

    print("=" * 126)
    print(
        "FINAL PAIRED TEST | ~0.5B DIRECT TRANSFORMER "
        "vs CONTROLLED FLOW MATCHING"
    )
    print("=" * 126)

    print(
        "GPU               :",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )

    print(
        "Paired test       :",
        len(test_pairs),
    )

    print(
        "Transformer root  :",
        args.transformer_model_root,
    )

    print(
        "Flow root         :",
        args.flow_model_root,
    )

    print(
        "Flow noise seed   :",
        TEST_NOISE_SEED,
    )

    print(
        "Models loaded     : sequentially (VRAM safe)"
    )

    results: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # -------------------------------------------------------------------------
    # DIRECT TRANSFORMER
    # -------------------------------------------------------------------------

    for branch in BRANCHES:

        key = f"transformer_{branch}"

        results[key] = evaluate_transformer(
            branch=branch,
            pairs=test_pairs,
            checkpoint_path=checkpoints[
                ("transformer", branch)
            ],
            device=device,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
        )

        print(
            f"DONE {key}: "
            f"ADE={results[key]['metrics']['ade_m']:.4f}m"
        )

    # -------------------------------------------------------------------------
    # FLOW MATCHING
    # -------------------------------------------------------------------------

    for branch in BRANCHES:

        key = f"flow_{branch}"

        results[key] = evaluate_flow(
            branch=branch,
            pairs=test_pairs,
            checkpoint_path=checkpoints[
                ("flow", branch)
            ],
            device=device,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            solver_steps_override=args.solver_steps,
        )

        print(
            f"DONE {key}: "
            f"ADE={results[key]['metrics']['ade_m']:.4f}m"
        )

    # -------------------------------------------------------------------------
    # PAIR CONSISTENCY CHECK
    # -------------------------------------------------------------------------

    canonical_ids = results[
        "transformer_traj_only"
    ]["ids"]

    for key, r in results.items():

        if r["ids"] != canonical_ids:
            raise RuntimeError(
                f"ID ordering mismatch: {key}"
            )

    flow_traj_steps = int(
        results[
            "flow_traj_only"
        ]["solver_steps"]
    )

    flow_coc_steps = int(
        results[
            "flow_coc_reasoning"
        ]["solver_steps"]
    )

    if flow_traj_steps != flow_coc_steps:

        raise RuntimeError(
            "Flow solver-step mismatch: "
            f"traj_only={flow_traj_steps}, "
            f"coc_reasoning={flow_coc_steps}"
        )

    # -------------------------------------------------------------------------
    # PAIRED COMPARISONS
    # -------------------------------------------------------------------------

    comparisons = {

        "reasoning_effect_transformer":
            paired_comparison(
                results[
                    "transformer_traj_only"
                ],
                results[
                    "transformer_coc_reasoning"
                ],
                "Transformer: CoC vs Traj-only",
                args.bootstrap,
                args.seed,
            ),

        "reasoning_effect_flow":
            paired_comparison(
                results[
                    "flow_traj_only"
                ],
                results[
                    "flow_coc_reasoning"
                ],
                "Flow: CoC vs Traj-only",
                args.bootstrap,
                args.seed + 1,
            ),

        "flow_effect_traj_only":
            paired_comparison(
                results[
                    "transformer_traj_only"
                ],
                results[
                    "flow_traj_only"
                ],
                "Traj-only: Flow vs Transformer",
                args.bootstrap,
                args.seed + 2,
            ),

        "flow_effect_coc":
            paired_comparison(
                results[
                    "transformer_coc_reasoning"
                ],
                results[
                    "flow_coc_reasoning"
                ],
                "CoC: Flow vs Transformer",
                args.bootstrap,
                args.seed + 3,
            ),
    }

    # -------------------------------------------------------------------------
    # RESULT TABLE
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 126
    )

    print(
        "RESULT TABLE"
    )

    print(
        "=" * 126
    )

    print(
        f"{'MODEL':<18}"
        f"{'BRANCH':<18}"
        f"{'PARAMS(B)':>12}"
        f"{'ADE(m)':>12}"
        f"{'FDE(m)':>12}"
        f"{'HEAD(rad)':>12}"
        f"{'AE ms':>12}"
        f"{'STEPS':>8}"
    )

    print(
        "-" * 126
    )

    order = [
        "transformer_traj_only",
        "transformer_coc_reasoning",
        "flow_traj_only",
        "flow_coc_reasoning",
    ]

    for key in order:

        r = results[key]
        m = r["metrics"]

        steps = int(
            r.get(
                "solver_steps",
                1,
            )
        )

        print(
            f"{r['family']:<18}"
            f"{r['branch']:<18}"
            f"{r['trainable_params']/1e9:>12.3f}"
            f"{m['ade_m']:>12.4f}"
            f"{m['fde_m']:>12.4f}"
            f"{m['heading_mae_rad']:>12.4f}"
            f"{r['ae_latency_ms_mean']:>12.3f}"
            f"{steps:>8d}"
        )

    # -------------------------------------------------------------------------
    # PAIRED EFFECTS
    # -------------------------------------------------------------------------

    print(
        "\nPAIRED EFFECTS "
        "(positive relative improvement = B is better)"
    )

    print(
        "-" * 126
    )

    for name, c in comparisons.items():

        print(
            f"{c['label']:<42} "
            f"delta={c['mean_delta_b_minus_a_m']:+.4f}m | "
            f"relative="
            f"{c['relative_improvement_b_vs_a_percent']:+.2f}% | "
            f"B win="
            f"{c['b_win_rate']*100:.2f}% | "
            f"95%CI=["
            f"{c['bootstrap_95ci_delta_m'][0]:+.4f}, "
            f"{c['bootstrap_95ci_delta_m'][1]:+.4f}]"
        )

    # -------------------------------------------------------------------------
    # JSON SUMMARY
    # -------------------------------------------------------------------------

    summary = {
        "test_samples": len(
            test_pairs
        ),
        "flow_noise_seed":
            TEST_NOISE_SEED,
        "results": {
            k: {
                kk: vv
                for kk, vv in r.items()
                if kk not in (
                    "ids",
                    "per_sample",
                )
            }
            for k, r in results.items()
        },
        "comparisons":
            comparisons,
    }

    summary_path = (
        args.result_root
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # PER SAMPLE
    # -------------------------------------------------------------------------

    per_sample_path = (
        args.result_root
        / "per_sample.jsonl"
    )

    with per_sample_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, sample_id in enumerate(
            canonical_ids
        ):

            row = {
                "id": sample_id
            }

            for key in order:

                row[
                    f"{key}_ade_m"
                ] = float(
                    results[
                        key
                    ]["per_sample"]["ade_m"][i]
                )

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 126)
    print("FILES")
    print("=" * 126)

    print(
        "Result TXT        :",
        args.result_root / "result.txt",
    )

    print(
        "Summary JSON      :",
        summary_path,
    )

    print(
        "Per-sample JSONL  :",
        per_sample_path,
    )

    print()
    print(
        "Saved:",
        args.result_root,
    )


def main():

    args = parse_args()

    # 결과 폴더를 테스트 시작 전에 먼저 생성
    args.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_path = (
        args.result_root
        / "result.txt"
    )

    original_stdout = sys.stdout

    # w = 실행할 때마다 이전 결과 덮어쓰기
    with txt_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as txt_file:

        sys.stdout = Tee(
            original_stdout,
            txt_file,
        )

        try:
            run_test(args)

        finally:
            sys.stdout = original_stdout

    print(
        f"\nTXT result saved: {txt_path}"
    )


if __name__ == "__main__":
    main()