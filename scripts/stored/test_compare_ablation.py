#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the two Alpamayo-style ablation branches on exactly paired test IDs.

Outputs:
  - ADE / FDE / Heading MAE
  - Action Expert latency (cache -> trajectory only; NOT full VLM E2E)
  - paired per-sample deltas and win rate
  - bootstrap 95% CI for the mean ADE difference

Interpretation:
  delta ADE = CoC - Traj-only
    < 0 : CoC reasoning branch is better
    > 0 : trajectory-only branch is better
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_model_compare import (
    build_action_expert,
    per_sample_metrics_np,
    trajectory_metrics_np,
)
from train_compare_ablation import (
    BRANCHES,
    MODEL_ROOT,
    REASONING_CACHE_ROOT,
    TRAJ_ONLY_CACHE_ROOT,
    PairEntry,
    PairedBranchDataset,
    collate_kv,
    load_paired_manifest,
    move_batch,
)


RESULT_ROOT = Path("/home/lhh/lab/VLM/Result/ActionExpert2way")
DEFAULT_BATCH_SIZE = 1
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260823


def build_test_loader(
    pairs: Sequence[PairEntry],
    branch: str,
    batch_size: int,
) -> DataLoader:
    dataset = PairedBranchDataset(pairs, branch=branch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
    )


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["action_config"]
    model = build_action_expert(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def evaluate_branch(
    branch: str,
    pairs: Sequence[PairEntry],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
) -> Dict[str, Any]:
    loader = build_test_loader(pairs, branch=branch, batch_size=batch_size)
    model, ckpt = load_checkpoint_model(checkpoint_path, device)

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    ids: List[str] = []
    latency_ms: List[float] = []

    # Warm-up using the first available batch. Do not consume evaluation ordering.
    first_batch = next(iter(loader))
    warm_inputs, _ = move_batch(first_batch, device)
    for _ in range(max(0, warmup_batches)):
        _ = model(**warm_inputs)
    torch.cuda.synchronize(device)

    for batch in loader:
        inputs, gt = move_batch(batch, device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred = model(**inputs)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        bs = int(gt.shape[0])
        latency_ms.extend([elapsed_ms / bs] * bs)
        preds.append(pred.float().cpu().numpy())
        gts.append(gt.float().cpu().numpy())
        ids.extend([str(x) for x in batch["id"]])

    pred_np = np.concatenate(preds, axis=0)
    gt_np = np.concatenate(gts, axis=0)
    metrics = trajectory_metrics_np(pred_np, gt_np)
    per_sample = per_sample_metrics_np(pred_np, gt_np)

    result = {
        "branch": branch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "samples": len(ids),
        "metrics": metrics,
        "ae_latency_ms_mean": float(np.mean(latency_ms)),
        "ae_latency_ms_median": float(np.median(latency_ms)),
        "ids": ids,
        "per_sample": {
            key: value.astype(np.float64).tolist()
            for key, value in per_sample.items()
        },
    }

    del model, loader
    torch.cuda.empty_cache()
    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be non-empty 1D")

    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-only-cache-root", type=Path, default=TRAJ_ONLY_CACHE_ROOT)
    parser.add_argument("--reasoning-cache-root", type=Path, default=REASONING_CACHE_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    test_pairs = load_paired_manifest(
        "test", args.traj_only_cache_root, args.reasoning_cache_root
    )
    if not test_pairs:
        raise RuntimeError("No paired test samples")

    checkpoints = {
        "traj_only": args.model_root / "traj_only" / "best.pt",
        "coc_reasoning": args.model_root / "coc_reasoning" / "best.pt",
    }

    print("=" * 112)
    print("ALPAMAYO-STYLE REASONING ABLATION TEST")
    print("=" * 112)
    print("GPU              :", torch.cuda.get_device_name(args.gpu_id))
    print("Paired test      :", len(test_pairs))
    print("Batch            :", args.batch_size)
    print("Timing           : Action Expert only (cached VLM memory -> trajectory)")
    print()

    results: Dict[str, Any] = {}
    for branch in BRANCHES:
        results[branch] = evaluate_branch(
            branch=branch,
            pairs=test_pairs,
            checkpoint_path=checkpoints[branch],
            device=device,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
        )

    base = results["traj_only"]
    coc = results["coc_reasoning"]

    if base["ids"] != coc["ids"]:
        raise RuntimeError("Test ID ordering mismatch between branches")

    base_ade = np.asarray(base["per_sample"]["ade_m"], dtype=np.float64)
    coc_ade = np.asarray(coc["per_sample"]["ade_m"], dtype=np.float64)
    delta_ade = coc_ade - base_ade

    base_fde = np.asarray(base["per_sample"]["fde_m"], dtype=np.float64)
    coc_fde = np.asarray(coc["per_sample"]["fde_m"], dtype=np.float64)
    delta_fde = coc_fde - base_fde

    base_head = np.asarray(base["per_sample"]["heading_mae_rad"], dtype=np.float64)
    coc_head = np.asarray(coc["per_sample"]["heading_mae_rad"], dtype=np.float64)
    delta_head = coc_head - base_head

    ci_low, ci_high = bootstrap_mean_ci(
        delta_ade,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    base_mean_ade = float(base_ade.mean())
    coc_mean_ade = float(coc_ade.mean())
    rel_improvement = (
        100.0 * (base_mean_ade - coc_mean_ade) / base_mean_ade
        if base_mean_ade != 0.0
        else float("nan")
    )

    paired_summary = {
        "delta_definition": "coc_reasoning - traj_only (negative means CoC is better)",
        "mean_delta_ade_m": float(delta_ade.mean()),
        "median_delta_ade_m": float(np.median(delta_ade)),
        "mean_delta_fde_m": float(delta_fde.mean()),
        "mean_delta_heading_rad": float(delta_head.mean()),
        "coc_win_rate_ade": float((delta_ade < 0.0).mean()),
        "tie_rate_ade": float(np.isclose(delta_ade, 0.0, atol=1e-9).mean()),
        "relative_ade_improvement_percent": float(rel_improvement),
        "bootstrap_95ci_mean_delta_ade_m": [ci_low, ci_high],
        "bootstrap_resamples": int(args.bootstrap),
    }

    print(
        f"{'BRANCH':<22}{'ADE(m)':>12}{'FDE(m)':>12}{'Heading(rad)':>16}"
        f"{'AE mean(ms)':>14}{'AE med(ms)':>13}"
    )
    print("-" * 89)
    for branch in BRANCHES:
        r = results[branch]
        m = r["metrics"]
        print(
            f"{branch:<22}"
            f"{m['ade_m']:>12.4f}"
            f"{m['fde_m']:>12.4f}"
            f"{m['heading_mae_rad']:>16.4f}"
            f"{r['ae_latency_ms_mean']:>14.3f}"
            f"{r['ae_latency_ms_median']:>13.3f}"
        )

    print("\nPAIRED COMPARISON")
    print("-" * 89)
    print(f"ADE delta (CoC - Traj-only) : {delta_ade.mean():+.4f} m")
    print(f"Relative ADE improvement     : {rel_improvement:+.2f}%")
    print(f"CoC ADE win rate             : {(delta_ade < 0.0).mean() * 100.0:.2f}%")
    print(f"Bootstrap 95% CI (ADE delta) : [{ci_low:+.4f}, {ci_high:+.4f}] m")
    print("  CI entirely below 0 => paired test supports CoC improvement.")

    args.result_root.mkdir(parents=True, exist_ok=True)

    # Compact summary without duplicating every per-sample array.
    summary_json = {
        "test_samples": len(test_pairs),
        "traj_only": {
            "checkpoint": base["checkpoint"],
            "checkpoint_epoch": base["checkpoint_epoch"],
            "metrics": base["metrics"],
            "ae_latency_ms_mean": base["ae_latency_ms_mean"],
            "ae_latency_ms_median": base["ae_latency_ms_median"],
        },
        "coc_reasoning": {
            "checkpoint": coc["checkpoint"],
            "checkpoint_epoch": coc["checkpoint_epoch"],
            "metrics": coc["metrics"],
            "ae_latency_ms_mean": coc["ae_latency_ms_mean"],
            "ae_latency_ms_median": coc["ae_latency_ms_median"],
        },
        "paired": paired_summary,
        "timing_scope": "Action Expert only; VLM prefill/reasoning generation excluded because cached memory is used.",
    }
    (args.result_root / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_sample_path = args.result_root / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for i, sample_id in enumerate(base["ids"]):
            row = {
                "id": sample_id,
                "traj_only_ade_m": float(base_ade[i]),
                "coc_reasoning_ade_m": float(coc_ade[i]),
                "delta_ade_m": float(delta_ade[i]),
                "traj_only_fde_m": float(base_fde[i]),
                "coc_reasoning_fde_m": float(coc_fde[i]),
                "delta_fde_m": float(delta_fde[i]),
                "traj_only_heading_rad": float(base_head[i]),
                "coc_reasoning_heading_rad": float(coc_head[i]),
                "delta_heading_rad": float(delta_head[i]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nSaved:")
    print("  ", args.result_root / "summary.json")
    print("  ", per_sample_path)


if __name__ == "__main__":
    main()
