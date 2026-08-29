#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the existing decoder+cross pair with the six newly trained cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

EXISTING_DECODER_CROSS_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_capacity"
)
REMAINING6_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_ablation_05b_remaining6"
)

ARCHITECTURES = (
    "encoder_self",
    "encoder_cross",
    "decoder_self",
    "decoder_cross",
)
BRANCHES = ("traj_only", "coc_reasoning")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
        }

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("val_metrics", {})
    return {
        "exists": True,
        "path": str(path),
        "epoch": int(ckpt.get("epoch", -1)),
        "params": int(ckpt.get("trainable_params", 0)),
        "loss": float(metrics.get("loss", float("nan"))),
        "ade_m": float(metrics.get("ade_m", float("nan"))),
        "fde_m": float(metrics.get("fde_m", float("nan"))),
        "heading_mae_rad": float(metrics.get("heading_mae_rad", float("nan"))),
    }


def resolve_checkpoint(
    existing_root: Path,
    remaining_root: Path,
    branch: str,
    architecture: str,
) -> Path:
    if architecture == "decoder_cross":
        # Existing two-way capacity-control output layout.
        return existing_root / branch / "best.pt"
    return remaining_root / f"{branch}_{architecture}" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=EXISTING_DECODER_CROSS_ROOT,
    )
    parser.add_argument(
        "--remaining-root",
        type=Path,
        default=REMAINING6_ROOT,
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results: Dict[str, Dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        for branch in BRANCHES:
            name = f"{branch}_{architecture}"
            path = resolve_checkpoint(
                args.existing_root,
                args.remaining_root,
                branch,
                architecture,
            )
            results[name] = load_checkpoint(path)

    print("=" * 118)
    print("CAPACITY-SCALED ACTION EXPERT 8-WAY | VALIDATION BEST CHECKPOINTS")
    print("=" * 118)
    print(
        f"{'CONDITION':<40}"
        f"{'PARAMS(B)':>12}"
        f"{'EPOCH':>8}"
        f"{'LOSS':>12}"
        f"{'ADE(m)':>11}"
        f"{'FDE(m)':>11}"
        f"{'HEAD(rad)':>12}"
    )
    print("-" * 118)

    for architecture in ARCHITECTURES:
        for branch in BRANCHES:
            name = f"{branch}_{architecture}"
            r = results[name]
            if not r["exists"]:
                print(f"{name:<40}{'MISSING':>12}  {r['path']}")
                continue
            print(
                f"{name:<40}"
                f"{r['params']/1e9:>12.3f}"
                f"{r['epoch']:>8d}"
                f"{r['loss']:>12.5f}"
                f"{r['ade_m']:>11.4f}"
                f"{r['fde_m']:>11.4f}"
                f"{r['heading_mae_rad']:>12.4f}"
            )

    print("\n" + "=" * 118)
    print("CoC reasoning relative ADE change vs traj_only (same architecture)")
    print("=" * 118)
    for architecture in ARCHITECTURES:
        base = results[f"traj_only_{architecture}"]
        coc = results[f"coc_reasoning_{architecture}"]
        if not base["exists"] or not coc["exists"]:
            print(f"{architecture:<20}: MISSING")
            continue

        b = base["ade_m"]
        c = coc["ade_m"]
        improvement = 100.0 * (b - c) / b if b != 0 else float("nan")
        print(
            f"{architecture:<20}: "
            f"traj={b:.4f}m -> coc={c:.4f}m | "
            f"relative improvement={improvement:+.2f}%"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nJSON saved:", args.json_out)


if __name__ == "__main__":
    main()
