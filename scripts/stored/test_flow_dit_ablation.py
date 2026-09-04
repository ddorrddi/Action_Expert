#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired Flow-only test for the controlled ~0.5B Flow-Matching Action Expert.

Compares ONLY:
  1) Flow / traj_only
  2) Flow / coc_reasoning

Fairness controls:
  - exact same paired test IDs
  - exact same Gaussian x0 stream for both branches
  - exact same solver steps
  - exact same Action Expert architecture
  - models loaded sequentially for VRAM safety

Timing scope:
  cached VLM memory -> final trajectory

Outputs:
  - result.txt
  - summary.json
  - per_sample.jsonl
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

from scripts.action_model_flow_dit import (
    TrajectoryNormalizer,
    build_flow_dit,
    euler_sample,
    per_sample_metrics_np,
    trajectory_metrics_np,
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


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/alpamayo_flow_dit_05b_controlled_v2"
)

RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/Result/dit_05b_controlled_v2"
)

DEFAULT_BATCH_SIZE = 1
DEFAULT_BOOTSTRAP = 5000
DEFAULT_SEED = 20260823

# Both branches MUST use exactly the same Gaussian x0 sequence.
TEST_NOISE_SEED = 20260825


# =============================================================================
# CONSOLE + TXT LOGGER
# =============================================================================

class Tee:
    """
    Write the same output to multiple streams.

    Used so all console output is simultaneously written to result.txt.
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
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


# =============================================================================
# HELPERS
# =============================================================================

def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_test_loader(
    pairs: Sequence[PairEntry],
    branch: str,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        PairedBranchDataset(
            pairs,
            branch=branch,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_kv,
    )


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    torch.Tensor,
]:
    inputs = {
        "memory": batch["memory"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
        "memory_mask": batch["memory_mask"].to(
            device=device,
            non_blocking=True,
        ),
        "segment_ids": batch["segment_ids"].to(
            device=device,
            non_blocking=True,
        ),
    }

    gt = batch["trajectory"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    return inputs, gt


def load_flow_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[
    torch.nn.Module,
    Dict[str, Any],
    TrajectoryNormalizer,
]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ckpt["action_config"]

    model = build_flow_dit(
        input_dim=int(
            ckpt["input_dim"]
        ),
        hidden_dim=int(
            cfg["hidden_dim"]
        ),
        num_steps=int(
            cfg["num_steps"]
        ),
        num_layers=int(
            cfg["num_layers"]
        ),
        num_heads=int(
            cfg["num_heads"]
        ),
        ff_dim=int(
            cfg["ff_dim"]
        ),
        dropout=float(
            cfg["dropout"]
        ),
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

    normalizer = (
        TrajectoryNormalizer
        .from_dict(
            ckpt["normalizer"]
        )
        .to(device)
    )

    return (
        model,
        ckpt,
        normalizer,
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate_branch(
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

    model, ckpt, normalizer = (
        load_flow_model(
            checkpoint_path,
            device,
        )
    )

    solver_steps = (
        int(
            solver_steps_override
        )
        if solver_steps_override
        is not None
        else int(
            ckpt[
                "action_config"
            ][
                "solver_steps"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # WARM-UP
    # -------------------------------------------------------------------------

    first_batch = next(
        iter(loader)
    )

    warm_inputs, _ = move_batch(
        first_batch,
        device,
    )

    warm_noise = torch.zeros(
        (
            int(
                warm_inputs[
                    "memory"
                ].shape[0]
            ),
            model.num_steps,
            3,
        ),
        device=device,
        dtype=torch.float32,
    )

    for _ in range(
        max(
            0,
            warmup_batches,
        )
    ):
        _ = euler_sample(
            model=model,
            memory=warm_inputs[
                "memory"
            ],
            memory_mask=warm_inputs[
                "memory_mask"
            ],
            segment_ids=warm_inputs[
                "segment_ids"
            ],
            normalizer=normalizer,
            solver_steps=solver_steps,
            noise=warm_noise,
        )

    sync_cuda(
        device
    )

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    preds = []
    gts = []
    ids = []
    latency_ms = []

    # Critical fairness control:
    #
    # Reset to the SAME seed for each branch.
    #
    # Since test sample ordering is also identical,
    # each sample receives the exact same Gaussian x0
    # in traj_only and coc_reasoning.
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

        bs = int(
            gt.shape[0]
        )

        noise = torch.randn(
            (
                bs,
                model.num_steps,
                3,
            ),
            generator=noise_rng,
            dtype=torch.float32,
            device="cpu",
        ).to(
            device
        )

        sync_cuda(
            device
        )

        t0 = time.perf_counter()

        pred = euler_sample(
            model=model,
            memory=inputs[
                "memory"
            ],
            memory_mask=inputs[
                "memory_mask"
            ],
            segment_ids=inputs[
                "segment_ids"
            ],
            normalizer=normalizer,
            solver_steps=solver_steps,
            noise=noise,
        )

        sync_cuda(
            device
        )

        elapsed_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        latency_ms.extend(
            [
                elapsed_ms / bs
            ]
            * bs
        )

        preds.append(
            pred
            .float()
            .cpu()
            .numpy()
        )

        gts.append(
            gt
            .float()
            .cpu()
            .numpy()
        )

        ids.extend(
            [
                str(x)
                for x
                in batch["id"]
            ]
        )

    pred_np = np.concatenate(
        preds,
        axis=0,
    )

    gt_np = np.concatenate(
        gts,
        axis=0,
    )

    metrics = (
        trajectory_metrics_np(
            pred_np,
            gt_np,
        )
    )

    per_sample = (
        per_sample_metrics_np(
            pred_np,
            gt_np,
        )
    )

    result = {
        "family":
            "flow_matching",

        "branch":
            branch,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_epoch":
            int(
                ckpt["epoch"]
            ),

        "trainable_params":
            int(
                ckpt[
                    "trainable_params"
                ]
            ),

        "samples":
            len(ids),

        "solver_steps":
            int(
                solver_steps
            ),

        "noise_seed":
            int(
                TEST_NOISE_SEED
            ),

        "metrics":
            metrics,

        "ae_latency_ms_mean":
            float(
                np.mean(
                    latency_ms
                )
            ),

        "ae_latency_ms_median":
            float(
                np.median(
                    latency_ms
                )
            ),

        "ids":
            ids,

        "per_sample": {
            k:
                v.astype(
                    np.float64
                ).tolist()
            for k, v
            in per_sample.items()
        },
    }

    del (
        model,
        loader,
        ckpt,
        normalizer,
    )

    cleanup_cuda()

    return result


# =============================================================================
# STATISTICS
# =============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[
    float,
    float,
]:
    if n_bootstrap <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = values.size

    means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):
        sample_idx = (
            rng.integers(
                0,
                n,
                size=n,
            )
        )

        means[i] = (
            values[
                sample_idx
            ].mean()
        )

    low, high = np.percentile(
        means,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(low),
        float(high),
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
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
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # OUTPUT DIRECTORY + TXT LOG
    # -------------------------------------------------------------------------

    args.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_txt_path = (
        args.result_root
        / "result.txt"
    )

    log_file = (
        result_txt_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
    )

    original_stdout = (
        sys.stdout
    )

    original_stderr = (
        sys.stderr
    )

    # Save both normal console output and traceback/errors.
    sys.stdout = Tee(
        original_stdout,
        log_file,
    )

    sys.stderr = Tee(
        original_stderr,
        log_file,
    )

    try:
        # ---------------------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required"
            )

        torch.cuda.set_device(
            args.gpu_id
        )

        if not (
            torch.cuda
            .is_bf16_supported()
        ):
            raise RuntimeError(
                "Use BF16-capable "
                "RTX 3080 Ti / GPU 0"
            )

        device = torch.device(
            f"cuda:{args.gpu_id}"
        )

        # ---------------------------------------------------------------------
        # TEST MANIFEST
        # ---------------------------------------------------------------------

        test_pairs = (
            load_paired_manifest(
                "test",
                args.traj_only_cache_root,
                args.reasoning_cache_root,
            )
        )

        # ---------------------------------------------------------------------
        # CHECKPOINTS
        # ---------------------------------------------------------------------

        checkpoints = {
            "traj_only":
                args.model_root
                / "traj_only"
                / "best.pt",

            "coc_reasoning":
                args.model_root
                / "coc_reasoning"
                / "best.pt",
        }

        for (
            branch,
            path,
        ) in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint "
                    f"{branch}: {path}"
                )

        # ---------------------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------------------

        print(
            "=" * 118
        )

        print(
            "FLOW-ONLY PAIRED TEST | "
            "~0.5B CONTROLLED "
            "FLOW MATCHING v2"
        )

        print(
            "=" * 118
        )

        print(
            "GPU              :",
            torch.cuda.get_device_name(
                args.gpu_id
            ),
        )

        print(
            "Paired test      :",
            len(
                test_pairs
            ),
        )

        print(
            "Model root       :",
            args.model_root,
        )

        print(
            "Result root      :",
            args.result_root,
        )

        print(
            "Flow noise seed  :",
            TEST_NOISE_SEED,
        )

        print(
            "Noise fairness   :",
            "identical x0 per sample",
        )

        print(
            "Timing           :",
            "cached VLM memory "
            "-> final trajectory",
        )

        print(
            "Branches         :",
            ", ".join(
                BRANCHES
            ),
        )

        # ---------------------------------------------------------------------
        # RUN TWO BRANCHES
        # ---------------------------------------------------------------------

        results: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for branch in BRANCHES:
            results[
                branch
            ] = evaluate_branch(
                branch=branch,
                pairs=test_pairs,
                checkpoint_path=(
                    checkpoints[
                        branch
                    ]
                ),
                device=device,
                batch_size=(
                    args.batch_size
                ),
                warmup_batches=(
                    args.warmup_batches
                ),
                solver_steps_override=(
                    args.solver_steps
                ),
            )

            r = results[
                branch
            ]

            print(
                f"DONE {branch:<14} | "
                f"ADE="
                f"{r['metrics']['ade_m']:.4f}m "
                f"FDE="
                f"{r['metrics']['fde_m']:.4f}m "
                f"Head="
                f"{r['metrics']['heading_mae_rad']:.4f}rad "
                f"AE="
                f"{r['ae_latency_ms_mean']:.3f}ms"
            )

        # ---------------------------------------------------------------------
        # FAIRNESS VALIDATION
        # ---------------------------------------------------------------------

        base = results[
            "traj_only"
        ]

        coc = results[
            "coc_reasoning"
        ]

        if (
            base["ids"]
            != coc["ids"]
        ):
            raise RuntimeError(
                "Test ID ordering mismatch "
                "between Flow branches"
            )

        if (
            int(
                base[
                    "solver_steps"
                ]
            )
            != int(
                coc[
                    "solver_steps"
                ]
            )
        ):
            raise RuntimeError(
                "Solver-step mismatch: "
                f"traj_only="
                f"{base['solver_steps']} "
                f"coc_reasoning="
                f"{coc['solver_steps']}"
            )

        # ---------------------------------------------------------------------
        # PER-SAMPLE METRICS
        # ---------------------------------------------------------------------

        base_ade = np.asarray(
            base[
                "per_sample"
            ][
                "ade_m"
            ],
            dtype=np.float64,
        )

        coc_ade = np.asarray(
            coc[
                "per_sample"
            ][
                "ade_m"
            ],
            dtype=np.float64,
        )

        delta_ade = (
            coc_ade
            - base_ade
        )

        base_fde = np.asarray(
            base[
                "per_sample"
            ][
                "fde_m"
            ],
            dtype=np.float64,
        )

        coc_fde = np.asarray(
            coc[
                "per_sample"
            ][
                "fde_m"
            ],
            dtype=np.float64,
        )

        delta_fde = (
            coc_fde
            - base_fde
        )

        base_head = np.asarray(
            base[
                "per_sample"
            ][
                "heading_mae_rad"
            ],
            dtype=np.float64,
        )

        coc_head = np.asarray(
            coc[
                "per_sample"
            ][
                "heading_mae_rad"
            ],
            dtype=np.float64,
        )

        delta_head = (
            coc_head
            - base_head
        )

        relative_improvement = (
            100.0
            * (
                base_ade.mean()
                - coc_ade.mean()
            )
            / base_ade.mean()
            if (
                base_ade.mean()
                != 0
            )
            else float(
                "nan"
            )
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                delta_ade,
                args.bootstrap,
                args.seed,
            )
        )

        coc_win_rate = float(
            (
                delta_ade
                < 0
            ).mean()
        )

        # ---------------------------------------------------------------------
        # RESULT TABLE
        # ---------------------------------------------------------------------

        print(
            "\n"
            + "=" * 118
        )

        print(
            "FLOW RESULT TABLE"
        )

        print(
            "=" * 118
        )

        print(
            f"{'BRANCH':<22}"
            f"{'PARAMS(B)':>12}"
            f"{'ADE(m)':>12}"
            f"{'FDE(m)':>12}"
            f"{'HEAD(rad)':>12}"
            f"{'STEPS':>8}"
            f"{'AE ms':>12}"
        )

        print(
            "-" * 118
        )

        for branch in BRANCHES:
            r = results[
                branch
            ]

            m = r[
                "metrics"
            ]

            print(
                f"{branch:<22}"
                f"{r['trainable_params']/1e9:>12.3f}"
                f"{m['ade_m']:>12.4f}"
                f"{m['fde_m']:>12.4f}"
                f"{m['heading_mae_rad']:>12.4f}"
                f"{r['solver_steps']:>8d}"
                f"{r['ae_latency_ms_mean']:>12.3f}"
            )

        # ---------------------------------------------------------------------
        # PAIRED COMPARISON
        # ---------------------------------------------------------------------

        print(
            "\nPAIRED COMPARISON"
        )

        print(
            "-" * 118
        )

        print(
            "ADE delta "
            "(CoC - Traj-only)     : "
            f"{delta_ade.mean():+.4f} m"
        )

        print(
            "FDE delta "
            "(CoC - Traj-only)     : "
            f"{delta_fde.mean():+.4f} m"
        )

        print(
            "Heading delta "
            "(CoC - Traj-only) : "
            f"{delta_head.mean():+.4f} rad"
        )

        print(
            "Relative ADE improvement     : "
            f"{relative_improvement:+.2f}%"
        )

        print(
            "CoC ADE win rate             : "
            f"{coc_win_rate * 100.0:.2f}%"
        )

        print(
            "Bootstrap 95% CI "
            "(ADE delta)    : "
            f"[{ci_low:+.4f}, "
            f"{ci_high:+.4f}] m"
        )

        # ---------------------------------------------------------------------
        # SAVE SUMMARY.JSON
        # ---------------------------------------------------------------------

        summary = {
            "test_samples":
                len(
                    test_pairs
                ),

            "flow_noise_seed":
                int(
                    TEST_NOISE_SEED
                ),

            "solver_steps":
                int(
                    base[
                        "solver_steps"
                    ]
                ),

            "traj_only": {
                k: v
                for k, v
                in base.items()
                if k not in (
                    "ids",
                    "per_sample",
                )
            },

            "coc_reasoning": {
                k: v
                for k, v
                in coc.items()
                if k not in (
                    "ids",
                    "per_sample",
                )
            },

            "paired": {
                "mean_delta_ade_coc_minus_traj_m":
                    float(
                        delta_ade.mean()
                    ),

                "mean_delta_fde_coc_minus_traj_m":
                    float(
                        delta_fde.mean()
                    ),

                "mean_delta_heading_coc_minus_traj_rad":
                    float(
                        delta_head.mean()
                    ),

                "relative_ade_improvement_percent":
                    float(
                        relative_improvement
                    ),

                "coc_ade_win_rate":
                    float(
                        coc_win_rate
                    ),

                "bootstrap_95ci_mean_delta_ade_m": [
                    float(
                        ci_low
                    ),
                    float(
                        ci_high
                    ),
                ],
            },
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

        # ---------------------------------------------------------------------
        # SAVE PER_SAMPLE.JSONL
        # ---------------------------------------------------------------------

        per_sample_path = (
            args.result_root
            / "per_sample.jsonl"
        )

        with per_sample_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for (
                i,
                sample_id,
            ) in enumerate(
                base[
                    "ids"
                ]
            ):
                row = {
                    "id":
                        sample_id,

                    "traj_only_ade_m":
                        float(
                            base_ade[i]
                        ),

                    "coc_reasoning_ade_m":
                        float(
                            coc_ade[i]
                        ),

                    "delta_ade_coc_minus_traj_m":
                        float(
                            delta_ade[i]
                        ),

                    "traj_only_fde_m":
                        float(
                            base_fde[i]
                        ),

                    "coc_reasoning_fde_m":
                        float(
                            coc_fde[i]
                        ),

                    "delta_fde_coc_minus_traj_m":
                        float(
                            delta_fde[i]
                        ),

                    "traj_only_heading_mae_rad":
                        float(
                            base_head[i]
                        ),

                    "coc_reasoning_heading_mae_rad":
                        float(
                            coc_head[i]
                        ),

                    "delta_heading_coc_minus_traj_rad":
                        float(
                            delta_head[i]
                        ),
                }

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # ---------------------------------------------------------------------
        # FINAL PATHS
        # ---------------------------------------------------------------------

        print(
            "\n"
            + "=" * 118
        )

        print(
            "SAVE COMPLETE"
        )

        print(
            "=" * 118
        )

        print(
            "Saved root :",
            args.result_root,
        )

        print(
            "Text log   :",
            result_txt_path,
        )

        print(
            "Summary    :",
            summary_path,
        )

        print(
            "Per sample :",
            per_sample_path,
        )

    finally:
        # Restore terminal streams before closing the txt file.
        sys.stdout.flush()
        sys.stderr.flush()

        sys.stdout = (
            original_stdout
        )

        sys.stderr = (
            original_stderr
        )

        log_file.close()


if __name__ == "__main__":
    main()