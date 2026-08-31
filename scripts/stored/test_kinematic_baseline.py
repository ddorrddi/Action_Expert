#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kinematic baselines for the current ActionExpert8 Part1 split.

Evaluates:
  1) CV               : constant velocity, straight in current ego frame
  2) CA               : constant signed longitudinal acceleration
                         (default: stops at v=0 instead of reversing)
  3) CA_HEADING       : CA transformed ego->global with current heading,
                         then global->current-ego exactly like the GT frame.
                         This MUST be numerically identical to CA for XY and
                         is included as a frame-consistency sanity check.

Why CA_HEADING == CA:
  GT trajectory is already expressed in the CURRENT EGO frame
  (x=forward, y=left). Current absolute/global heading is used only to define
  that coordinate frame. It does not by itself provide future curvature/yaw-rate.

The ADE/FDE definitions match action_model_ablation_v2.py:
  ADE = mean Euclidean XY error over 10 waypoints
  FDE = mean Euclidean XY error at waypoint 10 (5.0 s)
  heading MAE = wrapped absolute yaw error

Default target:
  /home/lhh/lab/Action_Expert/dataset/ActionExpert8/part1/test.jsonl

GT:
  trajectory: [10, 3] = [x_forward_m, y_left_m, yaw_rad]
  t = 0.5, 1.0, ..., 5.0 sec
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_JSONL = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert8/part1/test.jsonl"
)
DEFAULT_RAW_ROOT = Path("/media/HDD/nuReasoning/train/part_1")
DEFAULT_VLM_SCRIPT_DIR = Path("/home/lhh/lab/VLM/scripts")
DEFAULT_OUTPUT_DIR = Path(
    "/home/lhh/lab/Action_Expert/results/kinematic_baseline"
)

NUM_STEPS = 10
DT = 0.5
TIMES = np.arange(DT, NUM_STEPS * DT + 1e-9, DT, dtype=np.float64)


# =============================================================================
# nuReasoning legacy pickle compatibility
# =============================================================================

_LEGACY_PICKLE_CLASS_CACHE: Dict[Tuple[str, str], type] = {}


def _legacy_pickle_class(module: str, name: str) -> type:
    key = (str(module), str(name))
    if key not in _LEGACY_PICKLE_CLASS_CACHE:
        cls = type(str(name), (), {})
        cls.__module__ = str(module)
        _LEGACY_PICKLE_CLASS_CACHE[key] = cls
    return _LEGACY_PICKLE_CLASS_CACHE[key]


class NuReasoningCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in {"data_schema", "data_schema_v0"}:
            return _legacy_pickle_class(module, name)
        return super().find_class(module, name)


def load_nureasoning_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return NuReasoningCompatUnpickler(f).load()


# =============================================================================
# IO
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []
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
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# CURRENT EGO STATE RESTORATION
# =============================================================================

_RAW_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def raw_clip_metadata(raw_root: Path, clip: str) -> Dict[str, Any]:
    clip = str(clip).strip()
    if not clip:
        raise ValueError("Missing clip name")

    key = f"{raw_root.resolve()}::{clip}"
    if key not in _RAW_METADATA_CACHE:
        path = raw_root / clip / "metadata.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        _RAW_METADATA_CACHE[key] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return _RAW_METADATA_CACHE[key]


def resolve_raw_ego_path(row: Dict[str, Any], raw_root: Path) -> Path:
    clip = str(row.get("clip", "")).strip()
    sid = str(row.get("id", "")).strip()
    frame_index = int(row.get("frame_index", -1))

    metadata = raw_clip_metadata(raw_root, clip)
    frames = metadata.get("frames", [])

    if not isinstance(frames, list):
        raise RuntimeError(f"metadata.frames is not list: clip={clip}")

    chosen = None

    # Same matching order as the latest Action Expert training code.
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        token = str(frame.get("token", "")).strip()
        if token and token == sid:
            chosen = frame
            break

    if chosen is None:
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            try:
                idx = int(frame.get("frame_index", -999999))
            except Exception:
                continue
            if idx == frame_index:
                chosen = frame
                break

    if chosen is None:
        raise RuntimeError(
            f"Could not match raw frame: "
            f"id={sid}, clip={clip}, frame_index={frame_index}"
        )

    ego_value = chosen.get("ego_state")
    if not ego_value:
        raise KeyError(f"Missing ego_state: id={sid}")

    ego_path = Path(str(ego_value)).expanduser()
    if not ego_path.is_absolute():
        ego_path = (raw_root / clip / ego_path).resolve()

    if not ego_path.is_file():
        raise FileNotFoundError(ego_path)

    return ego_path


def import_reasoning_core(vlm_script_dir: Path):
    vlm_script_dir = vlm_script_dir.expanduser().resolve()

    if not vlm_script_dir.is_dir():
        raise FileNotFoundError(vlm_script_dir)

    if str(vlm_script_dir) not in sys.path:
        sys.path.insert(0, str(vlm_script_dir))

    import reasoning_v2_core as core  # type: ignore

    return core


def restore_ego_context(
    row: Dict[str, Any],
    raw_root: Path,
    core,
) -> Tuple[float, float, float]:
    """
    Returns:
        speed_mps
        signed longitudinal acceleration_mps2
        absolute/global heading_rad

    Uses the exact helper semantics used by Reasoning_VLM_v2.
    """
    ego_path = resolve_raw_ego_path(row, raw_root)
    ego_state = load_nureasoning_pickle(ego_path)

    _, _, heading = core.extract_pose(ego_state)
    speed = core.extract_speed_mps(ego_state)
    acceleration = core.extract_acceleration_mps2(
        ego_state,
        heading,
    )

    if speed is None or not math.isfinite(float(speed)):
        raise RuntimeError(f"Invalid speed: id={row.get('id')}")
    if acceleration is None or not math.isfinite(float(acceleration)):
        raise RuntimeError(f"Invalid acceleration: id={row.get('id')}")
    if not math.isfinite(float(heading)):
        raise RuntimeError(f"Invalid heading: id={row.get('id')}")

    return float(speed), float(acceleration), float(heading)


# =============================================================================
# METRICS: identical to current Action Expert evaluation
# =============================================================================

def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_metrics_np(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 3 or pred.shape[1:] != (NUM_STEPS, 3):
        raise ValueError(f"Bad pred shape: {pred.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"GT shape {gt.shape} != pred shape {pred.shape}")

    xy_error = np.linalg.norm(
        pred[..., :2] - gt[..., :2],
        axis=-1,
    )
    heading_error = np.abs(
        wrap_angle_np(pred[..., 2] - gt[..., 2])
    )

    return {
        "ade_m": float(xy_error.mean()),
        "fde_m": float(xy_error[:, -1].mean()),
        "heading_mae_rad": float(heading_error.mean()),
        "longitudinal_mae_m": float(
            np.abs(pred[..., 0] - gt[..., 0]).mean()
        ),
        "lateral_mae_m": float(
            np.abs(pred[..., 1] - gt[..., 1]).mean()
        ),
        "final_longitudinal_mae_m": float(
            np.abs(pred[:, -1, 0] - gt[:, -1, 0]).mean()
        ),
        "final_lateral_mae_m": float(
            np.abs(pred[:, -1, 1] - gt[:, -1, 1]).mean()
        ),
    }


def per_sample_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, np.ndarray]:
    xy = np.linalg.norm(
        pred[..., :2] - gt[..., :2],
        axis=-1,
    )
    heading = np.abs(
        wrap_angle_np(pred[..., 2] - gt[..., 2])
    )

    return {
        "ade_m": xy.mean(axis=1),
        "fde_m": xy[:, -1],
        "heading_mae_rad": heading.mean(axis=1),
    }


# =============================================================================
# KINEMATIC MODELS
# =============================================================================

def distance_cv(v0: float, times: np.ndarray) -> np.ndarray:
    return max(0.0, float(v0)) * times


def distance_ca(
    v0: float,
    a: float,
    times: np.ndarray,
    stop_clamp: bool,
) -> np.ndarray:
    """
    Signed longitudinal constant-acceleration model.

    stop_clamp=True:
      If braking would make velocity negative, hold position after full stop
      instead of allowing the vehicle to reverse unrealistically.
    """
    v0 = max(0.0, float(v0))
    a = float(a)

    s = v0 * times + 0.5 * a * times * times

    if stop_clamp and a < 0.0:
        t_stop = v0 / (-a) if v0 > 0.0 else 0.0
        s_stop = (
            v0 * t_stop + 0.5 * a * t_stop * t_stop
            if t_stop > 0.0
            else 0.0
        )
        s = np.where(times > t_stop, s_stop, s)

    # Numerical/physical guard: forward displacement cannot be negative
    # for this straight-driving baseline.
    return np.maximum(s, 0.0)


def straight_ego_prediction(distance: np.ndarray) -> np.ndarray:
    pred = np.zeros((NUM_STEPS, 3), dtype=np.float64)
    pred[:, 0] = distance
    pred[:, 1] = 0.0

    # Current-ego frame: current yaw is zero by definition.
    # With no yaw-rate/curvature input, constant-heading prediction = 0.
    pred[:, 2] = 0.0
    return pred


def ca_heading_frame_correct_prediction(
    distance: np.ndarray,
    heading: float,
) -> np.ndarray:
    """
    Demonstrates the correct use of CURRENT ABSOLUTE heading.

    1) Move `distance` along current heading in GLOBAL coordinates.
    2) Rotate that displacement back to CURRENT EGO coordinates using
       the same transform used by the dataset.

    Algebraically:
       global dx = s cos(h), dy = s sin(h)
       ego x = cos(h)*dx + sin(h)*dy = s
       ego y = -sin(h)*dx + cos(h)*dy = 0

    Therefore heading does NOT change XY prediction in the current-ego frame.
    """
    h = float(heading)
    c = math.cos(h)
    s = math.sin(h)

    dx_global = distance * c
    dy_global = distance * s

    x_ego = c * dx_global + s * dy_global
    y_ego = -s * dx_global + c * dy_global

    pred = np.zeros((NUM_STEPS, 3), dtype=np.float64)
    pred[:, 0] = x_ego
    pred[:, 1] = y_ego
    pred[:, 2] = 0.0
    return pred


# =============================================================================
# REPORT
# =============================================================================

def print_metrics(name: str, m: Dict[str, float], cv_ade: float) -> None:
    delta = m["ade_m"] - cv_ade
    improve = (
        (cv_ade - m["ade_m"]) / cv_ade * 100.0
        if cv_ade > 0
        else 0.0
    )

    print(f"\n[{name}]")
    print(f"  ADE                    : {m['ade_m']:.4f} m")
    print(f"  FDE                    : {m['fde_m']:.4f} m")
    print(f"  heading MAE            : {m['heading_mae_rad']:.4f} rad")
    print(f"  longitudinal MAE       : {m['longitudinal_mae_m']:.4f} m")
    print(f"  lateral MAE            : {m['lateral_mae_m']:.4f} m")
    print(f"  final longitudinal MAE : {m['final_longitudinal_mae_m']:.4f} m")
    print(f"  final lateral MAE      : {m['final_lateral_mae_m']:.4f} m")
    print(f"  ADE delta vs CV        : {delta:+.4f} m")
    print(f"  ADE improvement vs CV  : {improve:+.2f}%")


def interpretation(best_ade: float) -> str:
    if best_ade < 1.5:
        return (
            "Dynamics-only baseline is already in the 1.x-or-better region. "
            "A learned Action Expert should be expected to compete with 1.x."
        )
    if best_ade < 2.5:
        return (
            "1.x is plausible, but the Action Expert must recover additional "
            "lateral/scene information beyond current speed+acceleration."
        )
    if best_ade < 3.5:
        return (
            "Current-state kinematics alone do not explain a 1.x target. "
            "Substantial gains must come from scene conditioning, trajectory "
            "history/curvature/yaw-rate, representation, or training."
        )
    return (
        "Current-state kinematics are far from 1.x. "
        "Acceleration/absolute heading alone are unlikely to reduce a 3.x "
        "Action Expert directly to 1.x."
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="ActionExpert8 split JSONL to evaluate",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="nuReasoning Part1 raw root",
    )
    p.add_argument(
        "--vlm-script-dir",
        type=Path,
        default=DEFAULT_VLM_SCRIPT_DIR,
        help="Directory containing reasoning_v2_core.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    p.add_argument(
        "--no-stop-clamp",
        action="store_true",
        help="Allow CA to reverse after velocity crosses zero",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.jsonl = args.jsonl.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.vlm_script_dir = args.vlm_script_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    core = import_reasoning_core(args.vlm_script_dir)
    rows = read_jsonl(args.jsonl)

    if args.max_samples is not None:
        rows = rows[: max(0, int(args.max_samples))]

    if not rows:
        raise RuntimeError("No rows to evaluate")

    print("=" * 100)
    print("KINEMATIC BASELINE | CURRENT ACTION EXPERT METRIC")
    print("=" * 100)
    print("JSONL       :", args.jsonl)
    print("Raw root    :", args.raw_root)
    print("Samples     :", len(rows))
    print("Horizon     : 5.0 s")
    print("Waypoints   : 10 @ 0.5 s")
    print("GT frame    : current ego | x=forward, y=left")
    print("CA stop     :", not args.no_stop_clamp)
    print("Metric      : same XY ADE/FDE definition as Action Expert")
    print()
    print(
        "NOTE: heading_rad is absolute/global current heading. "
        "Because GT is already current-ego-frame, a frame-correct "
        "CA+heading XY prediction must equal CA."
    )

    gt_all: List[np.ndarray] = []
    pred_cv_all: List[np.ndarray] = []
    pred_ca_all: List[np.ndarray] = []
    pred_ca_heading_all: List[np.ndarray] = []

    sample_context: List[Dict[str, Any]] = []

    max_heading_equiv_diff = 0.0

    for idx, row in enumerate(rows, 1):
        sid = str(row.get("id", "")).strip()
        gt = np.asarray(row.get("trajectory"), dtype=np.float64)

        if gt.shape != (NUM_STEPS, 3):
            raise ValueError(
                f"trajectory must be [10,3]: id={sid}, got={gt.shape}"
            )
        if not np.isfinite(gt).all():
            raise ValueError(f"Non-finite trajectory: id={sid}")

        speed, acceleration, heading = restore_ego_context(
            row=row,
            raw_root=args.raw_root,
            core=core,
        )

        d_cv = distance_cv(speed, TIMES)
        d_ca = distance_ca(
            speed,
            acceleration,
            TIMES,
            stop_clamp=not args.no_stop_clamp,
        )

        pred_cv = straight_ego_prediction(d_cv)
        pred_ca = straight_ego_prediction(d_ca)
        pred_ca_heading = ca_heading_frame_correct_prediction(
            d_ca,
            heading,
        )

        diff = float(
            np.max(np.abs(pred_ca_heading[..., :2] - pred_ca[..., :2]))
        )
        max_heading_equiv_diff = max(max_heading_equiv_diff, diff)

        gt_all.append(gt)
        pred_cv_all.append(pred_cv)
        pred_ca_all.append(pred_ca)
        pred_ca_heading_all.append(pred_ca_heading)

        sample_context.append(
            {
                "id": sid,
                "clip": str(row.get("clip", "")),
                "frame_index": int(row.get("frame_index", -1)),
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "heading_rad": heading,
            }
        )

        if idx % 50 == 0 or idx == len(rows):
            print(
                f"[LOAD] {idx:4d}/{len(rows):4d}",
                flush=True,
            )

    gt_np = np.stack(gt_all, axis=0)
    pred_cv_np = np.stack(pred_cv_all, axis=0)
    pred_ca_np = np.stack(pred_ca_all, axis=0)
    pred_ca_heading_np = np.stack(pred_ca_heading_all, axis=0)

    metrics = {
        "CV": trajectory_metrics_np(pred_cv_np, gt_np),
        "CA": trajectory_metrics_np(pred_ca_np, gt_np),
        "CA_HEADING_FRAME_CORRECT": trajectory_metrics_np(
            pred_ca_heading_np,
            gt_np,
        ),
    }

    cv_ade = metrics["CV"]["ade_m"]

    print("\n" + "=" * 100)
    print("RESULT")
    print("=" * 100)

    print_metrics("CV", metrics["CV"], cv_ade)
    print_metrics("CA", metrics["CA"], cv_ade)
    print_metrics(
        "CA + CURRENT HEADING (frame-correct)",
        metrics["CA_HEADING_FRAME_CORRECT"],
        cv_ade,
    )

    print("\n[FRAME CONSISTENCY]")
    print(
        "  max |CA_xy - CA_heading_xy| : "
        f"{max_heading_equiv_diff:.12e} m"
    )

    if max_heading_equiv_diff > 1e-8:
        raise RuntimeError(
            "CA+heading did not collapse to CA. "
            "Check coordinate-frame implementation."
        )

    # Per-sample records.
    ps_cv = per_sample_metrics(pred_cv_np, gt_np)
    ps_ca = per_sample_metrics(pred_ca_np, gt_np)

    output_rows: List[Dict[str, Any]] = []
    for i, ctx in enumerate(sample_context):
        output_rows.append(
            {
                **ctx,
                "cv_ade_m": float(ps_cv["ade_m"][i]),
                "cv_fde_m": float(ps_cv["fde_m"][i]),
                "ca_ade_m": float(ps_ca["ade_m"][i]),
                "ca_fde_m": float(ps_ca["fde_m"][i]),
                "pred_cv": pred_cv_np[i].tolist(),
                "pred_ca": pred_ca_np[i].tolist(),
                "gt_trajectory": gt_np[i].tolist(),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.jsonl.stem
    summary_path = args.output_dir / f"{stem}_kinematic_summary.json"
    samples_path = args.output_dir / f"{stem}_kinematic_samples.jsonl"

    best_name = min(
        ("CV", "CA"),
        key=lambda name: metrics[name]["ade_m"],
    )
    best_ade = float(metrics[best_name]["ade_m"])

    summary = {
        "jsonl": str(args.jsonl),
        "raw_root": str(args.raw_root),
        "samples": len(rows),
        "num_steps": NUM_STEPS,
        "dt_sec": DT,
        "horizon_sec": float(NUM_STEPS * DT),
        "frame": {
            "name": "current_ego",
            "x": "forward",
            "y": "left",
        },
        "ca_stop_clamp": not args.no_stop_clamp,
        "metrics": metrics,
        "ca_heading_equivalence_max_xy_diff_m": max_heading_equiv_diff,
        "best_nonlearned_baseline": best_name,
        "best_nonlearned_ade_m": best_ade,
        "interpretation": interpretation(best_ade),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(samples_path, output_rows)

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("Best baseline :", best_name)
    print(f"Best ADE      : {best_ade:.4f} m")
    print(interpretation(best_ade))

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", samples_path)


if __name__ == "__main__":
    main()
