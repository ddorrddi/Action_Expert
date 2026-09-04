#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare nuReasoning Part1 for Action Expert using the NEW fixed VLM split.

What is preserved from the original Action Expert dataset
----------------------------------------------------------
- Use every valid Part1 frame (no task cap, no row-level re-split).
- 3 synchronized cameras: front-left / front / front-right.
- Current speed, longitudinal acceleration, heading/yaw, mission command.
- GT future trajectory: 10 x (x, y, yaw), 0.5 s interval, 5 s horizon,
  expressed in the CURRENT ego frame.
- Require a valid reasoning trace, matching the original Action Expert
  preparation policy and keeping the reasoning ablation sample domain intact.

What changes
------------
- Train/val/test membership is taken ONLY from ~/splits_vlm.json.
- No 80/10/10 ActionExpert8 split is generated.
- The split is checked against /home/lhh/lab/Dataset/manifest.json when present.
- Train/val/test clip and driving-log leakage are hard failures.

Output
------
/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/
    train.jsonl
    val.jsonl
    test.jsonl
    manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HOME = Path.home()
PART1_ROOT = Path("/media/HDD/nuReasoning/train/part_1")
SPLIT_JSON_PATH = HOME / "splits_vlm.json"
CENTRAL_DATASET_MANIFEST = HOME / "lab" / "Dataset" / "manifest.json"
OUTPUT_DIR = HOME / "lab" / "Dataset" / "ActionExpert" / "part1_fixedsplit"

VLM_SCRIPT_DIR = HOME / "lab" / "VLM" / "scripts"
if str(VLM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_SCRIPT_DIR))

import scripts.stored.reasoning_v2_core as core


SPLIT_NAMES = ("train", "val", "test")
NUM_STEPS = 10
XY_MATCH_TOL_M = 1.0e-3


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


# =============================================================================
# FIXED SPLIT
# =============================================================================

def load_fixed_split(split_json: Path) -> Dict[str, Any]:
    if not split_json.is_file():
        raise FileNotFoundError(split_json)

    obj = read_json(split_json)
    if not isinstance(obj, dict):
        raise RuntimeError("splits_vlm.json root must be a dict")

    raw_splits = obj.get("splits")
    clip_to_log = obj.get("clip_to_log")

    if not isinstance(raw_splits, dict):
        raise RuntimeError('splits_vlm.json must contain dict field "splits"')
    if not isinstance(clip_to_log, dict):
        raise RuntimeError('splits_vlm.json must contain dict field "clip_to_log"')

    splits: Dict[str, List[str]] = {}
    for name in SPLIT_NAMES:
        values = raw_splits.get(name)
        if not isinstance(values, list):
            raise RuntimeError(f'splits["{name}"] must be a list')
        cleaned = [str(x).strip() for x in values if str(x).strip()]
        if len(cleaned) != len(set(cleaned)):
            raise RuntimeError(f"Duplicate clips inside split={name}")
        splits[name] = cleaned

    split_sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    if split_sets["train"] & split_sets["val"]:
        raise RuntimeError("train/val clip overlap in splits_vlm.json")
    if split_sets["train"] & split_sets["test"]:
        raise RuntimeError("train/test clip overlap in splits_vlm.json")
    if split_sets["val"] & split_sets["test"]:
        raise RuntimeError("val/test clip overlap in splits_vlm.json")

    clean_clip_to_log = {
        str(k).strip(): str(v).strip()
        for k, v in clip_to_log.items()
    }
    all_clips = set().union(*split_sets.values())

    missing = [
        clip
        for clip in sorted(all_clips)
        if not clean_clip_to_log.get(clip)
    ]
    if missing:
        raise RuntimeError(
            f"Missing clip_to_log entries: count={len(missing)}, sample={missing[:5]}"
        )

    split_logs = {
        name: {clean_clip_to_log[c] for c in split_sets[name]}
        for name in SPLIT_NAMES
    }
    if split_logs["train"] & split_logs["val"]:
        raise RuntimeError("train/val driving-log overlap")
    if split_logs["train"] & split_logs["test"]:
        raise RuntimeError("train/test driving-log overlap")
    if split_logs["val"] & split_logs["test"]:
        raise RuntimeError("val/test driving-log overlap")

    clip_to_split = {}
    for name in SPLIT_NAMES:
        for clip in splits[name]:
            clip_to_split[clip] = name

    return {
        "raw": obj,
        "path": split_json,
        "sha256": sha256_file(split_json),
        "splits": splits,
        "split_sets": split_sets,
        "split_logs": split_logs,
        "clip_to_log": clean_clip_to_log,
        "clip_to_split": clip_to_split,
        "all_clips": all_clips,
    }


def validate_central_manifest(split_spec: Dict[str, Any]) -> None:
    """
    The central VLM fixed-split manifest is only a consistency guard.
    Action Expert rows are rebuilt from raw Part1 and are NOT taken from the
    task-capped central VLM JSONL files.
    """
    if not CENTRAL_DATASET_MANIFEST.is_file():
        print(
            "[INFO] Central Dataset manifest not found; "
            "splits_vlm.json remains authoritative."
        )
        return

    manifest = read_json(CENTRAL_DATASET_MANIFEST)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Invalid central manifest: {CENTRAL_DATASET_MANIFEST}")

    policy = manifest.get("split_policy") or {}
    if not isinstance(policy, dict):
        raise RuntimeError("Invalid split_policy in central manifest")

    saved_sha = str(policy.get("source_sha256", "")).strip()
    if saved_sha and saved_sha != split_spec["sha256"]:
        raise RuntimeError(
            "Central fixed-split manifest and current splits_vlm.json differ.\n"
            f"central={saved_sha}\n"
            f"current={split_spec['sha256']}"
        )

    source_split_clips = manifest.get("source_split_clips")
    if isinstance(source_split_clips, dict):
        for name in SPLIT_NAMES:
            mset = {
                str(x).strip()
                for x in (source_split_clips.get(name) or [])
                if str(x).strip()
            }
            if mset and mset != split_spec["split_sets"][name]:
                raise RuntimeError(
                    f"Central manifest clip membership mismatch: split={name}"
                )

    print("Central fixedsplit manifest check: PASS")


# =============================================================================
# ACTION EXPERT TRAJECTORY TARGET
# =============================================================================

def trajectory_to_ego_xyyaw_5s(ego_state: Any) -> List[List[float]]:
    """
    Original Action Expert convention:
      10 x (x, y, yaw), 0.5...5.0 s, current ego frame.
    """
    x0, y0, yaw0 = core.extract_pose(ego_state)

    future = core.get_field(ego_state, "trajectory_future", None)
    arr = core.future_array(future)
    if arr is None:
        raise RuntimeError("trajectory_future is missing/invalid")

    raw_numeric = None
    try:
        candidate = np.asarray(future, dtype=np.float64)
        if (
            candidate.ndim == 2
            and candidate.shape[0] >= 2
            and candidate.shape[1] >= 2
            and np.isfinite(candidate[:, :2]).all()
        ):
            raw_numeric = candidate
    except Exception:
        raw_numeric = None

    first_is_current = (
        np.linalg.norm(
            arr[0, :2] - np.asarray([x0, y0], dtype=np.float64)
        )
        < 0.25
    )

    if first_is_current:
        t_raw = np.arange(arr.shape[0], dtype=np.float64) * 0.1
    else:
        t_raw = (np.arange(arr.shape[0], dtype=np.float64) + 1.0) * 0.1

    if t_raw[-1] < 5.0 - 1.0e-6:
        raise RuntimeError(
            f"trajectory_future shorter than 5 s: end={t_raw[-1]:.3f}"
        )

    target_t = np.arange(0.5, 5.0 + 1.0e-6, 0.5)
    gx = np.interp(target_t, t_raw, arr[:, 0])
    gy = np.interp(target_t, t_raw, arr[:, 1])

    if (
        raw_numeric is not None
        and raw_numeric.shape[0] == arr.shape[0]
        and raw_numeric.shape[1] >= 3
        and np.isfinite(raw_numeric[:, 2]).all()
    ):
        raw_yaw = np.unwrap(raw_numeric[:, 2].astype(np.float64))
    else:
        dx_raw = np.gradient(arr[:, 0])
        dy_raw = np.gradient(arr[:, 1])
        raw_yaw = np.unwrap(np.arctan2(dy_raw, dx_raw))

    gyaw = np.interp(target_t, t_raw, raw_yaw)

    dx = gx - float(x0)
    dy = gy - float(y0)
    c = math.cos(float(yaw0))
    s = math.sin(float(yaw0))

    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    eyaw = (gyaw - float(yaw0) + np.pi) % (2.0 * np.pi) - np.pi

    traj = np.stack([ex, ey, eyaw], axis=1)
    if traj.shape != (NUM_STEPS, 3) or not np.isfinite(traj).all():
        raise RuntimeError(f"Invalid Action Expert trajectory: {traj.shape}")

    return np.round(traj, 4).tolist()


# =============================================================================
# FRAME PARSING
# =============================================================================

def parse_frame(
    clip_dir: Path,
    metadata: Dict[str, Any],
    frame: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(frame, dict):
        return None, "bad_frame"

    reasoning_path = core.resolve_reasoning_path(frame, clip_dir)
    ego_path = core.resolve_path(
        clip_dir,
        core.get_ci(frame, ["ego_state"]),
    )
    images = core.resolve_three_front_images(frame, clip_dir)

    if reasoning_path is None or not reasoning_path.is_file():
        return None, "missing_reasoning"
    if ego_path is None or not ego_path.is_file():
        return None, "missing_ego"
    if images is None:
        return None, "missing_images"

    try:
        reasoning_obj = core.read_json(reasoning_path)
        ego_state = core.load_pickle(ego_path)
    except Exception:
        return None, "load_error"

    driving = core.get_ci(reasoning_obj, ["Driving"], {}) or {}
    decision = core.get_ci(
        driving,
        ["Driving decision", "driving_decision"],
        {},
    ) or {}

    reasoning_trace = str(
        core.get_ci(
            driving,
            ["Reasoning trace", "reasoning_trace", "Reasoning"],
            "",
        )
    ).strip()
    if not reasoning_trace:
        return None, "empty_reasoning"

    try:
        _, _, heading = core.extract_pose(ego_state)
        speed = core.extract_speed_mps(ego_state)
        acceleration = core.extract_acceleration_mps2(
            ego_state,
            heading,
        )
        trajectory = trajectory_to_ego_xyyaw_5s(ego_state)
        central_xy = core.trajectory_to_ego_xy_5s(ego_state)
    except Exception:
        return None, "ego_parse_error"

    speed = finite_float(speed)
    acceleration = finite_float(acceleration)
    heading = finite_float(heading)

    if speed is None:
        return None, "missing_speed"
    if acceleration is None:
        return None, "missing_acceleration"
    if heading is None:
        return None, "missing_heading"
    if central_xy is None:
        return None, "missing_xy_trajectory"

    central_xy_np = np.asarray(central_xy, dtype=np.float64)
    traj_np = np.asarray(trajectory, dtype=np.float64)
    if central_xy_np.shape != (NUM_STEPS, 2):
        return None, "bad_xy_trajectory"

    xy_max_abs_err = float(
        np.abs(traj_np[:, :2] - central_xy_np).max()
    )
    if xy_max_abs_err > XY_MATCH_TOL_M:
        raise RuntimeError(
            "Action Expert x/y reconstruction differs from fixed VLM "
            f"trajectory convention: clip={clip_dir.name}, "
            f"frame={frame.get('frame_index')}, max_abs_err={xy_max_abs_err:.6f}m"
        )

    frame_index = int(
        core.number(core.get_ci(frame, ["frame_index"], -1))
        if core.number(core.get_ci(frame, ["frame_index"], -1)) is not None
        else -1
    )
    timestamp_us = int(core.frame_timestamp(frame))

    frame_token = str(
        core.get_ci(frame, ["token"], "")
        or f"frame_{frame_index}_{timestamp_us}"
    ).strip()

    command = core.mission_command(frame)
    if not command:
        return None, "missing_command"

    longitudinal = str(
        core.get_ci(decision, ["Longitudinal"], "")
    ).strip()
    lateral = str(
        core.get_ci(decision, ["Lateral"], "")
    ).strip()
    scene_description = str(
        core.get_ci(driving, ["Scene description"], "")
    ).strip()

    sample_id = (
        f"{clip_dir.name}::{frame_index}::{timestamp_us}::{frame_token}"
    )

    row = {
        "id": sample_id,
        "clip": clip_dir.name,
        "scenario_type": str(metadata.get("scenario_type", "unknown")),
        "frame_index": frame_index,
        "timestamp_us": timestamp_us,
        "images": [str(x) for x in images],
        "command": str(command),
        "mission_command": str(command),
        "speed_mps": float(speed),
        "acceleration_mps2": float(acceleration),
        "heading_rad": float(heading),
        "trajectory": trajectory,
        "gt_reasoning": reasoning_trace,
        "scene_description": scene_description,
        "longitudinal": longitudinal,
        "lateral": lateral,
        "xy_match_max_abs_err_m": xy_max_abs_err,
        "_source": {
            "metadata": str(clip_dir / "metadata.json"),
            "reasoning": str(reasoning_path),
            "ego_state": str(ego_path),
        },
    }
    return row, None


def scan_clip(
    clip_dir: Path,
    max_per_clip: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = defaultdict(int)

    metadata_path = clip_dir / "metadata.json"
    if not metadata_path.is_file():
        stats["missing_metadata"] += 1
        return [], dict(stats)

    try:
        metadata = read_json(metadata_path)
    except Exception:
        stats["bad_metadata"] += 1
        return [], dict(stats)

    frames = metadata.get("frames")
    if not isinstance(frames, list):
        stats["bad_frames"] += 1
        return [], dict(stats)

    rows = []
    for frame in frames:
        stats["frames_seen"] += 1
        row, error = parse_frame(clip_dir, metadata, frame)
        if row is None:
            stats[error or "unknown_error"] += 1
            continue
        rows.append(row)
        stats["valid"] += 1

    # Preserve raw temporal order. DataLoader handles training shuffle later.
    rows.sort(
        key=lambda x: (
            int(x.get("frame_index", -1)),
            int(x.get("timestamp_us", 0)),
            str(x.get("id", "")),
        )
    )

    if max_per_clip > 0:
        rows = rows[:max_per_clip]

    return rows, dict(stats)


# =============================================================================
# MAIN
# =============================================================================

def scenario_summary(rows: Sequence[Dict[str, Any]]) -> str:
    counter = Counter(
        str(row.get("scenario_type", "unknown"))
        for row in rows
    )
    return ", ".join(
        f"{name}:{count}"
        for name, count in counter.most_common(8)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--part1-root", type=Path, default=PART1_ROOT)
    p.add_argument("--split-json", type=Path, default=SPLIT_JSON_PATH)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--max-per-clip",
        type=int,
        default=0,
        help="0 = all valid frames. Positive values are for debugging only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.part1_root = args.part1_root.expanduser().resolve()
    args.split_json = args.split_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.part1_root.is_dir():
        raise FileNotFoundError(args.part1_root)

    # Required for nuReasoning ego_state pickles.
    core.install_pickle_aliases()

    split_spec = load_fixed_split(args.split_json)
    validate_central_manifest(split_spec)

    print("=" * 120)
    print("ACTION EXPERT PART1 PREPARE | FIXED VLM SPLIT")
    print("=" * 120)
    print("Raw Part1      :", args.part1_root)
    print("Split JSON     :", args.split_json)
    print("Split SHA256   :", split_spec["sha256"])
    print("Output         :", args.output_dir)
    print(
        "Source clips   :",
        {name: len(split_spec["splits"][name]) for name in SPLIT_NAMES},
    )
    print(
        "Max per clip   :",
        "ALL" if args.max_per_clip <= 0 else args.max_per_clip,
    )
    print("Trajectory     : 10 x (x,y,yaw), current ego, 0.5 s, 5 s")
    print("Task cap       : NONE")
    print("Random re-split: NONE")
    print()

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    valid_clips_by_split: Dict[str, List[str]] = {
        name: [] for name in SPLIT_NAMES
    }
    unavailable_by_split: Dict[str, List[str]] = {
        name: [] for name in SPLIT_NAMES
    }
    total_stats = defaultdict(int)

    total_clips = sum(len(split_spec["splits"][name]) for name in SPLIT_NAMES)
    scanned = 0

    for split_name in SPLIT_NAMES:
        for clip in split_spec["splits"][split_name]:
            scanned += 1
            clip_dir = args.part1_root / clip

            if not clip_dir.is_dir():
                unavailable_by_split[split_name].append(clip)
                total_stats["missing_clip_dir"] += 1
                continue

            rows, stats = scan_clip(
                clip_dir=clip_dir,
                max_per_clip=args.max_per_clip,
            )
            for key, value in stats.items():
                total_stats[key] += int(value)

            if rows:
                valid_clips_by_split[split_name].append(clip)
                rows_by_split[split_name].extend(rows)
            else:
                unavailable_by_split[split_name].append(clip)

            if scanned % 25 == 0 or scanned == total_clips:
                current_rows = sum(len(v) for v in rows_by_split.values())
                print(
                    f"[SCAN] {scanned:4d}/{total_clips:4d} clips | "
                    f"rows={current_rows:6d} | "
                    f"train={len(rows_by_split['train']):5d} "
                    f"val={len(rows_by_split['val']):5d} "
                    f"test={len(rows_by_split['test']):5d}",
                    flush=True,
                )

    for name in SPLIT_NAMES:
        if not rows_by_split[name]:
            raise RuntimeError(f"No valid rows for split={name}")

        ids = [row["id"] for row in rows_by_split[name]]
        if len(ids) != len(set(ids)):
            duplicates = [
                sid for sid, count in Counter(ids).items() if count > 1
            ]
            raise RuntimeError(
                f"Duplicate IDs in split={name}: {duplicates[:5]}"
            )

        invalid_clips = {
            str(row["clip"]) for row in rows_by_split[name]
        } - split_spec["split_sets"][name]
        if invalid_clips:
            raise RuntimeError(
                f"Wrong-split clips in {name}: {sorted(invalid_clips)[:5]}"
            )

    id_sets = {
        name: {row["id"] for row in rows_by_split[name]}
        for name in SPLIT_NAMES
    }
    if id_sets["train"] & id_sets["val"]:
        raise RuntimeError("train/val ID leakage")
    if id_sets["train"] & id_sets["test"]:
        raise RuntimeError("train/test ID leakage")
    if id_sets["val"] & id_sets["test"]:
        raise RuntimeError("val/test ID leakage")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        write_jsonl(
            args.output_dir / f"{name}.jsonl",
            rows_by_split[name],
        )

    manifest = {
        "dataset": "nuReasoning Part1 Action Expert fixedsplit",
        "part1_root": str(args.part1_root),
        "split_policy": {
            "type": "fixed_external_driving_log_split",
            "source_path": str(args.split_json),
            "source_sha256": split_spec["sha256"],
            "clip_disjoint": True,
            "driving_log_disjoint": True,
            "random_resplit": False,
        },
        "sampling_policy": {
            "use_all_valid_frames": args.max_per_clip <= 0,
            "max_per_clip": int(args.max_per_clip),
            "task_cap": None,
            "require_reasoning_trace": True,
        },
        "input": [
            "front_left",
            "front",
            "front_right",
            "mission_command",
            "speed_mps",
            "acceleration_mps2",
            "heading_rad",
        ],
        "trajectory": {
            "frame": "current_ego",
            "waypoints": 10,
            "interval_s": 0.5,
            "horizon_s": 5.0,
            "fields": ["x_m", "y_m", "yaw_rad"],
            "xy_cross_checked_against_reasoning_v2_core": True,
            "xy_tolerance_m": XY_MATCH_TOL_M,
        },
        "source_split_clips": {
            name: split_spec["splits"][name]
            for name in SPLIT_NAMES
        },
        "valid_clips": valid_clips_by_split,
        "unavailable_clips": unavailable_by_split,
        "rows": {
            name: len(rows_by_split[name])
            for name in SPLIT_NAMES
        },
        "scenario_counts": {
            name: dict(
                Counter(
                    str(row.get("scenario_type", "unknown"))
                    for row in rows_by_split[name]
                )
            )
            for name in SPLIT_NAMES
        },
        "scan_stats": dict(total_stats),
        "central_vlm_manifest": (
            str(CENTRAL_DATASET_MANIFEST)
            if CENTRAL_DATASET_MANIFEST.is_file()
            else None
        ),
        "central_vlm_manifest_sha256": (
            sha256_file(CENTRAL_DATASET_MANIFEST)
            if CENTRAL_DATASET_MANIFEST.is_file()
            else None
        ),
    }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("PREPARE COMPLETE")
    print("=" * 120)
    for name in SPLIT_NAMES:
        print(
            f"{name:5s}: "
            f"rows={len(rows_by_split[name]):6d} | "
            f"valid_clips={len(valid_clips_by_split[name]):4d} | "
            f"unavailable={len(unavailable_by_split[name]):4d} | "
            f"{scenario_summary(rows_by_split[name])}"
        )
    print("Clip leakage      : NONE")
    print("Driving-log leak  : NONE")
    print("Random re-split   : NONE")
    print("Task/sample cap   : NONE")
    print("Files             :", args.output_dir)


if __name__ == "__main__":
    main()
