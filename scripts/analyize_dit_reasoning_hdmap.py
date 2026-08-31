#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning_VLM_v2 | Flow/DiT ONLY | Reasoning ↔ Trajectory ↔ HD-Map Analysis
================================================================================

분석 대상
---------
오직 두 개의 Flow/DiT Action Expert만 비교한다.

  1) direct_flow_dit
     - Input Context K/V만 conditioning

  2) reasoning_flow_dit
     - Input Context K/V + generated Reasoning K/V delta conditioning

Transformer Action Expert는 로드하지 않는다.

이 스크립트의 목적
------------------
"VLM Reasoning이 실제 Flow/DiT trajectory에 도움이 되는가?"
"Reasoning 품질이 낮아서 현재 ADE가 3.xx 수준에 머무는가?"
"생성된 Reasoning 의도와 실제 Reasoning-DiT trajectory가 일치하는가?"
를 샘플 단위 + 전체 통계로 동시에 확인한다.

핵심 통제
---------
- Direct / Reasoning DiT에 동일한 sample-wise Gaussian x0 seed 사용
- GT trajectory 동일성 검사
- 기존 test_dit_hdmap_trajectory_v2.py의
  KV 구성 / checkpoint loader / normalizer / Euler sampler / metric / HD-map parser 재사용
- DFlash 미사용
- Transformer 미사용

출력
----
<output>/<timestamp>/
  samples.csv
  samples.jsonl
  summary.json
  report.md
  top_20_improved.json
  top_20_degraded.json
  counterexamples.json
  scatter/
      token_f1_vs_delta_ade.png
      rouge_l_vs_delta_ade.png
      semantic_intent_vs_delta_ade.png
      delta_ade_histogram.png
  reasoning_improved/*.png
  reasoning_degraded/*.png
  similar/*.png

각 PNG
------
- Front Left / Front / Front Right
- nuReasoning 실제 HD map (ego frame)
- route_path (존재 시)
- GT trajectory
- Direct Flow/DiT trajectory
- Reasoning Flow/DiT trajectory
- Mission Command
- Generated Reasoning / GT Reasoning
- Token F1 / ROUGE-L
- Direct / Reasoning ADE/FDE/Heading
- ΔADE
- Generated-vs-GT reasoning intent 일치
- Reasoning intent ↔ trajectory heuristic alignment
- route-path proximity diagnostic (route가 존재할 때)

주의
----
Reasoning intent parser와 trajectory intent parser는 "보조적 정성 분석용 heuristic"이다.
주 지표는 ADE/FDE/Heading + F1/ROUGE-L + 샘플 시각화이다.

실행
----
cd /home/lhh/lab/Action_Expert/scripts

python3 analyze_dit_reasoning_hdmap.py \
    --gpu-id 0 \
    --limit -1

통계만:
python3 analyze_dit_reasoning_hdmap.py \
    --gpu-id 0 \
    --limit -1 \
    --no-visuals

한 샘플:
python3 analyze_dit_reasoning_hdmap.py \
    --gpu-id 0 \
    --sample-index 37

HD-map frame 자동판정이 틀릴 경우:
python3 analyze_dit_reasoning_hdmap.py \
    --gpu-id 0 \
    --sample-index 37 \
    --map-frame global
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from PIL import Image
except Exception:
    Image = None


# =============================================================================
# PATHS / EXISTING VERIFIED HELPER
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_ROOT / "scripts"

for _p in (SCRIPT_DIR, ACTION_ROOT, ACTION_SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import test_dit_hdmap_trajectory as dit_base
except Exception as exc:
    raise RuntimeError(
        "Failed to import the verified HD-map DiT helper.\n"
        "Expected file:\n"
        "  /home/lhh/lab/Action_Expert/scripts/test_dit_hdmap_trajectory.py\n"
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc


DEFAULT_DATASET = Path(
    "/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl"
)
DEFAULT_RAW_PART2_ROOT = Path(
    "/media/HDD/nuReasoning/train/part_2"
)
DEFAULT_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert10_v2/part2_test_kv_cache"
)
DEFAULT_MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/reasoning_vlm_v2_10model"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/lhh/lab/E2E/Result/dit_reasoning_hdmap_analysis"
)

BRANCHES = ("direct", "reasoning")
FLOW_SEED = 20260841
GLOBAL_SEED = 20260823
DEFAULT_DELTA_THRESHOLD = 1.0
CAMERA_NAMES = ("Front Left", "Front", "Front Right")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def mean_valid(values: Sequence[Any]) -> Optional[float]:
    vals = [finite_or_none(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_or_none(value)
    return "N/A" if x is None else f"{x:.{digits}f}"


def safe_name(text: Any) -> str:
    out = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text))
    return out[:120] if out else "sample"


def wrap_text(text: Any, width: int = 52, max_chars: int = 1000) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if not text:
        return "N/A"
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# CORRELATION
# =============================================================================

def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j

    return ranks


def paired_numeric(
    x: Sequence[Any],
    y: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[float, float]] = []

    for a, b in zip(x, y):
        aa = finite_or_none(a)
        bb = finite_or_none(b)
        if aa is not None and bb is not None:
            pairs.append((aa, bb))

    if not pairs:
        return np.empty(0), np.empty(0)

    arr = np.asarray(pairs, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def pearson_corr(x: Sequence[Any], y: Sequence[Any]) -> Optional[float]:
    xx, yy = paired_numeric(x, y)

    if len(xx) < 3:
        return None
    if np.std(xx) <= 1e-12 or np.std(yy) <= 1e-12:
        return None

    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_corr(x: Sequence[Any], y: Sequence[Any]) -> Optional[float]:
    xx, yy = paired_numeric(x, y)

    if len(xx) < 3:
        return None

    rx = rankdata_average(xx)
    ry = rankdata_average(yy)

    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return None

    return float(np.corrcoef(rx, ry)[0, 1])


# =============================================================================
# REASONING TEXT / SEMANTIC INTENT HEURISTICS
# =============================================================================

def extract_reasoning_intent(text: str) -> Dict[str, Optional[str]]:
    """
    보조 분석용.
    자연어 Reasoning에서 longitudinal/lateral driving intent를 거칠게 추출.
    """
    s = re.sub(r"\s+", " ", str(text or "").lower())

    longitudinal: Optional[str] = None
    lateral: Optional[str] = None

    # Safety-critical / deceleration expressions first.
    if re.search(
        r"\b(stop|stopping|come to a stop|brake|braking|decelerate|"
        r"decelerating|slow down|slowing down|reduce speed|yield)\b",
        s,
    ):
        longitudinal = "slow_or_stop"
    elif re.search(
        r"\b(accelerate|accelerating|speed up|gently accelerate)\b",
        s,
    ):
        longitudinal = "accelerate"
    elif re.search(
        r"\b(maintain speed|keep speed|constant speed|continue at .* speed)\b",
        s,
    ):
        longitudinal = "maintain"

    if re.search(
        r"\b(turn left|left turn|merge left|change lanes? to the left|"
        r"move left|veer left)\b",
        s,
    ):
        lateral = "left"
    elif re.search(
        r"\b(turn right|right turn|merge right|change lanes? to the right|"
        r"move right|veer right)\b",
        s,
    ):
        lateral = "right"
    elif re.search(
        r"\b(go straight|continue straight|keep lane|maintain lane|"
        r"stay in (?:the )?lane)\b",
        s,
    ):
        lateral = "straight"

    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
    }


def intent_match(
    pred: Dict[str, Optional[str]],
    gt: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    evaluated = 0
    matched = 0

    out: Dict[str, Any] = {}

    for axis in ("longitudinal", "lateral"):
        p = pred.get(axis)
        g = gt.get(axis)

        if g is None or p is None:
            out[f"{axis}_match"] = None
            continue

        evaluated += 1
        ok = bool(p == g)
        matched += int(ok)
        out[f"{axis}_match"] = ok

    out["evaluated_axes"] = evaluated
    out["matched_axes"] = matched
    out["score"] = None if evaluated == 0 else float(matched / evaluated)
    return out


def trajectory_motion_summary(
    trajectory: np.ndarray,
    current_speed_mps: Optional[float],
    dt: float = 0.5,
) -> Dict[str, Any]:
    """
    보조 분석용 trajectory intent heuristic.
    x forward / y left인 ego frame을 가정.
    """
    traj = np.asarray(trajectory, dtype=np.float64)

    if traj.shape != (10, 3):
        raise ValueError(f"trajectory must be [10,3], got={traj.shape}")

    xy = traj[:, :2]
    origin = np.zeros((1, 2), dtype=np.float64)
    segments = np.diff(np.concatenate([origin, xy], axis=0), axis=0)
    speeds = np.linalg.norm(segments, axis=1) / float(dt)

    first_speed = float(np.mean(speeds[:2]))
    final_speed = float(np.mean(speeds[-2:]))

    speed_ref = (
        float(current_speed_mps)
        if current_speed_mps is not None
        else first_speed
    )

    if final_speed <= 1.0 or final_speed <= speed_ref - 0.75:
        lon = "slow_or_stop"
    elif final_speed >= speed_ref + 0.75:
        lon = "accelerate"
    else:
        lon = "maintain"

    final_y = float(traj[-1, 1])
    final_heading = float(traj[-1, 2])

    if final_y >= 1.5 or final_heading >= 0.15:
        lat = "left"
    elif final_y <= -1.5 or final_heading <= -0.15:
        lat = "right"
    else:
        lat = "straight"

    return {
        "first_speed_est_mps": first_speed,
        "final_speed_est_mps": final_speed,
        "current_speed_mps": current_speed_mps,
        "final_x_m": float(traj[-1, 0]),
        "final_y_m": final_y,
        "final_heading_rad": final_heading,
        "longitudinal": lon,
        "lateral": lat,
    }


def intent_vs_trajectory(
    reasoning_intent: Dict[str, Optional[str]],
    motion: Dict[str, Any],
) -> Dict[str, Any]:
    evaluated = 0
    matched = 0
    out: Dict[str, Any] = {}

    for axis in ("longitudinal", "lateral"):
        reason_value = reasoning_intent.get(axis)

        if reason_value is None:
            out[f"{axis}_match"] = None
            continue

        evaluated += 1
        ok = bool(reason_value == motion.get(axis))
        matched += int(ok)
        out[f"{axis}_match"] = ok

    out["evaluated_axes"] = evaluated
    out["matched_axes"] = matched
    out["score"] = None if evaluated == 0 else float(matched / evaluated)
    return out


# =============================================================================
# ROUTE-PATH GEOMETRIC DIAGNOSTIC
# =============================================================================

def point_to_segment_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def point_to_polyline_distance(
    p: np.ndarray,
    line: np.ndarray,
) -> float:
    if len(line) == 0:
        return float("nan")

    if len(line) == 1:
        return float(np.linalg.norm(p - line[0]))

    return min(
        point_to_segment_distance(p, line[i], line[i + 1])
        for i in range(len(line) - 1)
    )


def trajectory_route_metrics(
    trajectory: np.ndarray,
    route_ego: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """
    route_path와의 단순 기하학적 거리.
    NPS / driveable-area compliance가 아니며 보조 diagnostic일 뿐이다.
    """
    if route_ego is None or len(route_ego) < 2:
        return {
            "mean_route_distance_m": None,
            "final_route_distance_m": None,
            "max_route_distance_m": None,
        }

    traj = np.asarray(trajectory, dtype=np.float64)
    route = np.asarray(route_ego, dtype=np.float64)

    distances = np.asarray(
        [point_to_polyline_distance(p[:2], route) for p in traj],
        dtype=np.float64,
    )

    return {
        "mean_route_distance_m": float(np.mean(distances)),
        "final_route_distance_m": float(distances[-1]),
        "max_route_distance_m": float(np.max(distances)),
    }


# =============================================================================
# HD-MAP CACHE / LOADING
# =============================================================================

class HDMapClipCache:
    """
    map.pkl은 같은 clip 내 여러 frame에서 반복 사용되므로
    clip별 raw vector geometry를 한 번만 로드한다.
    """

    def __init__(self, raw_root: Path):
        self.raw_root = Path(raw_root)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_clip(self, clip: str) -> Dict[str, Any]:
        clip = str(clip)

        if clip in self._cache:
            return self._cache[clip]

        clip_dir, metadata = dit_base.load_clip_metadata(
            self.raw_root,
            clip,
        )
        map_path = dit_base.resolve_map_path(clip_dir, metadata)
        map_obj = dit_base.load_nureasoning_pickle(map_path)
        map_geoms = dit_base.extract_map_geometries(map_obj)

        if not map_geoms:
            raise RuntimeError(
                f"No HD-map vector geometry extracted: {map_path}"
            )

        bundle = {
            "clip_dir": clip_dir,
            "metadata": metadata,
            "map_path": map_path,
            "map_geoms": map_geoms,
            "map_schema": dit_base.summarize_map_schema(map_obj),
        }
        self._cache[clip] = bundle
        return bundle


def load_sample_hdmap(
    row: Dict[str, Any],
    clip_cache: HDMapClipCache,
    map_frame_arg: str,
) -> Dict[str, Any]:
    clip = str(row["clip"])
    clip_bundle = clip_cache.get_clip(clip)

    clip_dir = clip_bundle["clip_dir"]
    metadata = clip_bundle["metadata"]
    map_path = clip_bundle["map_path"]
    map_geoms = clip_bundle["map_geoms"]

    frame = dit_base.match_raw_frame(row, metadata)
    ego_path = dit_base.resolve_ego_path(clip_dir, frame)
    ego_state = dit_base.load_nureasoning_pickle(ego_path)
    ego_pose = dit_base.extract_ego_pose(ego_state)

    auto_frame, frame_stats = dit_base.estimate_map_frame(
        map_geoms,
        ego_pose[0],
        ego_pose[1],
    )

    used_frame = (
        auto_frame if map_frame_arg == "auto" else map_frame_arg
    )

    map_geoms_ego = dit_base.map_to_ego_frame(
        map_geoms,
        used_frame,
        ego_pose,
    )

    route = dit_base.extract_route_path(frame)
    route_ego = dit_base.route_to_ego_frame(
        route,
        used_frame,
        ego_pose,
    )

    return {
        "clip_dir": clip_dir,
        "metadata": metadata,
        "frame": frame,
        "ego_path": ego_path,
        "ego_pose": ego_pose,
        "map_path": map_path,
        "map_schema": clip_bundle["map_schema"],
        "map_geoms_raw": map_geoms,
        "map_geoms_ego": map_geoms_ego,
        "map_frame_detected": auto_frame,
        "map_frame_used": used_frame,
        "map_frame_stats": frame_stats,
        "route_ego": route_ego,
    }


# =============================================================================
# CAMERA IMAGE HELPERS
# =============================================================================

def resolve_row_images(row: Dict[str, Any]) -> List[Optional[Path]]:
    images = row.get("images")

    if isinstance(images, dict):
        vals = [
            images.get("front_left"),
            images.get("front"),
            images.get("front_right"),
        ]
    elif isinstance(images, list) and len(images) == 3:
        vals = list(images)
    else:
        return [None, None, None]

    out: List[Optional[Path]] = []
    for value in vals:
        if value is None:
            out.append(None)
            continue

        p = Path(str(value)).expanduser()
        out.append(p if p.is_file() else None)

    return out


# =============================================================================
# PLOTTING
# =============================================================================

def draw_heading_arrows(
    ax,
    trajectory: np.ndarray,
    color: Optional[str] = None,
    stride: int = 2,
) -> None:
    traj = np.asarray(trajectory, dtype=np.float64)

    for i in range(0, len(traj), max(1, int(stride))):
        x, y, yaw = traj[i]
        kwargs = dict(
            head_width=0.42,
            head_length=0.58,
            alpha=0.55,
            length_includes_head=True,
            zorder=8,
        )
        if color is not None:
            kwargs["color"] = color

        ax.arrow(
            x,
            y,
            1.5 * math.cos(float(yaw)),
            1.5 * math.sin(float(yaw)),
            **kwargs,
        )


def draw_hdmap_trajectory_panel(
    ax,
    hdmap: Dict[str, Any],
    gt: np.ndarray,
    direct_pred: np.ndarray,
    reasoning_pred: np.ndarray,
    sample_result: Dict[str, Any],
    map_radius: float,
    max_map_elements: int,
) -> int:
    map_geoms_ego = hdmap["map_geoms_ego"]
    route_ego = hdmap["route_ego"]

    all_traj = np.concatenate(
        [
            gt[:, :2],
            direct_pred[:, :2],
            reasoning_pred[:, :2],
        ],
        axis=0,
    )
    traj_extent = float(np.max(np.linalg.norm(all_traj, axis=1)))
    radius = max(float(map_radius), traj_extent + 10.0)

    drawn = 0
    for geom in map_geoms_ego:
        if drawn >= max_map_elements:
            break

        xy = np.asarray(geom.xy, dtype=np.float64)
        if len(xy) < 2:
            continue

        # Cull far geometry before drawing.
        if (
            np.nanmax(xy[:, 0]) < -radius * 1.4
            or np.nanmin(xy[:, 0]) > radius * 1.4
            or np.nanmax(xy[:, 1]) < -radius * 1.4
            or np.nanmin(xy[:, 1]) > radius * 1.4
        ):
            continue

        style = dit_base.geometry_style(geom.category)
        ax.plot(xy[:, 0], xy[:, 1], zorder=1, **style)
        drawn += 1

    if route_ego is not None and len(route_ego) >= 2:
        ax.plot(
            route_ego[:, 0],
            route_ego[:, 1],
            linewidth=2.2,
            linestyle=":",
            alpha=0.85,
            label="Route path",
            zorder=4,
        )

    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=180,
        label="Current ego",
        zorder=10,
    )
    ax.arrow(
        0.0,
        0.0,
        2.5,
        0.0,
        head_width=0.8,
        head_length=1.0,
        zorder=10,
        length_includes_head=True,
    )

    gt_line, = ax.plot(
        gt[:, 0],
        gt[:, 1],
        linewidth=3.0,
        marker="o",
        markersize=5,
        label="Ground Truth",
        zorder=8,
    )
    direct_line, = ax.plot(
        direct_pred[:, 0],
        direct_pred[:, 1],
        linewidth=2.5,
        linestyle="--",
        marker="x",
        markersize=6,
        label=(
            f"Direct DiT "
            f"(ADE={sample_result['direct_ade_m']:.3f}m)"
        ),
        zorder=9,
    )
    reasoning_line, = ax.plot(
        reasoning_pred[:, 0],
        reasoning_pred[:, 1],
        linewidth=2.5,
        linestyle="-.",
        marker="s",
        markersize=5,
        label=(
            f"Reasoning DiT "
            f"(ADE={sample_result['reasoning_ade_m']:.3f}m)"
        ),
        zorder=9,
    )

    draw_heading_arrows(ax, gt, gt_line.get_color())
    draw_heading_arrows(ax, direct_pred, direct_line.get_color())
    draw_heading_arrows(ax, reasoning_pred, reasoning_line.get_color())

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Ego-frame x [m] (forward)")
    ax.set_ylabel("Ego-frame y [m] (left)")

    route_r = sample_result.get("reasoning_route", {})
    route_text = ""
    if route_r.get("mean_route_distance_m") is not None:
        route_text = (
            f" | Reason route-dist="
            f"{route_r['mean_route_distance_m']:.2f}m"
        )

    ax.set_title(
        "nuReasoning HD Map + Flow/DiT trajectories\n"
        f"ΔADE={sample_result['delta_ade_m']:+.3f}m "
        f"(R-D), {sample_result['delta_class']}"
        f"{route_text}"
    )
    ax.legend(loc="best", fontsize=9)

    return drawn


def save_sample_visualization(
    output_path: Path,
    row: Dict[str, Any],
    hdmap: Dict[str, Any],
    result: Dict[str, Any],
    gt: np.ndarray,
    direct_pred: np.ndarray,
    reasoning_pred: np.ndarray,
    map_radius: float,
    max_map_elements: int,
) -> None:
    image_paths = resolve_row_images(row)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 1.12),
        height_ratios=(0.9, 1.35),
        hspace=0.15,
        wspace=0.12,
    )

    # -------------------------------------------------------------------------
    # 3 cameras
    # -------------------------------------------------------------------------
    for i, (cam_name, img_path) in enumerate(
        zip(CAMERA_NAMES, image_paths)
    ):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.set_title(cam_name)

        if img_path is None:
            ax.text(
                0.5,
                0.5,
                "Image unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        try:
            if Image is not None:
                with Image.open(img_path) as img:
                    ax.imshow(img.convert("RGB"))
            else:
                ax.imshow(plt.imread(img_path))
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Image load failed\n{type(exc).__name__}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # -------------------------------------------------------------------------
    # Quick summary
    # -------------------------------------------------------------------------
    quick = fig.add_subplot(gs[0, 3])
    quick.axis("off")

    semantic = result["reasoning_semantic_intent_match"]

    quick_lines = [
        "SAMPLE SUMMARY",
        "",
        f"ID      : {result['id']}",
        f"Command : {result.get('mission_command', '')}",
        f"Speed   : {fmt(result.get('current_speed_mps'), 2)} m/s",
        "",
        "TRAJECTORY",
        (
            f"Direct    ADE {result['direct_ade_m']:.4f} | "
            f"FDE {result['direct_fde_m']:.4f}"
        ),
        (
            f"Reasoning ADE {result['reasoning_ade_m']:.4f} | "
            f"FDE {result['reasoning_fde_m']:.4f}"
        ),
        f"ΔADE (R-D): {result['delta_ade_m']:+.4f} m",
        f"Gain (D-R): {result['reasoning_gain_ade_m']:+.4f} m",
        f"Class      : {result['delta_class']}",
        "",
        "REASONING QUALITY",
        f"Token F1   : {fmt(result.get('reasoning_token_f1'), 4)}",
        f"ROUGE-L    : {fmt(result.get('reasoning_rouge_l_f1'), 4)}",
        f"Combined   : {fmt(result.get('reasoning_quality_mean'), 4)}",
        (
            "Intent match: "
            f"{fmt(semantic.get('score'), 2)} "
            f"({semantic.get('matched_axes', 0)}/"
            f"{semantic.get('evaluated_axes', 0)})"
        ),
        "",
        "HD MAP",
        f"Frame detected: {hdmap['map_frame_detected']}",
        f"Frame used    : {hdmap['map_frame_used']}",
        f"Route available: {hdmap['route_ego'] is not None}",
    ]

    quick.text(
        0.0,
        1.0,
        "\n".join(quick_lines),
        transform=quick.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        family="monospace",
    )

    # -------------------------------------------------------------------------
    # HD map + trajectories
    # -------------------------------------------------------------------------
    map_ax = fig.add_subplot(gs[1, :3])

    drawn = draw_hdmap_trajectory_panel(
        map_ax,
        hdmap=hdmap,
        gt=gt,
        direct_pred=direct_pred,
        reasoning_pred=reasoning_pred,
        sample_result=result,
        map_radius=map_radius,
        max_map_elements=max_map_elements,
    )

    # -------------------------------------------------------------------------
    # Detailed reasoning panel
    # -------------------------------------------------------------------------
    info = fig.add_subplot(gs[1, 3])
    info.axis("off")

    pred_intent = result["generated_reasoning_intent"]
    gt_intent = result["gt_reasoning_intent"]

    d_align = result["generated_reasoning_vs_direct_trajectory"]
    r_align = result["generated_reasoning_vs_reasoning_trajectory"]
    gt_align = result["gt_reasoning_vs_gt_trajectory"]

    lines = [
        "REASONING / TRAJECTORY ALIGNMENT",
        "",
        "GENERATED REASONING",
        wrap_text(
            result.get("generated_reasoning", ""),
            width=44,
            max_chars=850,
        ),
        "",
        "GT REASONING",
        wrap_text(
            result.get("gt_reasoning", ""),
            width=44,
            max_chars=850,
        ),
        "",
        "INTENT PARSE (heuristic)",
        (
            f"Pred: lon={pred_intent.get('longitudinal')} | "
            f"lat={pred_intent.get('lateral')}"
        ),
        (
            f"GT  : lon={gt_intent.get('longitudinal')} | "
            f"lat={gt_intent.get('lateral')}"
        ),
        (
            "Pred-vs-GT semantic intent score: "
            f"{fmt(semantic.get('score'), 2)}"
        ),
        "",
        "TRAJECTORY INTENT (heuristic)",
        (
            f"GT traj       : "
            f"{result['gt_motion']['longitudinal']} / "
            f"{result['gt_motion']['lateral']}"
        ),
        (
            f"Direct traj   : "
            f"{result['direct_motion']['longitudinal']} / "
            f"{result['direct_motion']['lateral']}"
        ),
        (
            f"Reasoning traj: "
            f"{result['reasoning_motion']['longitudinal']} / "
            f"{result['reasoning_motion']['lateral']}"
        ),
        "",
        "REASONING -> TRAJECTORY ALIGNMENT",
        (
            f"Generated vs Direct traj   : "
            f"{fmt(d_align.get('score'), 2)}"
        ),
        (
            f"Generated vs Reasoning traj: "
            f"{fmt(r_align.get('score'), 2)}"
        ),
        (
            f"GT Reasoning vs GT traj    : "
            f"{fmt(gt_align.get('score'), 2)}"
        ),
        "",
        "PAIRWISE TRAJECTORY CHANGE",
        (
            f"Direct↔Reason mean XY: "
            f"{result['direct_reasoning_mean_xy_diff_m']:.3f} m"
        ),
        (
            f"Direct↔Reason final XY: "
            f"{result['direct_reasoning_final_xy_diff_m']:.3f} m"
        ),
        "",
        "ROUTE PROXIMITY (diagnostic only)",
        (
            f"GT mean     : "
            f"{fmt(result['gt_route'].get('mean_route_distance_m'), 3)} m"
        ),
        (
            f"Direct mean : "
            f"{fmt(result['direct_route'].get('mean_route_distance_m'), 3)} m"
        ),
        (
            f"Reason mean : "
            f"{fmt(result['reasoning_route'].get('mean_route_distance_m'), 3)} m"
        ),
        "",
        f"Map vectors drawn: {drawn}",
    ]

    info.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=info.transAxes,
        va="top",
        ha="left",
        fontsize=8.55,
        family="monospace",
    )

    fig.suptitle(
        "Reasoning_VLM_v2 | Flow/DiT-only Reasoning-Trajectory Analysis",
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatter(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
) -> None:
    x, y = paired_numeric(
        [r.get(metric_key) for r in rows],
        [r.get("delta_ade_m") for r in rows],
    )

    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.6)

    if np.std(x) > 1e-12:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=2)

    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("ΔADE [m] = Reasoning ADE - Direct ADE")
    ax.set_title(
        f"{metric_label} vs ΔADE\n"
        "negative ΔADE = Reasoning K/V improves trajectory"
    )
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_delta_histogram(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    vals = [
        finite_or_none(r.get("delta_ade_m"))
        for r in rows
    ]
    vals = [v for v in vals if v is not None]

    if len(vals) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(
        vals,
        bins=min(30, max(10, len(vals) // 10)),
        alpha=0.8,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("ΔADE [m] = Reasoning ADE - Direct ADE")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Reasoning K/V Effect")
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SELECTION
# =============================================================================

def select_rows(
    rows: Sequence[Dict[str, Any]],
    sample_index: Optional[int],
    sample_id: Optional[str],
    limit: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    if sample_id is not None:
        sid = str(sample_id)
        matches = [
            (i, row)
            for i, row in enumerate(rows)
            if str(row["id"]) == sid
        ]

        if not matches:
            raise KeyError(f"sample id not found: {sid}")

        return matches[:1]

    if sample_index is not None:
        idx = int(sample_index)

        if idx < 0 or idx >= len(rows):
            raise IndexError(
                f"--sample-index out of range: {idx}; dataset={len(rows)}"
            )

        return [(idx, rows[idx])]

    if limit == 0:
        raise ValueError("--limit must be positive or -1")

    selected = list(enumerate(rows))

    if limit > 0:
        selected = selected[:limit]

    return selected


def classify_delta(delta_ade: float, threshold: float) -> str:
    if delta_ade < -threshold:
        return "reasoning_improved"
    if delta_ade > threshold:
        return "reasoning_degraded"
    return "similar"


# =============================================================================
# CSV
# =============================================================================

CSV_FIELDS = [
    "index",
    "id",
    "clip",
    "mission_command",
    "current_speed_mps",

    "reasoning_token_f1",
    "reasoning_rouge_l_f1",
    "reasoning_quality_mean",
    "reasoning_semantic_intent_score",

    "direct_ade_m",
    "reasoning_ade_m",
    "delta_ade_m",
    "reasoning_gain_ade_m",

    "direct_fde_m",
    "reasoning_fde_m",
    "delta_fde_m",

    "direct_heading_mae_rad",
    "reasoning_heading_mae_rad",
    "delta_heading_rad",

    "delta_class",

    "direct_reasoning_mean_xy_diff_m",
    "direct_reasoning_final_xy_diff_m",

    "generated_intent_longitudinal",
    "generated_intent_lateral",
    "gt_intent_longitudinal",
    "gt_intent_lateral",

    "gt_motion_longitudinal",
    "gt_motion_lateral",
    "direct_motion_longitudinal",
    "direct_motion_lateral",
    "reasoning_motion_longitudinal",
    "reasoning_motion_lateral",

    "generated_vs_direct_alignment",
    "generated_vs_reasoning_alignment",
    "gt_reasoning_vs_gt_alignment",

    "gt_route_mean_distance_m",
    "direct_route_mean_distance_m",
    "reasoning_route_mean_distance_m",

    "map_frame_detected",
    "map_frame_used",
    "map_vector_count",
    "route_available",

    "reasoning_generated_tokens",
    "reasoning_cached_tokens",

    "generated_reasoning",
    "gt_reasoning",
]


def save_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "index": r["index"],
                "id": r["id"],
                "clip": r.get("clip", ""),
                "mission_command": r.get("mission_command", ""),
                "current_speed_mps": r.get("current_speed_mps"),

                "reasoning_token_f1": r.get("reasoning_token_f1"),
                "reasoning_rouge_l_f1": r.get("reasoning_rouge_l_f1"),
                "reasoning_quality_mean": r.get("reasoning_quality_mean"),
                "reasoning_semantic_intent_score": (
                    r.get("reasoning_semantic_intent_match", {}).get("score")
                ),

                "direct_ade_m": r["direct_ade_m"],
                "reasoning_ade_m": r["reasoning_ade_m"],
                "delta_ade_m": r["delta_ade_m"],
                "reasoning_gain_ade_m": r["reasoning_gain_ade_m"],

                "direct_fde_m": r["direct_fde_m"],
                "reasoning_fde_m": r["reasoning_fde_m"],
                "delta_fde_m": r["delta_fde_m"],

                "direct_heading_mae_rad": r["direct_heading_mae_rad"],
                "reasoning_heading_mae_rad": r[
                    "reasoning_heading_mae_rad"
                ],
                "delta_heading_rad": r["delta_heading_rad"],

                "delta_class": r["delta_class"],

                "direct_reasoning_mean_xy_diff_m": r[
                    "direct_reasoning_mean_xy_diff_m"
                ],
                "direct_reasoning_final_xy_diff_m": r[
                    "direct_reasoning_final_xy_diff_m"
                ],

                "generated_intent_longitudinal": (
                    r["generated_reasoning_intent"].get("longitudinal")
                ),
                "generated_intent_lateral": (
                    r["generated_reasoning_intent"].get("lateral")
                ),
                "gt_intent_longitudinal": (
                    r["gt_reasoning_intent"].get("longitudinal")
                ),
                "gt_intent_lateral": (
                    r["gt_reasoning_intent"].get("lateral")
                ),

                "gt_motion_longitudinal": r["gt_motion"].get("longitudinal"),
                "gt_motion_lateral": r["gt_motion"].get("lateral"),
                "direct_motion_longitudinal": (
                    r["direct_motion"].get("longitudinal")
                ),
                "direct_motion_lateral": r["direct_motion"].get("lateral"),
                "reasoning_motion_longitudinal": (
                    r["reasoning_motion"].get("longitudinal")
                ),
                "reasoning_motion_lateral": (
                    r["reasoning_motion"].get("lateral")
                ),

                "generated_vs_direct_alignment": (
                    r["generated_reasoning_vs_direct_trajectory"].get("score")
                ),
                "generated_vs_reasoning_alignment": (
                    r["generated_reasoning_vs_reasoning_trajectory"].get(
                        "score"
                    )
                ),
                "gt_reasoning_vs_gt_alignment": (
                    r["gt_reasoning_vs_gt_trajectory"].get("score")
                ),

                "gt_route_mean_distance_m": (
                    r["gt_route"].get("mean_route_distance_m")
                ),
                "direct_route_mean_distance_m": (
                    r["direct_route"].get("mean_route_distance_m")
                ),
                "reasoning_route_mean_distance_m": (
                    r["reasoning_route"].get("mean_route_distance_m")
                ),

                "map_frame_detected": r.get("map_frame_detected"),
                "map_frame_used": r.get("map_frame_used"),
                "map_vector_count": r.get("map_vector_count"),
                "route_available": r.get("route_available"),

                "reasoning_generated_tokens": r.get(
                    "reasoning_generated_tokens"
                ),
                "reasoning_cached_tokens": r.get(
                    "reasoning_cached_tokens"
                ),

                "generated_reasoning": r.get("generated_reasoning", ""),
                "gt_reasoning": r.get("gt_reasoning", ""),
            })


# =============================================================================
# SUMMARY / COUNTEREXAMPLES
# =============================================================================

def quartile_summary(
    rows: Sequence[Dict[str, Any]],
    quality_key: str,
) -> List[Dict[str, Any]]:
    vals: List[Tuple[float, float]] = []

    for r in rows:
        q = finite_or_none(r.get(quality_key))
        d = finite_or_none(r.get("delta_ade_m"))

        if q is not None and d is not None:
            vals.append((q, d))

    if len(vals) < 8:
        return []

    vals.sort(key=lambda x: x[0])
    chunks = np.array_split(
        np.asarray(vals, dtype=np.float64),
        4,
    )

    out: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        if len(chunk) == 0:
            continue

        out.append({
            "quartile": i,
            "n": int(len(chunk)),
            "quality_mean": float(np.mean(chunk[:, 0])),
            "quality_min": float(np.min(chunk[:, 0])),
            "quality_max": float(np.max(chunk[:, 0])),
            "delta_ade_mean_m": float(np.mean(chunk[:, 1])),
            "reasoning_win_rate": float(
                np.mean(chunk[:, 1] < 0.0)
            ),
        })

    return out


def build_counterexamples(
    rows: Sequence[Dict[str, Any]],
    n: int = 20,
) -> Dict[str, Any]:
    valid = [
        r for r in rows
        if finite_or_none(r.get("reasoning_quality_mean")) is not None
    ]

    if not valid:
        return {
            "high_quality_but_degraded": [],
            "low_quality_but_improved": [],
        }

    qualities = np.asarray(
        [float(r["reasoning_quality_mean"]) for r in valid],
        dtype=np.float64,
    )

    high_cut = float(np.percentile(qualities, 75))
    low_cut = float(np.percentile(qualities, 25))

    high_bad = [
        r for r in valid
        if r["reasoning_quality_mean"] >= high_cut
        and r["delta_ade_m"] > 0
    ]
    high_bad.sort(key=lambda r: r["delta_ade_m"], reverse=True)

    low_good = [
        r for r in valid
        if r["reasoning_quality_mean"] <= low_cut
        and r["delta_ade_m"] < 0
    ]
    low_good.sort(key=lambda r: r["delta_ade_m"])

    def compact(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "index": r["index"],
            "id": r["id"],
            "reasoning_quality_mean": r["reasoning_quality_mean"],
            "reasoning_token_f1": r.get("reasoning_token_f1"),
            "reasoning_rouge_l_f1": r.get("reasoning_rouge_l_f1"),
            "delta_ade_m": r["delta_ade_m"],
            "direct_ade_m": r["direct_ade_m"],
            "reasoning_ade_m": r["reasoning_ade_m"],
            "generated_reasoning": r["generated_reasoning"],
            "gt_reasoning": r["gt_reasoning"],
        }

    return {
        "quality_definition": "mean(Token F1, ROUGE-L F1)",
        "high_quality_cutoff_q75": high_cut,
        "low_quality_cutoff_q25": low_cut,
        "high_quality_but_degraded": [
            compact(r) for r in high_bad[:n]
        ],
        "low_quality_but_improved": [
            compact(r) for r in low_good[:n]
        ],
    }


def build_summary(
    rows: Sequence[Dict[str, Any]],
    delta_threshold: float,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No result rows")

    d_ade = np.asarray(
        [r["direct_ade_m"] for r in rows],
        dtype=np.float64,
    )
    r_ade = np.asarray(
        [r["reasoning_ade_m"] for r in rows],
        dtype=np.float64,
    )
    d_fde = np.asarray(
        [r["direct_fde_m"] for r in rows],
        dtype=np.float64,
    )
    r_fde = np.asarray(
        [r["reasoning_fde_m"] for r in rows],
        dtype=np.float64,
    )
    d_head = np.asarray(
        [r["direct_heading_mae_rad"] for r in rows],
        dtype=np.float64,
    )
    r_head = np.asarray(
        [r["reasoning_heading_mae_rad"] for r in rows],
        dtype=np.float64,
    )

    delta_ade = r_ade - d_ade
    delta_fde = r_fde - d_fde
    delta_head = r_head - d_head

    f1 = [r.get("reasoning_token_f1") for r in rows]
    rouge = [r.get("reasoning_rouge_l_f1") for r in rows]
    quality = [r.get("reasoning_quality_mean") for r in rows]
    semantic = [
        r["reasoning_semantic_intent_match"].get("score")
        for r in rows
    ]
    gen_direct_align = [
        r["generated_reasoning_vs_direct_trajectory"].get("score")
        for r in rows
    ]
    gen_reason_align = [
        r["generated_reasoning_vs_reasoning_trajectory"].get("score")
        for r in rows
    ]
    gt_gt_align = [
        r["gt_reasoning_vs_gt_trajectory"].get("score")
        for r in rows
    ]

    classes = Counter(r["delta_class"] for r in rows)
    map_frames = Counter(r.get("map_frame_used") for r in rows)

    direct_route = [
        r["direct_route"].get("mean_route_distance_m")
        for r in rows
    ]
    reasoning_route = [
        r["reasoning_route"].get("mean_route_distance_m")
        for r in rows
    ]

    return {
        "samples": len(rows),
        "delta_definition": (
            "ΔADE = Reasoning ADE - Direct ADE; negative is improvement"
        ),
        "gain_definition": (
            "Gain = Direct ADE - Reasoning ADE; positive is improvement"
        ),
        "strong_delta_threshold_m": float(delta_threshold),

        "trajectory": {
            "direct": {
                "ade_m": float(np.mean(d_ade)),
                "fde_m": float(np.mean(d_fde)),
                "heading_mae_rad": float(np.mean(d_head)),
            },
            "reasoning": {
                "ade_m": float(np.mean(r_ade)),
                "fde_m": float(np.mean(r_fde)),
                "heading_mae_rad": float(np.mean(r_head)),
            },
            "delta_reasoning_minus_direct": {
                "ade_m": float(np.mean(delta_ade)),
                "fde_m": float(np.mean(delta_fde)),
                "heading_mae_rad": float(np.mean(delta_head)),
                "ade_relative_pct": float(
                    100.0 * np.mean(delta_ade) / np.mean(d_ade)
                ),
                "fde_relative_pct": float(
                    100.0 * np.mean(delta_fde) / np.mean(d_fde)
                ),
            },
            "reasoning_ade_win_rate": float(
                np.mean(delta_ade < 0.0)
            ),
            "reasoning_fde_win_rate": float(
                np.mean(delta_fde < 0.0)
            ),
        },

        "reasoning_quality": {
            "token_f1_mean": mean_valid(f1),
            "rouge_l_f1_mean": mean_valid(rouge),
            "combined_mean": mean_valid(quality),
            "semantic_intent_match_mean": mean_valid(semantic),
        },

        "correlation_with_delta_ade": {
            "interpretation": (
                "If better reasoning causes better Reasoning-DiT trajectory, "
                "correlation with ΔADE should tend negative."
            ),
            "token_f1": {
                "pearson": pearson_corr(f1, delta_ade),
                "spearman": spearman_corr(f1, delta_ade),
            },
            "rouge_l": {
                "pearson": pearson_corr(rouge, delta_ade),
                "spearman": spearman_corr(rouge, delta_ade),
            },
            "combined_quality": {
                "pearson": pearson_corr(quality, delta_ade),
                "spearman": spearman_corr(quality, delta_ade),
            },
            "semantic_intent_match": {
                "pearson": pearson_corr(semantic, delta_ade),
                "spearman": spearman_corr(semantic, delta_ade),
            },
        },

        "reasoning_trajectory_alignment_heuristic": {
            "warning": (
                "Auxiliary heuristic only. GT-reasoning vs GT-trajectory "
                "alignment indicates whether the heuristic is plausible "
                "for this dataset."
            ),
            "generated_vs_direct_mean": mean_valid(gen_direct_align),
            "generated_vs_reasoning_mean": mean_valid(gen_reason_align),
            "gt_reasoning_vs_gt_mean": mean_valid(gt_gt_align),
        },

        "route_proximity_diagnostic": {
            "warning": (
                "Distance to route_path only; not driveable-area compliance "
                "or NPS."
            ),
            "direct_mean_route_distance_m": mean_valid(direct_route),
            "reasoning_mean_route_distance_m": mean_valid(reasoning_route),
        },

        "sample_classes": {
            "reasoning_improved": int(
                classes.get("reasoning_improved", 0)
            ),
            "similar": int(classes.get("similar", 0)),
            "reasoning_degraded": int(
                classes.get("reasoning_degraded", 0)
            ),
        },

        "hdmap": {
            "map_frame_used_counts": dict(map_frames),
            "route_available_count": int(
                sum(bool(r.get("route_available")) for r in rows)
            ),
        },

        "quality_quartiles": {
            "token_f1": quartile_summary(
                rows,
                "reasoning_token_f1",
            ),
            "rouge_l": quartile_summary(
                rows,
                "reasoning_rouge_l_f1",
            ),
            "combined": quartile_summary(
                rows,
                "reasoning_quality_mean",
            ),
        },
    }


def save_report_md(
    path: Path,
    summary: Dict[str, Any],
    top_improved: Sequence[Dict[str, Any]],
    top_degraded: Sequence[Dict[str, Any]],
) -> None:
    t = summary["trajectory"]
    q = summary["reasoning_quality"]
    c = summary["correlation_with_delta_ade"]
    cls = summary["sample_classes"]
    align = summary["reasoning_trajectory_alignment_heuristic"]

    lines: List[str] = [
        "# Flow/DiT-only Reasoning-Trajectory-HDMap Analysis",
        "",
        "## Overall trajectory",
        "",
        "| Mode | ADE (m) | FDE (m) | Heading (rad) |",
        "|---|---:|---:|---:|",
        (
            f"| Direct | {t['direct']['ade_m']:.4f} | "
            f"{t['direct']['fde_m']:.4f} | "
            f"{t['direct']['heading_mae_rad']:.4f} |"
        ),
        (
            f"| Reasoning | {t['reasoning']['ade_m']:.4f} | "
            f"{t['reasoning']['fde_m']:.4f} | "
            f"{t['reasoning']['heading_mae_rad']:.4f} |"
        ),
        "",
        (
            f"- Mean ΔADE (Reasoning-Direct): "
            f"{t['delta_reasoning_minus_direct']['ade_m']:+.4f} m"
        ),
        (
            f"- ADE relative change: "
            f"{t['delta_reasoning_minus_direct']['ade_relative_pct']:+.2f}%"
        ),
        (
            f"- Reasoning ADE win rate: "
            f"{100*t['reasoning_ade_win_rate']:.2f}%"
        ),
        "",
        "## Reasoning quality",
        "",
        f"- Token F1 mean: {fmt(q['token_f1_mean'])}",
        f"- ROUGE-L mean: {fmt(q['rouge_l_f1_mean'])}",
        f"- Combined quality mean: {fmt(q['combined_mean'])}",
        (
            f"- Semantic intent match mean (heuristic): "
            f"{fmt(q['semantic_intent_match_mean'])}"
        ),
        "",
        "## Reasoning quality vs ΔADE correlation",
        "",
        "| Quality | Pearson | Spearman |",
        "|---|---:|---:|",
        (
            f"| Token F1 | {fmt(c['token_f1']['pearson'])} | "
            f"{fmt(c['token_f1']['spearman'])} |"
        ),
        (
            f"| ROUGE-L | {fmt(c['rouge_l']['pearson'])} | "
            f"{fmt(c['rouge_l']['spearman'])} |"
        ),
        (
            f"| Combined | {fmt(c['combined_quality']['pearson'])} | "
            f"{fmt(c['combined_quality']['spearman'])} |"
        ),
        (
            f"| Semantic intent | "
            f"{fmt(c['semantic_intent_match']['pearson'])} | "
            f"{fmt(c['semantic_intent_match']['spearman'])} |"
        ),
        "",
        (
            "**Interpretation:** negative correlation means higher reasoning "
            "quality tends to reduce Reasoning-DiT ADE relative to Direct DiT."
        ),
        "",
        "## Reasoning ↔ trajectory heuristic alignment",
        "",
        (
            f"- Generated reasoning vs Direct trajectory: "
            f"{fmt(align['generated_vs_direct_mean'])}"
        ),
        (
            f"- Generated reasoning vs Reasoning trajectory: "
            f"{fmt(align['generated_vs_reasoning_mean'])}"
        ),
        (
            f"- GT reasoning vs GT trajectory: "
            f"{fmt(align['gt_reasoning_vs_gt_mean'])}"
        ),
        "",
        "These are auxiliary heuristics, not primary metrics.",
        "",
        "## Strong sample classes",
        "",
        f"- Improved: {cls['reasoning_improved']}",
        f"- Similar: {cls['similar']}",
        f"- Degraded: {cls['reasoning_degraded']}",
        "",
        "## Top improved",
        "",
        "| Rank | ID | ΔADE | Direct ADE | Reasoning ADE | F1 | ROUGE-L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for i, r in enumerate(top_improved[:10], 1):
        lines.append(
            f"| {i} | {r['id']} | {r['delta_ade_m']:+.4f} | "
            f"{r['direct_ade_m']:.4f} | {r['reasoning_ade_m']:.4f} | "
            f"{fmt(r.get('reasoning_token_f1'))} | "
            f"{fmt(r.get('reasoning_rouge_l_f1'))} |"
        )

    lines += [
        "",
        "## Top degraded",
        "",
        "| Rank | ID | ΔADE | Direct ADE | Reasoning ADE | F1 | ROUGE-L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for i, r in enumerate(top_degraded[:10], 1):
        lines.append(
            f"| {i} | {r['id']} | {r['delta_ade_m']:+.4f} | "
            f"{r['direct_ade_m']:.4f} | {r['reasoning_ade_m']:.4f} | "
            f"{fmt(r.get('reasoning_token_f1'))} | "
            f"{fmt(r.get('reasoning_rouge_l_f1'))} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--raw-part2-root",
        type=Path,
        default=DEFAULT_RAW_PART2_ROOT,
    )
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--gpu-id", type=int, default=0)

    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="-1=all, positive=first N",
    )
    p.add_argument("--sample-index", type=int, default=None)
    p.add_argument("--sample-id", type=str, default=None)

    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument("--flow-seed", type=int, default=FLOW_SEED)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)

    p.add_argument(
        "--delta-threshold",
        type=float,
        default=DEFAULT_DELTA_THRESHOLD,
        help=(
            "Strong classification threshold. "
            "improved if ΔADE<-thr, degraded if ΔADE>+thr."
        ),
    )

    p.add_argument(
        "--map-frame",
        choices=("auto", "global", "ego"),
        default="auto",
    )
    p.add_argument(
        "--map-radius",
        type=float,
        default=50.0,
        help="Minimum HD-map BEV radius in meters.",
    )
    p.add_argument(
        "--max-map-elements",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--no-visuals",
        action="store_true",
        help="Run full numeric analysis without per-sample PNGs.",
    )
    p.add_argument(
        "--visual-limit",
        type=int,
        default=-1,
        help="-1=all selected rows, positive=first N selected rows.",
    )

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.delta_threshold < 0:
        raise ValueError("--delta-threshold must be >= 0")

    if args.map_radius <= 0:
        raise ValueError("--map-radius must be > 0")

    if args.max_map_elements < 1:
        raise ValueError("--max-map-elements must be >= 1")

    if args.solver_steps is not None and args.solver_steps < 1:
        raise ValueError("--solver-steps must be >= 1")

    for path in (
        args.dataset,
        args.raw_part2_root,
        args.cache_root,
        args.model_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dit_base.set_seed(args.seed)

    # -------------------------------------------------------------------------
    # Dataset / cache manifest
    # -------------------------------------------------------------------------
    rows = dit_base.load_rows(args.dataset)

    selected = select_rows(
        rows,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        limit=args.limit,
    )

    manifest = dit_base.load_cache_manifest(args.cache_root)
    cache_by_id = dit_base.manifest_by_id(manifest)

    missing = [
        row["id"]
        for _, row in selected
        if row["id"] not in cache_by_id
    ]

    if missing:
        raise RuntimeError(
            "Selected rows missing from test KV cache. "
            f"First missing IDs: {missing[:5]}"
        )

    # -------------------------------------------------------------------------
    # ONLY TWO DiTs
    # -------------------------------------------------------------------------
    flow_models: Dict[str, Dict[str, Any]] = {}

    for branch in BRANCHES:
        ckpt_path = dit_base.flow_checkpoint_path(
            args.model_root,
            branch,
        )

        model, ckpt, normalizer, ckpt_solver_steps = dit_base.load_flow(
            ckpt_path,
            device,
        )

        solver_steps = (
            int(args.solver_steps)
            if args.solver_steps is not None
            else int(ckpt_solver_steps)
        )

        flow_models[branch] = {
            "model": model,
            "ckpt": ckpt,
            "normalizer": normalizer,
            "solver_steps": solver_steps,
            "checkpoint": str(ckpt_path),
        }

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / timestamp

    for sub in (
        "",
        "reasoning_improved",
        "reasoning_degraded",
        "similar",
        "scatter",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 140)
    print(
        "Reasoning_VLM_v2 | Flow/DiT ONLY | "
        "REASONING ↔ TRAJECTORY ↔ HD-MAP ANALYSIS"
    )
    print("=" * 140)
    print("GPU              :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset          :", args.dataset)
    print("Samples          :", len(selected))
    print("Raw PART2        :", args.raw_part2_root)
    print("KV cache         :", args.cache_root)
    print("Direct DiT       :", flow_models["direct"]["checkpoint"])
    print("Reasoning DiT    :", flow_models["reasoning"]["checkpoint"])
    print(
        "Solver steps     :",
        flow_models["direct"]["solver_steps"],
        "/",
        flow_models["reasoning"]["solver_steps"],
    )
    print("Flow seed        :", args.flow_seed)
    print("Same noise       : YES, per sample / both branches")
    print("Map frame        :", args.map_frame)
    print("Transformer      : NOT LOADED")
    print("DFlash           : NOT USED")
    print("Output           :", run_dir)
    print("=" * 140)

    map_cache = HDMapClipCache(args.raw_part2_root)
    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------------
    for local_i, (dataset_index, row) in enumerate(selected, 1):
        sid = str(row["id"])
        cache_item = cache_by_id[sid]
        cache = dit_base.load_cache_record(cache_item)

        # HD map / raw frame is mandatory for this script.
        hdmap = load_sample_hdmap(
            row,
            clip_cache=map_cache,
            map_frame_arg=args.map_frame,
        )

        frame = hdmap["frame"]

        context = dit_base.extract_sample_context(
            row=row,
            frame=frame,
            cache=cache,
        )

        gt: Optional[np.ndarray] = None
        preds: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        # EXACT SAME initial Gaussian noise per sample.
        sample_seed = int(args.flow_seed + dataset_index)

        for branch in BRANCHES:
            inputs, branch_gt = dit_base.action_input_from_record(
                cache=cache,
                cache_item=cache_item,
                branch=branch,
                device=device,
            )

            branch_gt = np.asarray(branch_gt, dtype=np.float64)

            if gt is None:
                gt = branch_gt
            elif not np.allclose(
                gt,
                branch_gt,
                atol=0.0,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"GT mismatch between branches: id={sid}"
                )

            info = flow_models[branch]

            pred = dit_base.run_flow_one(
                model=info["model"],
                normalizer=info["normalizer"],
                solver_steps=info["solver_steps"],
                inputs=inputs,
                sample_seed=sample_seed,
            )

            preds[branch] = np.asarray(pred, dtype=np.float64)
            metrics[branch] = dit_base.sample_metrics(
                preds[branch],
                gt,
            )

            del inputs

        assert gt is not None

        direct = preds["direct"]
        reasoning = preds["reasoning"]

        dm = metrics["direct"]
        rm = metrics["reasoning"]

        delta_ade = float(rm["ade_m"] - dm["ade_m"])
        delta_fde = float(rm["fde_m"] - dm["fde_m"])
        delta_head = float(
            rm["heading_mae_rad"] - dm["heading_mae_rad"]
        )

        delta_class = classify_delta(
            delta_ade,
            args.delta_threshold,
        )

        generated_reasoning = str(
            context.get("generated_reasoning", "") or ""
        )
        gt_reasoning = str(
            context.get("gt_reasoning", "") or ""
        )

        text_metrics = context.get("text_metrics", {}) or {}
        token_f1 = finite_or_none(text_metrics.get("token_f1"))
        rouge_l = finite_or_none(text_metrics.get("rouge_l_f1"))

        quality_parts = [
            x for x in (token_f1, rouge_l)
            if x is not None
        ]
        reasoning_quality_mean = (
            float(np.mean(quality_parts))
            if quality_parts
            else None
        )

        current_speed = finite_or_none(row.get("speed_mps"))
        if current_speed is None:
            current_speed = finite_or_none(cache.get("speed_mps"))

        # Semantic intent.
        generated_intent = extract_reasoning_intent(
            generated_reasoning
        )
        gt_intent = extract_reasoning_intent(gt_reasoning)

        semantic_intent_match = intent_match(
            generated_intent,
            gt_intent,
        )

        # Trajectory motion summaries.
        gt_motion = trajectory_motion_summary(
            gt,
            current_speed,
        )
        direct_motion = trajectory_motion_summary(
            direct,
            current_speed,
        )
        reasoning_motion = trajectory_motion_summary(
            reasoning,
            current_speed,
        )

        gen_vs_direct = intent_vs_trajectory(
            generated_intent,
            direct_motion,
        )
        gen_vs_reason = intent_vs_trajectory(
            generated_intent,
            reasoning_motion,
        )
        gt_vs_gt = intent_vs_trajectory(
            gt_intent,
            gt_motion,
        )

        # Direct / Reasoning trajectory pairwise change.
        xy_diff = np.linalg.norm(
            direct[:, :2] - reasoning[:, :2],
            axis=1,
        )

        # Route diagnostics.
        route_ego = hdmap["route_ego"]
        gt_route = trajectory_route_metrics(gt, route_ego)
        direct_route = trajectory_route_metrics(
            direct,
            route_ego,
        )
        reasoning_route = trajectory_route_metrics(
            reasoning,
            route_ego,
        )

        result = {
            "index": int(dataset_index),
            "id": sid,
            "clip": str(row.get("clip", "")),
            "mission_command": str(
                context.get("command", "")
                or row.get("mission_command", "")
                or row.get("command", "")
            ),
            "current_speed_mps": current_speed,

            "reasoning_generated_tokens": context.get(
                "reasoning_generated_tokens"
            ),
            "reasoning_cached_tokens": context.get(
                "reasoning_cached_tokens"
            ),

            "generated_reasoning": generated_reasoning,
            "gt_reasoning": gt_reasoning,

            "reasoning_token_f1": token_f1,
            "reasoning_rouge_l_f1": rouge_l,
            "reasoning_quality_mean": reasoning_quality_mean,

            "generated_reasoning_intent": generated_intent,
            "gt_reasoning_intent": gt_intent,
            "reasoning_semantic_intent_match": semantic_intent_match,

            "gt_motion": gt_motion,
            "direct_motion": direct_motion,
            "reasoning_motion": reasoning_motion,

            "generated_reasoning_vs_direct_trajectory": gen_vs_direct,
            "generated_reasoning_vs_reasoning_trajectory": gen_vs_reason,
            "gt_reasoning_vs_gt_trajectory": gt_vs_gt,

            "direct_ade_m": float(dm["ade_m"]),
            "direct_fde_m": float(dm["fde_m"]),
            "direct_heading_mae_rad": float(
                dm["heading_mae_rad"]
            ),

            "reasoning_ade_m": float(rm["ade_m"]),
            "reasoning_fde_m": float(rm["fde_m"]),
            "reasoning_heading_mae_rad": float(
                rm["heading_mae_rad"]
            ),

            "delta_ade_m": delta_ade,
            "reasoning_gain_ade_m": -delta_ade,
            "delta_fde_m": delta_fde,
            "delta_heading_rad": delta_head,
            "delta_class": delta_class,

            "direct_reasoning_mean_xy_diff_m": float(
                np.mean(xy_diff)
            ),
            "direct_reasoning_final_xy_diff_m": float(
                xy_diff[-1]
            ),

            "gt_route": gt_route,
            "direct_route": direct_route,
            "reasoning_route": reasoning_route,

            "map_frame_detected": hdmap["map_frame_detected"],
            "map_frame_used": hdmap["map_frame_used"],
            "map_frame_stats": hdmap["map_frame_stats"],
            "map_vector_count": int(
                len(hdmap["map_geoms_raw"])
            ),
            "route_available": bool(
                route_ego is not None and len(route_ego) >= 2
            ),
            "map_path": str(hdmap["map_path"]),
            "ego_pose": [
                float(x) for x in hdmap["ego_pose"]
            ],

            "flow_sample_seed": sample_seed,

            "gt_trajectory": gt.tolist(),
            "direct_trajectory": direct.tolist(),
            "reasoning_trajectory": reasoning.tolist(),
        }

        results.append(result)

        # ---------------------------------------------------------------------
        # PNG
        # ---------------------------------------------------------------------
        visualize = not args.no_visuals

        if args.visual_limit >= 0:
            visualize = (
                visualize
                and local_i <= args.visual_limit
            )

        if visualize:
            png_name = (
                f"{dataset_index:04d}_{safe_name(sid)}_"
                f"dADE_{delta_ade:+.3f}_"
                f"F1_{fmt(token_f1, 3)}.png"
            )

            save_sample_visualization(
                output_path=run_dir / delta_class / png_name,
                row=row,
                hdmap=hdmap,
                result=result,
                gt=gt,
                direct_pred=direct,
                reasoning_pred=reasoning,
                map_radius=args.map_radius,
                max_map_elements=args.max_map_elements,
            )

        print(
            f"[{local_i:03d}/{len(selected):03d}] "
            f"id={sid} | "
            f"F1={fmt(token_f1, 3):>5s} | "
            f"ROUGE={fmt(rouge_l, 3):>5s} | "
            f"D_ADE={dm['ade_m']:.3f} | "
            f"R_ADE={rm['ade_m']:.3f} | "
            f"ΔADE={delta_ade:+.3f} | "
            f"{delta_class} | "
            f"map={hdmap['map_frame_used']}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Save per-sample
    # -------------------------------------------------------------------------
    write_jsonl(run_dir / "samples.jsonl", results)
    save_csv(run_dir / "samples.csv", results)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    save_scatter(
        run_dir / "scatter" / "token_f1_vs_delta_ade.png",
        results,
        "reasoning_token_f1",
        "Reasoning Token F1",
    )
    save_scatter(
        run_dir / "scatter" / "rouge_l_vs_delta_ade.png",
        results,
        "reasoning_rouge_l_f1",
        "Reasoning ROUGE-L F1",
    )

    # Flatten semantic score for plotting.
    for r in results:
        r["reasoning_semantic_intent_score"] = (
            r["reasoning_semantic_intent_match"].get("score")
        )

    save_scatter(
        run_dir / "scatter" / "semantic_intent_vs_delta_ade.png",
        results,
        "reasoning_semantic_intent_score",
        "Reasoning Semantic Intent Match",
    )

    save_delta_histogram(
        run_dir / "scatter" / "delta_ade_histogram.png",
        results,
    )

    # Re-save after semantic flat key addition.
    write_jsonl(run_dir / "samples.jsonl", results)
    save_csv(run_dir / "samples.csv", results)

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------
    by_delta = sorted(
        results,
        key=lambda r: r["delta_ade_m"],
    )

    top_improved = by_delta[: min(20, len(by_delta))]
    top_degraded = list(
        reversed(by_delta[-min(20, len(by_delta)):])
    )

    def compact_rank(
        rows_: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out = []

        for rank, r in enumerate(rows_, 1):
            out.append({
                "rank": rank,
                "index": r["index"],
                "id": r["id"],
                "delta_ade_m": r["delta_ade_m"],
                "direct_ade_m": r["direct_ade_m"],
                "reasoning_ade_m": r["reasoning_ade_m"],
                "reasoning_token_f1": r.get(
                    "reasoning_token_f1"
                ),
                "reasoning_rouge_l_f1": r.get(
                    "reasoning_rouge_l_f1"
                ),
                "reasoning_quality_mean": r.get(
                    "reasoning_quality_mean"
                ),
                "generated_reasoning": r.get(
                    "generated_reasoning"
                ),
                "gt_reasoning": r.get("gt_reasoning"),
            })

        return out

    write_json(
        run_dir / "top_20_improved.json",
        compact_rank(top_improved),
    )
    write_json(
        run_dir / "top_20_degraded.json",
        compact_rank(top_degraded),
    )

    counterexamples = build_counterexamples(results)
    write_json(
        run_dir / "counterexamples.json",
        counterexamples,
    )

    # -------------------------------------------------------------------------
    # Summary / report
    # -------------------------------------------------------------------------
    summary = build_summary(
        results,
        delta_threshold=args.delta_threshold,
    )

    summary["runtime"] = {
        "dataset": str(args.dataset),
        "raw_part2_root": str(args.raw_part2_root),
        "cache_root": str(args.cache_root),
        "model_root": str(args.model_root),

        "direct_checkpoint": flow_models["direct"]["checkpoint"],
        "reasoning_checkpoint": flow_models["reasoning"]["checkpoint"],

        "direct_epoch": int(
            flow_models["direct"]["ckpt"].get("epoch", -1)
        ),
        "reasoning_epoch": int(
            flow_models["reasoning"]["ckpt"].get("epoch", -1)
        ),

        "direct_solver_steps": int(
            flow_models["direct"]["solver_steps"]
        ),
        "reasoning_solver_steps": int(
            flow_models["reasoning"]["solver_steps"]
        ),

        "flow_seed_base": int(args.flow_seed),
        "same_noise_per_sample": True,
        "map_frame_arg": args.map_frame,
        "transformer_loaded": False,
        "dflash_used": False,
    }

    write_json(run_dir / "summary.json", summary)

    save_report_md(
        run_dir / "report.md",
        summary,
        top_improved,
        top_degraded,
    )

    # -------------------------------------------------------------------------
    # Console final
    # -------------------------------------------------------------------------
    traj = summary["trajectory"]
    q = summary["reasoning_quality"]
    corr = summary["correlation_with_delta_ade"]
    cls = summary["sample_classes"]

    print("\n" + "=" * 140)
    print("FINAL | DiT-ONLY REASONING ↔ TRAJECTORY ANALYSIS")
    print("=" * 140)

    print(
        f"{'MODE':<16}"
        f"{'ADE(m)':>12}"
        f"{'FDE(m)':>12}"
        f"{'HEAD(rad)':>14}"
    )
    print("-" * 54)
    print(
        f"{'Direct':<16}"
        f"{traj['direct']['ade_m']:>12.4f}"
        f"{traj['direct']['fde_m']:>12.4f}"
        f"{traj['direct']['heading_mae_rad']:>14.4f}"
    )
    print(
        f"{'Reasoning':<16}"
        f"{traj['reasoning']['ade_m']:>12.4f}"
        f"{traj['reasoning']['fde_m']:>12.4f}"
        f"{traj['reasoning']['heading_mae_rad']:>14.4f}"
    )

    delta = traj["delta_reasoning_minus_direct"]

    print(
        f"{'Delta(R-D)':<16}"
        f"{delta['ade_m']:>+12.4f}"
        f"{delta['fde_m']:>+12.4f}"
        f"{delta['heading_mae_rad']:>+14.4f}"
    )

    print()
    print(
        f"ADE relative change    : {delta['ade_relative_pct']:+.2f}%"
    )
    print(
        f"Reasoning ADE win rate : "
        f"{100*traj['reasoning_ade_win_rate']:.2f}%"
    )
    print(
        f"Strong improved        : {cls['reasoning_improved']}"
    )
    print(f"Similar                : {cls['similar']}")
    print(
        f"Strong degraded        : {cls['reasoning_degraded']}"
    )

    print("\nREASONING QUALITY")
    print("Token F1 mean          :", fmt(q["token_f1_mean"], 4))
    print("ROUGE-L mean           :", fmt(q["rouge_l_f1_mean"], 4))
    print("Combined mean          :", fmt(q["combined_mean"], 4))
    print(
        "Semantic intent match  :",
        fmt(q["semantic_intent_match_mean"], 4),
    )

    print("\nQUALITY -> ΔADE CORRELATION")
    print(
        "Token F1 Pearson       :",
        fmt(corr["token_f1"]["pearson"], 4),
    )
    print(
        "Token F1 Spearman      :",
        fmt(corr["token_f1"]["spearman"], 4),
    )
    print(
        "ROUGE-L Pearson        :",
        fmt(corr["rouge_l"]["pearson"], 4),
    )
    print(
        "ROUGE-L Spearman       :",
        fmt(corr["rouge_l"]["spearman"], 4),
    )
    print(
        "Combined Pearson       :",
        fmt(corr["combined_quality"]["pearson"], 4),
    )
    print(
        "Combined Spearman      :",
        fmt(corr["combined_quality"]["spearman"], 4),
    )

    print(
        "\nInterpretation: negative correlation means better reasoning "
        "tends to lower Reasoning-DiT ADE relative to Direct DiT."
    )

    print("\nOUTPUT")
    print("Run dir                :", run_dir)
    print("CSV                    :", run_dir / "samples.csv")
    print("Summary                :", run_dir / "summary.json")
    print("Report                 :", run_dir / "report.md")
    print(
        "Counterexamples        :",
        run_dir / "counterexamples.json",
    )
    print("=" * 140)


if __name__ == "__main__":
    main()
