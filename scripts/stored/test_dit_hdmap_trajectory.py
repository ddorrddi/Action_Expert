#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning_VLM_v2 + Flow/DiT | nuReasoning PART2 HD-MAP TRAJECTORY VISUAL TEST
================================================================================

Purpose
-------
Visualize how the trained Flow/DiT Action Expert generates a 10-step trajectory
on top of the original nuReasoning HD map.

This script intentionally tests ONLY the two Flow/DiT checkpoints:

    direct_flow_dit
    reasoning_flow_dit

Transformer Action Experts are not loaded.

Inputs reused from the existing E2E test
----------------------------------------
1) Held-out PART2 dataset rows
   /home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl

2) Already-built Reasoning_VLM_v2 test KV cache
   /home/lhh/lab/Action_Expert/dataset/ActionExpert10_v2/part2_test_kv_cache

3) Flow/DiT checkpoints
   /home/lhh/lab/models/action_expert/reasoning_vlm_v2_10model/flow/
       direct_flow_dit/best.pt
       reasoning_flow_dit/best.pt

4) Original nuReasoning PART2 clip folders
   /media/HDD/nuReasoning/train/part_2/<clip>/
       metadata.json
       map.pkl
       ego_state/*.pkl

What is drawn
-------------
- nuReasoning static vector HD map in the CURRENT EGO FRAME
- route_path from metadata.json when numeric route coordinates are available
- ground-truth trajectory [10, 3]
- direct Flow/DiT trajectory
- reasoning Flow/DiT trajectory
- mission command
- generated VLM reasoning + GT reasoning
- per-sample reasoning Token F1 / ROUGE-L F1
- per-sample DiT ADE / FDE / heading error

Coordinate handling
-------------------
The Action Expert target/prediction is already ego-relative [x, y, yaw].
The map.pkl representation can be stored in either global/map coordinates or
local ego-like coordinates depending on dataset version. Therefore this script:

1) loads map geometry from map.pkl,
2) estimates whether the map is global or local,
3) if global, transforms map points into the current ego frame,
4) leaves DiT / GT trajectories untouched in ego coordinates.

The detected map frame is printed for every sample. You can override it with:

    --map-frame global
    --map-frame ego

Examples
--------
# Recommended: compare both DiTs for first 10 PART2 samples
python3 test_dit_hdmap_trajectory.py \
    --gpu-id 0 \
    --limit 10

# Only reasoning Flow/DiT
python3 test_dit_hdmap_trajectory.py \
    --gpu-id 0 \
    --branches reasoning \
    --limit 20

# Draw one exact dataset row
python3 test_dit_hdmap_trajectory.py \
    --gpu-id 0 \
    --sample-index 37

# If automatic map-coordinate detection is wrong
python3 test_dit_hdmap_trajectory.py \
    --gpu-id 0 \
    --sample-index 37 \
    --map-frame global

Notes
-----
- This script does NOT rebuild the VLM cache. It is a DiT-only visualization.
- If the test KV cache is missing, first run the existing AR-only E2E test once
  so that part2_test_kv_cache/manifest.jsonl exists.
- Both Flow branches use the SAME Gaussian seed for each sample, so differences
  are caused by the conditioning branch rather than different initial noise.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import random
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_ROOT / "scripts"
VLM_SCRIPT_DIR = Path("/home/lhh/lab/VLM/scripts")

for _p in (SCRIPT_DIR, ACTION_ROOT, ACTION_SCRIPT_DIR, VLM_SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_DATASET = Path("/home/lhh/lab/DFlash/dataset/part2_e2e_300.jsonl")
RAW_PART2_ROOT = Path("/media/HDD/nuReasoning/train/part_2")

DEFAULT_CACHE_ROOT = Path(
    "/home/lhh/lab/Action_Expert/dataset/ActionExpert10_v2/part2_test_kv_cache"
)

MODEL_ROOT = Path(
    "/home/lhh/lab/models/action_expert/reasoning_vlm_v2_10model"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/lhh/lab/E2E/Result/dit_hdmap_trajectory"
)


# =============================================================================
# ACTION MODEL IMPORTS
# =============================================================================

try:
    from scripts.stored.action_model_flow_dit import (
        TrajectoryNormalizer,
        build_flow_dit,
        euler_sample,
    )
except ImportError:
    from scripts.stored.action_model_flow_dit import (
        TrajectoryNormalizer,
        build_flow_dit,
        euler_sample,
    )

try:
    import reasoning_v2_core as vlm_v2_core
except Exception:
    vlm_v2_core = None


# =============================================================================
# CONFIG
# =============================================================================

BRANCHES = ("direct", "reasoning")
NUM_STEPS = 10
FLOW_TEST_SEED = 20260841
SEED = 20260823
SCRIPT_VERSION = "2026-08-31-clean-info-v2"

# Map geometry categories used only for visualization style.
CATEGORY_PRIORITY = (
    "crosswalk",
    "stop",
    "traffic_light",
    "intersection",
    "roadblock",
    "connector",
    "boundary",
    "baseline",
    "lane",
    "map",
)


# =============================================================================
# BASIC HELPERS
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSON parse failed {path}:{n}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(text: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text))
    return out[:120] if out else "sample"


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# TEXT / REASONING METRICS
# =============================================================================

def normalize_text(text: Any) -> str:
    """Same normalization used by the current Reasoning_VLM_v2 evaluation."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.+-]", "", text, flags=re.UNICODE)
    return text.strip()


def token_f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    pc = Counter(p)
    gc = Counter(g)
    common = sum((pc & gc).values())

    if common <= 0:
        return 0.0

    precision = common / len(p)
    recall = common / len(g)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def rouge_l_f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    dp = [0] * (len(g) + 1)
    for x in p:
        prev = 0
        for j, y in enumerate(g, 1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old

    lcs = dp[-1]
    if lcs <= 0:
        return 0.0

    precision = lcs / len(p)
    recall = lcs / len(g)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def compact_text(text: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def wrapped_label(label: str, text: Any, width: int = 118) -> str:
    value = str(text).strip() or "N/A"
    prefix = f"{label}: "
    wrapped = textwrap.wrap(
        value,
        width=max(20, width - len(prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return prefix + "N/A"
    lines = [prefix + wrapped[0]]
    indent = " " * len(prefix)
    lines.extend(indent + x for x in wrapped[1:])
    return "\n".join(lines)


# =============================================================================
# nuReasoning PICKLE COMPATIBILITY
# =============================================================================

_LEGACY_PICKLE_CLASS_CACHE: Dict[Tuple[str, str], type] = {}


def _legacy_pickle_class(module: str, name: str) -> type:
    key = (str(module), str(name))
    if key not in _LEGACY_PICKLE_CLASS_CACHE:
        # Accept arbitrary constructor arguments as well.  This is useful if a
        # legacy data_schema object was serialized through a reduce/enum-like
        # path rather than as a plain dataclass.  We only need its stored
        # attributes/geometry for visualization, not the original methods.
        def _dummy_new(cls, *args, **kwargs):
            return object.__new__(cls)

        def _dummy_init(self, *args, **kwargs):
            pass

        cls = type(
            str(name),
            (),
            {"__new__": _dummy_new, "__init__": _dummy_init},
        )
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
# DATASET / METADATA RESOLUTION
# =============================================================================


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)

    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Missing row id")
    row["id"] = sid

    clip = str(row.get("clip", "")).strip()
    if not clip:
        raise ValueError(f"Missing clip: id={sid}")
    row["clip"] = clip

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (NUM_STEPS, 3):
        raise ValueError(
            f"trajectory must be [10,3]: id={sid}, got={gt.shape}"
        )
    if not np.isfinite(gt).all():
        raise ValueError(f"Non-finite trajectory: id={sid}")
    row["trajectory"] = gt.tolist()

    return row


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = [normalize_row(x) for x in read_jsonl(path)]
    ids = [x["id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sample IDs in dataset")
    return rows


def load_clip_metadata(raw_root: Path, clip: str) -> Tuple[Path, Dict[str, Any]]:
    clip_dir = raw_root / clip
    metadata_path = clip_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return clip_dir, metadata


def match_raw_frame(
    row: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    sid = str(row["id"])

    try:
        frame_index = int(row.get("frame_index", -1))
    except Exception:
        frame_index = -1

    frames = metadata.get("frames", [])
    if not isinstance(frames, list):
        raise RuntimeError("metadata.frames is not a list")

    # 1) Exact frame token / row id.
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        token = str(frame.get("token", "")).strip()
        if token and token == sid:
            return frame

    # 2) Explicit frame_index from prepared row.
    if frame_index >= 0:
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            try:
                idx = int(frame.get("frame_index", -999999))
            except Exception:
                continue
            if idx == frame_index:
                return frame

    raise RuntimeError(
        f"Could not match raw frame: id={sid}, "
        f"clip={row.get('clip')}, frame_index={frame_index}"
    )


def resolve_clip_file(clip_dir: Path, value: Any, fallback: str) -> Path:
    if value is None or str(value).strip() == "":
        path = clip_dir / fallback
    else:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = clip_dir / path

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_ego_path(clip_dir: Path, frame: Dict[str, Any]) -> Path:
    value = frame.get("ego_state")
    if not value:
        raise KeyError("Matched metadata frame does not contain ego_state")
    return resolve_clip_file(clip_dir, value, "")


def resolve_map_path(
    clip_dir: Path,
    metadata: Dict[str, Any],
) -> Path:
    # Official nuReasoning metadata uses map_annotation: "map.pkl".
    value = metadata.get("map_annotation", "map.pkl")
    return resolve_clip_file(clip_dir, value, "map.pkl")


# =============================================================================
# EGO POSE EXTRACTION
# =============================================================================


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _extract_xy_from_obj(obj: Any) -> Optional[Tuple[float, float]]:
    if obj is None:
        return None

    if isinstance(obj, dict):
        x = _finite_float(obj.get("x"))
        y = _finite_float(obj.get("y"))
        if x is not None and y is not None:
            return x, y

    x = _finite_float(getattr(obj, "x", None))
    y = _finite_float(getattr(obj, "y", None))
    if x is not None and y is not None:
        return x, y

    return None


def extract_ego_pose(ego_state: Any) -> Tuple[float, float, float]:
    """Return global/map-frame (x, y, heading) for the current ego state."""

    # Preferred: same helper already used by the user's Reasoning_VLM_v2 code.
    if vlm_v2_core is not None:
        try:
            x, y, heading = vlm_v2_core.extract_pose(ego_state)
            x = float(x)
            y = float(y)
            heading = float(heading)
            if all(math.isfinite(v) for v in (x, y, heading)):
                return x, y, heading
        except Exception:
            pass

    # Generic fallbacks for data_schema variants.
    candidates = [
        ego_state,
        getattr(ego_state, "pose", None),
        getattr(ego_state, "rear_axle", None),
        getattr(ego_state, "center", None),
        getattr(ego_state, "ego_pose", None),
    ]

    for obj in candidates:
        if obj is None:
            continue

        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj).reshape(-1)
            if arr.size >= 3 and np.isfinite(arr[:3]).all():
                return float(arr[0]), float(arr[1]), float(arr[2])

        xy = _extract_xy_from_obj(obj)
        if xy is None:
            continue

        heading = None
        if isinstance(obj, dict):
            for key in ("heading", "yaw", "heading_rad", "yaw_rad"):
                heading = _finite_float(obj.get(key))
                if heading is not None:
                    break
        else:
            for key in ("heading", "yaw", "heading_rad", "yaw_rad"):
                heading = _finite_float(getattr(obj, key, None))
                if heading is not None:
                    break

        if heading is not None:
            return float(xy[0]), float(xy[1]), float(heading)

    raise RuntimeError(
        "Could not extract ego (x, y, heading) from ego_state.pkl. "
        "reasoning_v2_core.extract_pose() also failed/unavailable."
    )


# =============================================================================
# MAP GEOMETRY EXTRACTION
# =============================================================================

@dataclass
class MapGeometry:
    category: str
    name: str
    xy: np.ndarray


def _numeric_xy_array(value: Any) -> Optional[np.ndarray]:
    """Convert common Nx2/Nx3 coordinate containers to finite Nx2 arrays."""
    if value is None:
        return None

    # Numpy / tensor first.
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size in (2, 3):
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] in (2, 3, 4) and arr.shape[0] >= 1:
            try:
                arr = arr.astype(np.float64, copy=False)[:, :2]
            except Exception:
                return None
            if np.isfinite(arr).all():
                return arr
        return None

    # Numeric list/tuple such as [[x,y], ...].
    if isinstance(value, (list, tuple)) and value:
        try:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1 and arr.size in (2, 3):
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] in (2, 3, 4) and arr.shape[0] >= 1:
                arr = arr[:, :2]
                if np.isfinite(arr).all():
                    return arr
        except Exception:
            pass

        # List of point-like objects with x/y.
        pts: List[Tuple[float, float]] = []
        for item in value:
            xy = _extract_xy_from_obj(item)
            if xy is None:
                pts = []
                break
            pts.append(xy)
        if pts:
            arr = np.asarray(pts, dtype=np.float64)
            if np.isfinite(arr).all():
                return arr

    return None


def _geometry_coords(value: Any) -> List[np.ndarray]:
    """Extract coordinates from shapely-like geometry without importing shapely."""
    out: List[np.ndarray] = []

    if value is None:
        return out

    # LineString / LinearRing style.
    try:
        coords = getattr(value, "coords")
        arr = _numeric_xy_array(list(coords))
        if arr is not None and len(arr) >= 2:
            out.append(arr)
            return out
    except Exception:
        pass

    # Polygon style.
    try:
        exterior = getattr(value, "exterior")
        coords = getattr(exterior, "coords")
        arr = _numeric_xy_array(list(coords))
        if arr is not None and len(arr) >= 3:
            out.append(arr)

        interiors = getattr(value, "interiors", [])
        for ring in interiors:
            try:
                ring_arr = _numeric_xy_array(list(ring.coords))
            except Exception:
                ring_arr = None
            if ring_arr is not None and len(ring_arr) >= 3:
                out.append(ring_arr)

        if out:
            return out
    except Exception:
        pass

    # MultiLineString / MultiPolygon style.
    try:
        geoms = list(getattr(value, "geoms"))
        for g in geoms:
            out.extend(_geometry_coords(g))
        if out:
            return out
    except Exception:
        pass

    return out


def category_from_path(path: str) -> str:
    p = path.lower()
    if "crosswalk" in p:
        return "crosswalk"
    if "traffic_light" in p or "trafficlight" in p:
        return "traffic_light"
    if "stop" in p:
        return "stop"
    if "intersection" in p:
        return "intersection"
    if "road_block" in p or "roadblock" in p:
        return "roadblock"
    if "connector" in p:
        return "connector"
    if "boundar" in p:
        return "boundary"
    if "baseline" in p or "centerline" in p or "center_line" in p:
        return "baseline"
    if "lane" in p:
        return "lane"
    return "map"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(
        value,
        (str, bytes, bool, int, float, complex, np.number),
    )


def extract_map_geometries(
    map_obj: Any,
    max_depth: int = 8,
    max_items: int = 20000,
) -> List[MapGeometry]:
    """
    Recursively extract vector-map geometry from different nuReasoning
    data_schema versions.

    The public dataset description states map.pkl contains lanes, baseline paths,
    boundaries, crosswalks, intersections, stop polygons, road blocks, traffic
    lights and lane connectors. The exact dataclass nesting may vary, so this is
    deliberately schema-tolerant rather than hard-coding one release.
    """

    found: List[MapGeometry] = []
    seen_objects: set[int] = set()
    seen_geometry: set[str] = set()
    visited_count = 0

    def add_geometry(path: str, arr: np.ndarray) -> None:
        if arr is None or len(arr) < 2:
            return
        arr = np.asarray(arr, dtype=np.float64)[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) < 2:
            return

        # Avoid absurd non-coordinate arrays and duplicate geometry.
        if np.max(np.abs(arr)) > 1e8:
            return

        digest = hashlib.sha1(
            np.round(arr, 3).astype(np.float32).tobytes()
        ).hexdigest()
        if digest in seen_geometry:
            return
        seen_geometry.add(digest)

        found.append(
            MapGeometry(
                category=category_from_path(path),
                name=path,
                xy=arr,
            )
        )

    def visit(value: Any, path: str, depth: int) -> None:
        nonlocal visited_count

        if depth > max_depth or visited_count >= max_items:
            return
        if _is_scalar(value):
            return

        visited_count += 1

        # Direct numeric coordinates.
        arr = _numeric_xy_array(value)
        if arr is not None and len(arr) >= 2:
            add_geometry(path, arr)
            return

        # Shapely-like geometry.
        geoms = _geometry_coords(value)
        if geoms:
            for i, geom_arr in enumerate(geoms):
                add_geometry(f"{path}.geometry[{i}]", geom_arr)
            return

        # Cycle protection for containers / objects.
        try:
            oid = id(value)
            if oid in seen_objects:
                return
            seen_objects.add(oid)
        except Exception:
            pass

        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}", depth + 1)
            return

        if isinstance(value, (list, tuple, set)):
            for i, child in enumerate(value):
                visit(child, f"{path}[{i}]", depth + 1)
            return

        # Dataclass / normal Python object.
        try:
            attrs = vars(value)
        except Exception:
            attrs = None

        if isinstance(attrs, dict):
            for key, child in attrs.items():
                if str(key).startswith("__"):
                    continue
                visit(child, f"{path}.{key}", depth + 1)
            return

        # Last-resort well-known geometry attributes.
        for key in (
            "points",
            "vertices",
            "coords",
            "polyline",
            "polygon",
            "geometry",
            "linestring",
            "baseline_path",
            "left_boundary",
            "right_boundary",
        ):
            try:
                child = getattr(value, key)
            except Exception:
                continue
            visit(child, f"{path}.{key}", depth + 1)

    visit(map_obj, "map", 0)
    return found


def summarize_map_schema(map_obj: Any) -> Dict[str, Any]:
    if isinstance(map_obj, dict):
        top = sorted(str(x) for x in map_obj.keys())
    else:
        try:
            top = sorted(str(x) for x in vars(map_obj).keys())
        except Exception:
            top = []

    return {
        "type": f"{type(map_obj).__module__}.{type(map_obj).__name__}",
        "top_level_fields": top,
    }


# =============================================================================
# COORDINATE TRANSFORMS
# =============================================================================


def global_to_ego_xy(
    xy: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    dx = xy[:, 0] - ego_x
    dy = xy[:, 1] - ego_y

    c = math.cos(ego_heading)
    s = math.sin(ego_heading)

    # Standard SE(2) inverse transform: world -> ego.
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return np.column_stack([x_local, y_local])


def estimate_map_frame(
    geoms: Sequence[MapGeometry],
    ego_x: float,
    ego_y: float,
) -> Tuple[str, Dict[str, float]]:
    """
    Decide whether map coordinates look global or already local.

    We compare the lower-distance quantile to current ego global position and
    to the origin. Local maps should cluster near (0,0); global maps should
    cluster near (ego_x, ego_y).
    """
    chunks = [g.xy for g in geoms if len(g.xy) >= 2]
    if not chunks:
        return "unknown", {
            "q10_to_ego_m": float("nan"),
            "q10_to_origin_m": float("nan"),
        }

    pts = np.concatenate(chunks, axis=0)

    # Bound computation cost for very large maps.
    if len(pts) > 200000:
        idx = np.linspace(0, len(pts) - 1, 200000).astype(np.int64)
        pts = pts[idx]

    d_ego = np.linalg.norm(pts - np.array([[ego_x, ego_y]]), axis=1)
    d_zero = np.linalg.norm(pts, axis=1)

    q_ego = float(np.percentile(d_ego, 10))
    q_zero = float(np.percentile(d_zero, 10))

    # Strong preference when one coordinate frame is clearly closer.
    if q_ego < 500.0 and q_ego < 0.45 * max(q_zero, 1e-6):
        frame = "global"
    elif q_zero < 500.0 and q_zero < 0.45 * max(q_ego, 1e-6):
        frame = "ego"
    else:
        frame = "global" if q_ego <= q_zero else "ego"

    return frame, {
        "q10_to_ego_m": q_ego,
        "q10_to_origin_m": q_zero,
    }


def map_to_ego_frame(
    geoms: Sequence[MapGeometry],
    detected_frame: str,
    ego_pose: Tuple[float, float, float],
) -> List[MapGeometry]:
    ego_x, ego_y, ego_heading = ego_pose
    out: List[MapGeometry] = []

    for g in geoms:
        xy = g.xy
        if detected_frame == "global":
            xy = global_to_ego_xy(xy, ego_x, ego_y, ego_heading)
        else:
            xy = np.asarray(xy, dtype=np.float64)

        out.append(MapGeometry(g.category, g.name, xy))

    return out


# =============================================================================
# ROUTE EXTRACTION
# =============================================================================


def extract_route_path(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    mission = frame.get("mission_goal")
    if not isinstance(mission, dict):
        return None

    route = mission.get("route_path")
    if route is None:
        return None

    arr = _numeric_xy_array(route)
    if arr is not None and len(arr) >= 2:
        return arr

    # Tolerate [{x:..., y:...}, ...] or point-like objects.
    if isinstance(route, list):
        pts: List[Tuple[float, float]] = []
        for item in route:
            xy = _extract_xy_from_obj(item)
            if xy is None:
                return None
            pts.append(xy)
        if len(pts) >= 2:
            return np.asarray(pts, dtype=np.float64)

    return None


def route_to_ego_frame(
    route_xy: Optional[np.ndarray],
    map_frame: str,
    ego_pose: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    if route_xy is None:
        return None
    if map_frame == "global":
        return global_to_ego_xy(route_xy, *ego_pose)
    return route_xy


# =============================================================================
# KV CACHE
# =============================================================================


def load_cache_manifest(cache_root: Path) -> List[Dict[str, Any]]:
    manifest_path = cache_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing test cache manifest: {manifest_path}\n"
            "Run the existing PART2 AR-only E2E test once first. "
            "This visualization script intentionally does not rebuild VLM cache."
        )

    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    for item in manifest:
        p = Path(str(item.get("cache_file", "")))
        if not p.is_file():
            raise FileNotFoundError(p)

    return manifest


def manifest_by_id(
    manifest: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in manifest:
        sid = str(x.get("id", "")).strip()
        if not sid:
            continue
        if sid in out:
            raise RuntimeError(f"Duplicate id in cache manifest: {sid}")
        out[sid] = dict(x)
    return out


def load_cache_record(cache_item: Dict[str, Any]) -> Dict[str, Any]:
    cache = torch.load(
        cache_item["cache_file"],
        map_location="cpu",
        weights_only=False,
    )

    required = ("direct_kv", "reasoning_delta_kv", "trajectory")
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(
            f"Missing cache fields id={cache_item.get('id')}: {missing}"
        )

    return cache


def action_input_from_record(
    cache: Dict[str, Any],
    cache_item: Dict[str, Any],
    branch: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    direct = cache["direct_kv"]
    delta = cache["reasoning_delta_kv"]
    gt = cache["trajectory"].float().cpu().numpy()

    if direct.ndim != 2 or delta.ndim != 2:
        raise RuntimeError(f"Bad KV rank: id={cache_item['id']}")
    if direct.shape[-1] != delta.shape[-1]:
        raise RuntimeError(f"KV dim mismatch: id={cache_item['id']}")

    if branch == "direct":
        memory = direct
        segment_ids = torch.zeros(direct.shape[0], dtype=torch.long)
    elif branch == "reasoning":
        memory = torch.cat([direct, delta], dim=0)
        segment_ids = torch.cat([
            torch.zeros(direct.shape[0], dtype=torch.long),
            torch.ones(delta.shape[0], dtype=torch.long),
        ])
    else:
        raise ValueError(branch)

    memory = memory.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    memory_mask = torch.ones(
        (1, memory.shape[1]),
        dtype=torch.bool,
        device=device,
    )
    segment_ids = segment_ids.unsqueeze(0).to(
        device=device,
        non_blocking=True,
    )

    return {
        "memory": memory,
        "memory_mask": memory_mask,
        "segment_ids": segment_ids,
    }, gt


def extract_sample_context(
    row: Dict[str, Any],
    frame: Dict[str, Any],
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    mission_goal = frame.get("mission_goal")
    frame_command = (
        mission_goal.get("command")
        if isinstance(mission_goal, dict)
        else None
    )

    command = first_nonempty(
        row.get("mission_command"),
        row.get("command"),
        cache.get("command"),
        frame_command,
    )

    generated_reasoning = first_nonempty(
        cache.get("reasoning_text"),
    )

    gt_reasoning = first_nonempty(
        row.get("gt_reasoning"),
        row.get("reasoning"),
        cache.get("gt_reasoning"),
    )

    if generated_reasoning and gt_reasoning:
        text_metrics: Dict[str, Optional[float]] = {
            "token_f1": token_f1(generated_reasoning, gt_reasoning),
            "rouge_l_f1": rouge_l_f1(generated_reasoning, gt_reasoning),
        }
    else:
        text_metrics = {
            "token_f1": None,
            "rouge_l_f1": None,
        }

    return {
        "command": command,
        "generated_reasoning": generated_reasoning,
        "gt_reasoning": gt_reasoning,
        "text_metrics": text_metrics,
        "reasoning_generated_tokens": cache.get("reasoning_generated_tokens"),
        "reasoning_cached_tokens": cache.get("reasoning_cached_tokens"),
    }


# =============================================================================
# FLOW/DiT LOAD / INFERENCE
# =============================================================================


def flow_checkpoint_path(model_root: Path, branch: str) -> Path:
    return model_root / "flow" / f"{branch}_flow_dit" / "best.pt"


def load_flow(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

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

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    normalizer = TrajectoryNormalizer.from_dict(
        ckpt["normalizer"]
    ).to(device)

    solver_steps = int(cfg.get("solver_steps", 10))
    return model, ckpt, normalizer, solver_steps


@torch.inference_mode()
def run_flow_one(
    model,
    normalizer,
    solver_steps: int,
    inputs: Dict[str, torch.Tensor],
    sample_seed: int,
) -> np.ndarray:
    # euler_sample in the user's current code uses a CPU generator.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(sample_seed))

    pred = euler_sample(
        model=model,
        memory=inputs["memory"],
        memory_mask=inputs["memory_mask"],
        segment_ids=inputs["segment_ids"],
        normalizer=normalizer,
        solver_steps=int(solver_steps),
        rng=rng,
    )

    pred_np = pred.float().cpu().numpy()
    if pred_np.shape != (1, NUM_STEPS, 3):
        raise RuntimeError(
            f"Unexpected Flow output shape: {pred_np.shape}; "
            f"expected=(1,{NUM_STEPS},3)"
        )
    return pred_np[0]


# =============================================================================
# METRICS
# =============================================================================


def sample_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    pos_err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    heading_err = np.abs(wrap_angle(pred[:, 2] - gt[:, 2]))

    return {
        "ade_m": float(np.mean(pos_err)),
        "fde_m": float(pos_err[-1]),
        "heading_mae_rad": float(np.mean(heading_err)),
    }


# =============================================================================
# PLOTTING
# =============================================================================


def geometry_style(category: str) -> Dict[str, Any]:
    # Neutral map palette; trajectories remain visually dominant.
    styles: Dict[str, Dict[str, Any]] = {
        "baseline": {"color": "0.45", "lw": 0.9, "ls": "--", "alpha": 0.75},
        "boundary": {"color": "0.25", "lw": 0.9, "ls": "-", "alpha": 0.75},
        "lane": {"color": "0.65", "lw": 0.7, "ls": "-", "alpha": 0.55},
        "connector": {"color": "0.55", "lw": 0.8, "ls": ":", "alpha": 0.65},
        "crosswalk": {"color": "0.50", "lw": 0.8, "ls": "-", "alpha": 0.50},
        "intersection": {"color": "0.75", "lw": 0.8, "ls": "-", "alpha": 0.45},
        "stop": {"color": "0.35", "lw": 0.9, "ls": "-", "alpha": 0.65},
        "roadblock": {"color": "0.60", "lw": 0.8, "ls": "-", "alpha": 0.45},
        "traffic_light": {"color": "0.20", "lw": 1.0, "ls": "-", "alpha": 0.75},
        "map": {"color": "0.75", "lw": 0.6, "ls": "-", "alpha": 0.35},
    }
    return dict(styles.get(category, styles["map"]))


def plot_heading_arrows(
    ax,
    trajectory: np.ndarray,
    color: str,
    stride: int = 2,
) -> None:
    traj = np.asarray(trajectory)
    for i in range(0, len(traj), max(1, stride)):
        x, y, yaw = traj[i]
        length = 1.5
        ax.arrow(
            x,
            y,
            length * math.cos(float(yaw)),
            length * math.sin(float(yaw)),
            head_width=0.45,
            head_length=0.65,
            fc=color,
            ec=color,
            alpha=0.75,
            length_includes_head=True,
            zorder=7,
        )


def plot_one_sample(
    output_path: Path,
    row: Dict[str, Any],
    frame: Dict[str, Any],
    map_geoms_ego: Sequence[MapGeometry],
    route_ego: Optional[np.ndarray],
    gt: np.ndarray,
    predictions: Dict[str, np.ndarray],
    trajectory_metrics: Dict[str, Dict[str, float]],
    command: str,
    generated_reasoning: str,
    gt_reasoning: str,
    reasoning_metrics: Dict[str, Optional[float]],
    map_radius: float,
    max_map_elements: int,
    text_max_chars: int,
    show: bool,
) -> None:
    """
    Save a clean qualitative figure.

    The image intentionally omits internal/debug metadata such as clip path,
    map frame detection statistics, map filename, and vector counts. Those are
    still saved in the adjacent JSON file and printed to the terminal.
    """
    fig = plt.figure(figsize=(12.5, 14.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[2.25, 8.0],
        hspace=0.08,
    )

    info_ax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    info_ax.axis("off")

    # ---------------------------------------------------------------------
    # Top information panel: only interpretation-relevant information.
    # ---------------------------------------------------------------------
    pred_reasoning_display = compact_text(generated_reasoning, text_max_chars)
    gt_reasoning_display = compact_text(gt_reasoning, text_max_chars)

    f1 = reasoning_metrics.get("token_f1")
    rouge = reasoning_metrics.get("rouge_l_f1")

    if f1 is None:
        f1_text = "N/A"
    else:
        f1_text = f"{float(f1) * 100.0:.2f}%"

    if rouge is None:
        rouge_text = "N/A"
    else:
        rouge_text = f"{float(rouge) * 100.0:.2f}%"

    info_lines = [
        wrapped_label("Command", command, width=120),
        wrapped_label("VLM Reasoning", pred_reasoning_display, width=120),
        wrapped_label("GT Reasoning", gt_reasoning_display, width=120),
        f"Reasoning metrics: Token F1 = {f1_text}   |   ROUGE-L F1 = {rouge_text}",
    ]

    # Compact trajectory metric summary in the same panel.
    traj_parts: List[str] = []
    for branch in predictions:
        m = trajectory_metrics[branch]
        traj_parts.append(
            f"{branch} DiT: ADE {m['ade_m']:.3f} m / "
            f"FDE {m['fde_m']:.3f} m / "
            f"Heading {m['heading_mae_rad']:.3f} rad"
        )
    if traj_parts:
        info_lines.append("Trajectory: " + "   |   ".join(traj_parts))

    info_ax.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.98,
        },
    )

    # ---------------------------------------------------------------------
    # HD map.
    # ---------------------------------------------------------------------
    drawn = 0
    for geom in map_geoms_ego:
        if drawn >= max_map_elements:
            break

        xy = geom.xy
        if len(xy) < 2:
            continue

        if (
            np.nanmax(xy[:, 0]) < -map_radius * 1.4
            or np.nanmin(xy[:, 0]) > map_radius * 1.4
            or np.nanmax(xy[:, 1]) < -map_radius * 1.4
            or np.nanmin(xy[:, 1]) > map_radius * 1.4
        ):
            continue

        style = geometry_style(geom.category)
        ax.plot(xy[:, 0], xy[:, 1], zorder=1, **style)
        drawn += 1

    # Route path.
    if route_ego is not None and len(route_ego) >= 2:
        ax.plot(
            route_ego[:, 0],
            route_ego[:, 1],
            color="tab:purple",
            lw=2.2,
            ls=":",
            alpha=0.85,
            label="Route path",
            zorder=4,
        )

    # Current ego pose.
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=180,
        color="black",
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
        fc="black",
        ec="black",
        zorder=10,
        length_includes_head=True,
    )

    # Ground truth trajectory.
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        color="tab:green",
        lw=3.0,
        marker="o",
        ms=5,
        label="Ground Truth",
        zorder=8,
    )
    plot_heading_arrows(ax, gt, "tab:green")

    branch_colors = {
        "direct": "tab:blue",
        "reasoning": "tab:red",
    }

    for branch, pred in predictions.items():
        m = trajectory_metrics[branch]
        label = (
            f"{branch}_flow_dit "
            f"(ADE {m['ade_m']:.2f} m, FDE {m['fde_m']:.2f} m)"
        )
        color = branch_colors.get(branch, "tab:orange")

        ax.plot(
            pred[:, 0],
            pred[:, 1],
            color=color,
            lw=2.6,
            marker="x",
            ms=6,
            label=label,
            zorder=9,
        )
        plot_heading_arrows(ax, pred, color)

    # Plot bounds based on trajectories, keeping a minimum HD-map radius.
    all_traj = [gt[:, :2]] + [x[:, :2] for x in predictions.values()]
    traj_pts = np.concatenate(all_traj, axis=0)
    traj_extent = float(np.max(np.linalg.norm(traj_pts, axis=1)))
    radius = max(float(map_radius), traj_extent + 10.0)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Ego-frame x [m] (forward)")
    ax.set_ylabel("Ego-frame y [m] (left)")
    ax.set_title(
        "Flow/DiT Trajectory on nuReasoning HD Map",
        fontsize=14,
        pad=10,
    )
    ax.legend(loc="best", fontsize=9)

    # Small non-intrusive sample index/footer for traceability.
    frame_index = frame.get("frame_index", row.get("frame_index", "?"))
    ax.text(
        0.995,
        0.005,
        f"frame {frame_index}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        alpha=0.55,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--raw-part2-root", type=Path, default=RAW_PART2_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--branches",
        nargs="+",
        choices=BRANCHES,
        default=list(BRANCHES),
        help="Flow/DiT branches to draw. Transformer models are never loaded.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows from the start. Ignored when --sample-index is used.",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Zero-based dataset row index to visualize exactly one sample.",
    )
    p.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Visualize exactly one sample by id. Overrides --limit.",
    )

    p.add_argument(
        "--solver-steps",
        type=int,
        default=None,
        help="Override checkpoint solver steps. Normally leave as 10.",
    )
    p.add_argument("--flow-seed", type=int, default=FLOW_TEST_SEED)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument(
        "--map-frame",
        choices=("auto", "global", "ego"),
        default="auto",
        help="Coordinate frame of map.pkl geometry.",
    )
    p.add_argument(
        "--map-radius",
        type=float,
        default=50.0,
        help="Minimum half-width/height of BEV plot in meters.",
    )
    p.add_argument(
        "--max-map-elements",
        type=int,
        default=10000,
        help="Safety cap on vector elements drawn per sample.",
    )
    p.add_argument(
        "--text-max-chars",
        type=int,
        default=520,
        help=(
            "Maximum characters of generated/GT reasoning shown in each PNG. "
            "Full text is always preserved in the adjacent JSON."
        ),
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also open matplotlib window after saving PNG.",
    )
    p.add_argument(
        "--dump-map-schema",
        action="store_true",
        help="Print top-level map.pkl fields for each selected clip.",
    )

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================


def select_rows(
    rows: Sequence[Dict[str, Any]],
    args,
) -> List[Tuple[int, Dict[str, Any]]]:
    if args.sample_id is not None:
        sid = str(args.sample_id)
        matches = [(i, row) for i, row in enumerate(rows) if row["id"] == sid]
        if not matches:
            raise KeyError(f"sample id not found: {sid}")
        return matches[:1]

    if args.sample_index is not None:
        idx = int(args.sample_index)
        if idx < 0 or idx >= len(rows):
            raise IndexError(
                f"--sample-index out of range: {idx}, dataset={len(rows)}"
            )
        return [(idx, rows[idx])]

    if args.limit == 0:
        raise ValueError("--limit must be > 0 or < 0 for all rows")

    chosen = list(enumerate(rows))
    if args.limit > 0:
        chosen = chosen[: args.limit]
    return chosen


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Flow/DiT inference")
    if args.map_radius <= 0:
        raise ValueError("--map-radius must be > 0")
    if args.max_map_elements < 1:
        raise ValueError("--max-map-elements must be >= 1")
    if args.text_max_chars < 80:
        raise ValueError("--text-max-chars must be >= 80")
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
    set_seed(args.seed)

    rows = load_rows(args.dataset)
    selected = select_rows(rows, args)

    manifest = load_cache_manifest(args.cache_root)
    cache_by_id = manifest_by_id(manifest)

    missing_cache = [row["id"] for _, row in selected if row["id"] not in cache_by_id]
    if missing_cache:
        raise RuntimeError(
            "Selected dataset rows are missing from current test KV cache. "
            f"First missing IDs: {missing_cache[:5]}"
        )

    print("=" * 120)
    print("Flow/DiT ONLY | nuReasoning PART2 HD-MAP TRAJECTORY VISUAL TEST")
    print("Script version :", SCRIPT_VERSION)
    print("=" * 120)
    print("GPU          :", torch.cuda.get_device_name(args.gpu_id))
    print("Dataset      :", args.dataset)
    print("Raw PART2    :", args.raw_part2_root)
    print("KV cache     :", args.cache_root)
    print("Model root   :", args.model_root)
    print("Branches     :", ", ".join(args.branches))
    print("Selected     :", len(selected))
    print("Map frame    :", args.map_frame)
    print("Transformer  : NOT USED")

    # Load only requested Flow/DiT models once.
    flow_models: Dict[str, Dict[str, Any]] = {}
    for branch in args.branches:
        ckpt_path = flow_checkpoint_path(args.model_root, branch)
        model, ckpt, normalizer, checkpoint_solver_steps = load_flow(
            ckpt_path,
            device,
        )

        solver_steps = (
            int(args.solver_steps)
            if args.solver_steps is not None
            else int(checkpoint_solver_steps)
        )

        flow_models[branch] = {
            "model": model,
            "ckpt": ckpt,
            "normalizer": normalizer,
            "solver_steps": solver_steps,
            "checkpoint": ckpt_path,
        }

        print(
            f"[LOAD] {branch}_flow_dit | "
            f"epoch={int(ckpt.get('epoch', -1))} | "
            f"solver_steps={solver_steps} | "
            f"D={int(ckpt['input_dim'])}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for out_idx, (dataset_index, row) in enumerate(selected, 1):
        sid = row["id"]
        clip = row["clip"]
        cache_item = cache_by_id[sid]

        print("\n" + "-" * 120)
        print(
            f"[{out_idx:03d}/{len(selected):03d}] "
            f"dataset_index={dataset_index} | id={sid} | clip={clip}"
        )
        print("-" * 120)

        clip_dir, metadata = load_clip_metadata(args.raw_part2_root, clip)
        frame = match_raw_frame(row, metadata)

        ego_path = resolve_ego_path(clip_dir, frame)
        map_path = resolve_map_path(clip_dir, metadata)

        ego_state = load_nureasoning_pickle(ego_path)
        ego_pose = extract_ego_pose(ego_state)

        map_obj = load_nureasoning_pickle(map_path)
        map_schema = summarize_map_schema(map_obj)
        if args.dump_map_schema:
            print("[MAP SCHEMA]", json.dumps(map_schema, ensure_ascii=False))

        map_geoms = extract_map_geometries(map_obj)
        if not map_geoms:
            raise RuntimeError(
                f"No vector geometry extracted from {map_path}. "
                "Run with --dump-map-schema and inspect the printed top-level fields."
            )

        auto_frame, frame_stats = estimate_map_frame(
            map_geoms,
            ego_pose[0],
            ego_pose[1],
        )
        map_frame = auto_frame if args.map_frame == "auto" else args.map_frame

        map_geoms_ego = map_to_ego_frame(
            map_geoms,
            map_frame,
            ego_pose,
        )

        route = extract_route_path(frame)
        route_ego = route_to_ego_frame(route, map_frame, ego_pose)

        print(
            f"[RAW] ego=({ego_pose[0]:.3f}, {ego_pose[1]:.3f}, "
            f"yaw={ego_pose[2]:.4f}) | map={map_path}"
        )
        print(
            f"[MAP] vectors={len(map_geoms)} | detected={auto_frame} | "
            f"used={map_frame} | "
            f"q10->ego={frame_stats['q10_to_ego_m']:.2f}m | "
            f"q10->origin={frame_stats['q10_to_origin_m']:.2f}m"
        )

        cache_record = load_cache_record(cache_item)
        sample_context = extract_sample_context(
            row=row,
            frame=frame,
            cache=cache_record,
        )

        command = sample_context["command"]
        generated_reasoning = sample_context["generated_reasoning"]
        gt_reasoning = sample_context["gt_reasoning"]
        reasoning_metrics = sample_context["text_metrics"]

        f1_value = reasoning_metrics.get("token_f1")
        rouge_value = reasoning_metrics.get("rouge_l_f1")

        print(f"[COMMAND] {command or 'N/A'}")
        print(f"[REASONING/PRED] {generated_reasoning or 'N/A'}")
        print(f"[REASONING/GT]   {gt_reasoning or 'N/A'}")
        if f1_value is not None and rouge_value is not None:
            print(
                f"[TEXT] Token F1={float(f1_value) * 100.0:.2f}% | "
                f"ROUGE-L F1={float(rouge_value) * 100.0:.2f}%"
            )
        else:
            print(
                "[TEXT] Token F1=N/A | ROUGE-L F1=N/A "
                "(GT reasoning or generated reasoning missing)"
            )

        gt_from_cache: Optional[np.ndarray] = None
        predictions: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        # SAME noise seed for direct/reasoning on this exact sample.
        sample_seed = int(args.flow_seed) + int(dataset_index)

        for branch in args.branches:
            inputs, gt = action_input_from_record(
                cache=cache_record,
                cache_item=cache_item,
                branch=branch,
                device=device,
            )

            if gt_from_cache is None:
                gt_from_cache = gt
            elif not np.allclose(gt_from_cache, gt, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    f"GT differs between cache loads: id={sid}"
                )

            bundle = flow_models[branch]
            pred = run_flow_one(
                model=bundle["model"],
                normalizer=bundle["normalizer"],
                solver_steps=bundle["solver_steps"],
                inputs=inputs,
                sample_seed=sample_seed,
            )

            predictions[branch] = pred
            metrics[branch] = sample_metrics(pred, gt)

            m = metrics[branch]
            print(
                f"[DIT] {branch:<9s} | seed={sample_seed} | "
                f"ADE={m['ade_m']:.4f}m | "
                f"FDE={m['fde_m']:.4f}m | "
                f"HEAD={m['heading_mae_rad']:.4f}rad"
            )

            del inputs

        assert gt_from_cache is not None

        # Prepared row GT and cache GT should be identical.
        row_gt = np.asarray(row["trajectory"], dtype=np.float32)
        if not np.allclose(gt_from_cache, row_gt, atol=1e-5, rtol=0.0):
            max_diff = float(np.max(np.abs(gt_from_cache - row_gt)))
            print(
                f"[WARN] dataset row GT != cache GT | max_abs_diff={max_diff:.6e}. "
                "Plotting cache GT because that is what the DiT evaluation uses."
            )

        png_name = (
            f"{dataset_index:04d}_{safe_name(sid)}_"
            + "_".join(args.branches)
            + ".png"
        )
        json_name = Path(png_name).with_suffix(".json").name

        png_path = run_dir / png_name
        json_path = run_dir / json_name

        plot_one_sample(
            output_path=png_path,
            row=row,
            frame=frame,
            map_geoms_ego=map_geoms_ego,
            route_ego=route_ego,
            gt=gt_from_cache,
            predictions=predictions,
            trajectory_metrics=metrics,
            command=command,
            generated_reasoning=generated_reasoning,
            gt_reasoning=gt_reasoning,
            reasoning_metrics=reasoning_metrics,
            map_radius=args.map_radius,
            max_map_elements=args.max_map_elements,
            text_max_chars=args.text_max_chars,
            show=args.show,
        )

        payload = {
            "dataset_index": int(dataset_index),
            "id": sid,
            "clip": clip,
            "frame_index": frame.get("frame_index"),
            "timestamp_us": frame.get("timestamp_us"),
            "ego_pose_global": {
                "x": float(ego_pose[0]),
                "y": float(ego_pose[1]),
                "heading": float(ego_pose[2]),
            },
            "map_path": str(map_path),
            "map_schema": map_schema,
            "map_vector_count": len(map_geoms),
            "map_frame_auto": auto_frame,
            "map_frame_used": map_frame,
            "map_frame_stats": frame_stats,
            "route_path_ego": (
                route_ego.tolist() if route_ego is not None else None
            ),
            "flow_seed": int(sample_seed),
            "mission_command": command,
            "generated_reasoning": generated_reasoning,
            "gt_reasoning": gt_reasoning,
            "reasoning_metrics": reasoning_metrics,
            "reasoning_generated_tokens": sample_context.get(
                "reasoning_generated_tokens"
            ),
            "reasoning_cached_tokens": sample_context.get(
                "reasoning_cached_tokens"
            ),
            "ground_truth": gt_from_cache.tolist(),
            "predictions": {
                branch: predictions[branch].tolist()
                for branch in predictions
            },
            "metrics": metrics,
            "checkpoints": {
                branch: str(flow_models[branch]["checkpoint"])
                for branch in args.branches
            },
            "solver_steps": {
                branch: int(flow_models[branch]["solver_steps"])
                for branch in args.branches
            },
            "png": str(png_path),
        }

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_row = {
            "dataset_index": int(dataset_index),
            "id": sid,
            "clip": clip,
            "mission_command": command,
            "generated_reasoning": generated_reasoning,
            "gt_reasoning": gt_reasoning,
            "reasoning_token_f1": reasoning_metrics.get("token_f1"),
            "reasoning_rouge_l_f1": reasoning_metrics.get("rouge_l_f1"),
            "map_frame": map_frame,
            "map_vector_count": len(map_geoms),
            "png": str(png_path),
        }
        for branch, m in metrics.items():
            summary_row[f"{branch}_ade_m"] = m["ade_m"]
            summary_row[f"{branch}_fde_m"] = m["fde_m"]
            summary_row[f"{branch}_heading_mae_rad"] = m[
                "heading_mae_rad"
            ]
        summary_rows.append(summary_row)

        print("[SAVE]", png_path)
        print("[SAVE]", json_path)

        del ego_state, map_obj, map_geoms, map_geoms_ego, cache_record
        cleanup_cuda()

    summary_path = run_dir / "summary.jsonl"
    write_jsonl(summary_path, summary_rows)

    # Aggregate metrics for quick sanity check.
    print("\n" + "=" * 120)
    print("DONE | Flow/DiT HD-MAP VISUALIZATION")
    print("=" * 120)
    print("Result dir :", run_dir)
    print("Summary    :", summary_path)

    text_f1s = [
        float(x["reasoning_token_f1"])
        for x in summary_rows
        if x.get("reasoning_token_f1") is not None
    ]
    text_rouges = [
        float(x["reasoning_rouge_l_f1"])
        for x in summary_rows
        if x.get("reasoning_rouge_l_f1") is not None
    ]
    if text_f1s:
        print(
            f"Reasoning | Token F1={np.mean(text_f1s) * 100.0:.2f}% | "
            f"ROUGE-L F1={np.mean(text_rouges) * 100.0:.2f}% | "
            f"N={len(text_f1s)}"
        )

    for branch in args.branches:
        ade_key = f"{branch}_ade_m"
        fde_key = f"{branch}_fde_m"
        head_key = f"{branch}_heading_mae_rad"

        ades = [float(x[ade_key]) for x in summary_rows if ade_key in x]
        fdes = [float(x[fde_key]) for x in summary_rows if fde_key in x]
        heads = [float(x[head_key]) for x in summary_rows if head_key in x]

        if ades:
            print(
                f"{branch:<9s} | "
                f"ADE={np.mean(ades):.4f}m | "
                f"FDE={np.mean(fdes):.4f}m | "
                f"HEAD={np.mean(heads):.4f}rad | "
                f"N={len(ades)}"
            )

    # Explicitly release large models at the end.
    flow_models.clear()
    cleanup_cuda()


if __name__ == "__main__":
    main()
