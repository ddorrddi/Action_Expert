#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM_Baseline -> Reasoning_VLM_v2

Continued LoRA SFT on nuReasoning Part1.

========================================================================
PURPOSE
========================================================================

DO NOT remove any original VLM_Baseline capability.

Existing VLM_Baseline tasks:
    1. decision
    2. counterfactual
    3. spatial
    4. spatial_obj
    5. trajectory

New task:
    6. reasoning

Input:
    - front-left image
    - front image
    - front-right image
    - mission command
    - current speed
    - current acceleration
    - current heading/yaw

Trajectory target:
    - 5 second horizon
    - 0.5 s interval
    - 10 waypoints
    - current ego frame
    - x forward
    - y left
    - text output:
        (x1,y1), (x2,y2), ... (x10,y10)

Training:
    - raw nuReasoning Part1
    - clip-disjoint 70 / 20 / 10 split
    - NO manual task sampling ratio
    - natural shuffled loading
    - inverse-sqrt task weighting
    - assistant-response-token-only CE
    - LoRA r=16 alpha=32
    - BF16
    - SDPA
    - gradient checkpointing

Base:
    ~/lab/models/vlm/VLM_Baseline

Raw dataset:
    /media/HDD/nuReasoning/train/part_1

Prepared:
    ~/lab/VLM/dataset/reasoning_v2_multitask

LoRA:
    ~/lab/VLM/checkpoints/reasoning_v2_lora

Final merged:
    ~/lab/models/vlm/Reasoning_VLM_v2
"""

from __future__ import annotations


# =============================================================================
# ENVIRONMENT
# =============================================================================

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)


# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import gc
import json
import math
import pickle
import random
import re
import shutil
import sys
import time
import types

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)


# =============================================================================
# PATHS
# =============================================================================

HOME = Path.home()

PART1_ROOT = Path(
    "/media/HDD/nuReasoning/train/part_1"
)

BASE_MODEL_PATH = (
    HOME
    / "lab"
    / "models"
    / "vlm"
    / "VLM_Baseline"
)

PROCESSOR_FALLBACK_PATH = (
    HOME
    / "lab"
    / "models"
    / "vlm"
    / "Qwen3-VL-2B-Instruct"
)

DATA_DIR = (
    HOME
    / "lab"
    / "VLM"
    / "dataset"
    / "reasoning_v2_multitask"
)

RUN_DIR = (
    HOME
    / "lab"
    / "VLM"
    / "checkpoints"
    / "reasoning_v2_lora"
)

FINAL_MODEL_DIR = (
    HOME
    / "lab"
    / "models"
    / "vlm"
    / "Reasoning_VLM_v2"
)


# =============================================================================
# ORIGINAL VLM_BASELINE TASK DEFINITIONS
# =============================================================================

LON_CLASSES = [
    "Maintain speed",
    "Gently accelerate",
    "Slow down gently",
    "Slow down quickly",
    "Gently come to a stop",
    "Quickly come to a stop",
    "Remain stopped",
]

LAT_CLASSES = [
    "No lateral action",
    "Slightly move left in the lane",
    "Slightly move right in the lane",
    "Left lane change",
    "Right lane change",
    "Turn left",
    "Turn right",
]

RISK_CLASSES = [
    "Safe",
    "Suboptimal",
    "Unsafe",
]

TASKS = (
    "reasoning",       # NEW
    "decision",
    "counterfactual",
    "spatial",
    "spatial_obj",
    "trajectory",
)

LEGACY_TASKS = (
    "decision",
    "counterfactual",
    "spatial",
    "spatial_obj",
    "trajectory",
)


# =============================================================================
# CAMERAS
# =============================================================================

CAMERA_ORDER = (
    "front_left",
    "front",
    "front_right",
)

CAMERA_ALIASES = {

    "front_left": (
        "front_left",
        "front-left",
        "frontleft",
        "CAM_M_L0",
        "cam_m_l0",
    ),

    "front": (
        "front",
        "front_center",
        "front-center",
        "CAM_M_F",
        "cam_m_f",
    ),

    "front_right": (
        "front_right",
        "front-right",
        "frontright",
        "CAM_M_R0",
        "cam_m_r0",
    ),
}

CAMERA_DIR_FALLBACK = {
    "front_left": "CAM_M_L0",
    "front": "CAM_M_F",
    "front_right": "CAM_M_R0",
}


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

DEFAULT_SEED = 20260829

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

DTYPE = torch.bfloat16

ATTN_IMPLEMENTATION = "sdpa"

MIN_PIXELS = 200_704
MAX_PIXELS = 200_704

MAX_SEQ_LEN = 2048

DEFAULT_EPOCHS = 5

DEFAULT_GRAD_ACCUM = 8

DEFAULT_LR = 2.0e-5

DEFAULT_WEIGHT_DECAY = 0.01

DEFAULT_WARMUP_RATIO = 0.05

DEFAULT_MAX_GRAD_NORM = 1.0

DEFAULT_LORA_R = 16

DEFAULT_LORA_ALPHA = 32

DEFAULT_LORA_DROPOUT = 0.05

DEFAULT_PATIENCE = 2

DEFAULT_MIN_DELTA = 1.0e-4

DEFAULT_LOG_EVERY = 20

# =============================================================================
# COMPACT DATASET DEFAULTS
# =============================================================================
# The original script can generate many rows from one frame because
# counterfactual / spatial / spatial_obj expand one annotation into many rows.
# This compact preset keeps all six tasks but bounds the total training set.
DEFAULT_MAX_FRAMES_PER_CLIP = 8

DEFAULT_PREP_TASK_CAPS = {
    "train": {
        "reasoning": 3000,
        "decision": 1500,
        "trajectory": 2000,
        "counterfactual": 500,
        "spatial": 500,
        "spatial_obj": 500,
    },
    "val": {
        "reasoning": 400,
        "decision": 250,
        "trajectory": 300,
        "counterfactual": 100,
        "spatial": 100,
        "spatial_obj": 100,
    },
    "test": {
        "reasoning": 300,
        "decision": 200,
        "trajectory": 250,
        "counterfactual": 75,
        "spatial": 75,
        "spatial_obj": 75,
    },
}


# =============================================================================
# LEGACY PERFORMANCE PROTECTION
# =============================================================================

# 기존 task validation CE가 시작 VLM_Baseline 대비
# 최대 5%까지 악화되는 것은 허용.
LEGACY_RELATIVE_TOLERANCE = 0.05

# 그 이상 악화되면 best checkpoint score에 penalty.
LEGACY_REGRESSION_PENALTY = 0.50


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def get_field(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(
        obj,
        key,
        default,
    )


def get_ci(
    mapping: Any,
    candidates: Sequence[str],
    default: Any = None,
) -> Any:

    if not isinstance(mapping, dict):
        return default

    lowered = {
        str(k).strip().lower(): v
        for k, v in mapping.items()
    }

    for candidate in candidates:

        key = str(
            candidate
        ).strip().lower()

        if key in lowered:
            return lowered[key]

    return default


def number(
    value: Any,
) -> Optional[float]:

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return None

    try:
        result = float(value)

    except Exception:
        return None

    if not math.isfinite(result):
        return None

    return result


def read_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def read_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(line)
                )

            except Exception as exc:

                raise RuntimeError(
                    f"JSONL parse error\n"
                    f"path={path}\n"
                    f"line={line_no}"
                ) from exc

    return rows


def write_jsonl(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cleanup_cuda() -> None:

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_path(
    clip_dir: Path,
    value: Any,
) -> Optional[Path]:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    if text.startswith(
        "file://"
    ):
        text = text[7:]

    path = Path(
        text
    ).expanduser()

    candidates = []

    if path.is_absolute():

        candidates.append(
            path
        )

    else:

        candidates.extend(
            [
                clip_dir / path,

                PART1_ROOT / path,

                PART1_ROOT.parent
                / path,

                Path(
                    "/media/HDD/nuReasoning"
                )
                / path,
            ]
        )

    for candidate in candidates:

        if candidate.exists():
            return candidate.resolve()

    return None


# =============================================================================
# PICKLE COMPATIBILITY
# =============================================================================

def install_pickle_aliases() -> None:

    candidates = [
        Path(
            "/media/HDD/nuReasoning"
        ),
        PART1_ROOT.parent,
        PART1_ROOT.parent.parent,
    ]

    for root in candidates:

        if root.exists():

            root_text = str(
                root
            )

            if root_text not in sys.path:
                sys.path.insert(
                    0,
                    root_text,
                )

    try:

        import data_schema  # type: ignore

        sys.modules.setdefault(
            "data_schema_v0",
            data_schema,
        )

        return

    except Exception:
        pass

    # Minimal fallback for old pickle references.
    module = types.ModuleType(
        "data_schema"
    )

    class EgoState:
        pass

    EgoState.__module__ = (
        "data_schema"
    )

    module.EgoState = EgoState

    sys.modules[
        "data_schema"
    ] = module

    sys.modules[
        "data_schema_v0"
    ] = module


def load_pickle(
    path: Path,
) -> Any:

    with path.open(
        "rb"
    ) as f:

        return pickle.load(f)


# =============================================================================
# CAMERA HELPERS
# =============================================================================

def extract_timestamp_from_path(
    value: Any,
) -> Optional[int]:

    if value is None:
        return None

    matches = re.findall(
        r"\d{12,20}",
        str(value),
    )

    if not matches:
        return None

    try:
        return int(
            matches[-1]
        )

    except Exception:
        return None


def frame_timestamp(
    frame: Dict[str, Any],
) -> int:

    for key in (
        "timestamp_us",
        "timestamp",
        "timestamp_usec",
        "timestamp_microseconds",
    ):

        value = get_ci(
            frame,
            [key],
        )

        parsed = number(
            value
        )

        if parsed is not None:
            return int(parsed)

    timestamp = (
        extract_timestamp_from_path(
            get_ci(
                frame,
                ["reasoning"],
            )
        )
    )

    return int(
        timestamp or 0
    )


def camera_dict(
    frame: Dict[str, Any],
) -> Dict[str, Any]:

    sensors = (
        get_ci(
            frame,
            ["sensors"],
            {},
        )
        or {}
    )

    cameras = get_ci(
        sensors,
        [
            "cameras",
            "camera",
        ],
        None,
    )

    if isinstance(
        cameras,
        dict,
    ):
        return cameras

    cameras = get_ci(
        frame,
        [
            "cameras",
            "camera",
        ],
        {},
    )

    if isinstance(
        cameras,
        dict,
    ):
        return cameras

    return {}


def find_camera_value(
    cameras: Dict[str, Any],
    canonical: str,
) -> Any:

    mapping = {
        str(k).strip().lower(): v
        for k, v in cameras.items()
    }

    for alias in CAMERA_ALIASES[
        canonical
    ]:

        alias = alias.lower()

        if alias in mapping:
            return mapping[alias]

    return None


_CAMERA_FILE_CACHE: Dict[
    Tuple[str, str],
    List[Tuple[int, Path]],
] = {}


def fallback_camera_files(
    clip_dir: Path,
    canonical: str,
) -> List[Tuple[int, Path]]:

    key = (
        str(clip_dir),
        canonical,
    )

    if key in _CAMERA_FILE_CACHE:
        return _CAMERA_FILE_CACHE[key]

    sensor_dir = CAMERA_DIR_FALLBACK[
        canonical
    ]

    files = []

    candidates = [
        clip_dir
        / "cameras"
        / sensor_dir,

        clip_dir
        / sensor_dir,
    ]

    for directory in candidates:

        if not directory.is_dir():
            continue

        for path in directory.iterdir():

            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                continue

            timestamp = (
                extract_timestamp_from_path(
                    path.name
                )
            )

            if timestamp is not None:

                files.append(
                    (
                        timestamp,
                        path.resolve(),
                    )
                )

    files.sort(
        key=lambda x: x[0]
    )

    _CAMERA_FILE_CACHE[
        key
    ] = files

    return files


def nearest_camera_image(
    clip_dir: Path,
    canonical: str,
    timestamp: int,
) -> Optional[Path]:

    files = fallback_camera_files(
        clip_dir,
        canonical,
    )

    if (
        not files
        or timestamp <= 0
    ):
        return None

    return min(
        files,
        key=lambda x: abs(
            x[0] - timestamp
        ),
    )[1]


def resolve_three_front_images(
    frame: Dict[str, Any],
    clip_dir: Path,
) -> Optional[List[str]]:

    cameras = camera_dict(
        frame
    )

    timestamp = frame_timestamp(
        frame
    )

    result = []

    for canonical in CAMERA_ORDER:

        value = find_camera_value(
            cameras,
            canonical,
        )

        path = resolve_path(
            clip_dir,
            value,
        )

        if path is None:

            path = nearest_camera_image(
                clip_dir,
                canonical,
                timestamp,
            )

        if (
            path is None
            or not path.is_file()
        ):
            return None

        result.append(
            str(path)
        )

    return result


def resolve_reasoning_path(
    frame: Dict[str, Any],
    clip_dir: Path,
) -> Optional[Path]:

    path = resolve_path(
        clip_dir,
        get_ci(
            frame,
            ["reasoning"],
        ),
    )

    if (
        path is not None
        and path.is_file()
    ):
        return path

    timestamp = frame_timestamp(
        frame
    )

    candidate = (
        clip_dir
        / "reasoning"
        / f"{timestamp}.json"
    )

    if candidate.is_file():
        return candidate.resolve()

    return None


# =============================================================================
# VECTOR HELPERS
# =============================================================================

def vector_xy(
    value: Any,
    xkeys: Sequence[str],
    ykeys: Sequence[str],
) -> Optional[
    Tuple[float, float]
]:

    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):

        x = None
        y = None

        for key in xkeys:

            parsed = number(
                value.get(key)
            )

            if parsed is not None:

                x = parsed
                break

        for key in ykeys:

            parsed = number(
                value.get(key)
            )

            if parsed is not None:

                y = parsed
                break

        if (
            x is not None
            and y is not None
        ):
            return (
                float(x),
                float(y),
            )

    for xkey in xkeys:

        x = number(
            getattr(
                value,
                xkey,
                None,
            )
        )

        if x is None:
            continue

        for ykey in ykeys:

            y = number(
                getattr(
                    value,
                    ykey,
                    None,
                )
            )

            if y is not None:

                return (
                    float(x),
                    float(y),
                )

    try:

        arr = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(-1)

        if (
            arr.size >= 2
            and np.isfinite(
                arr[:2]
            ).all()
        ):

            return (
                float(arr[0]),
                float(arr[1]),
            )

    except Exception:
        pass

    return None


def nested_containers(
    ego_state: Any,
) -> Iterable[Any]:

    yield ego_state

    for name in (
        "state",
        "kinematics",
        "motion",
        "dynamic_car_state",
    ):

        obj = get_field(
            ego_state,
            name,
            None,
        )

        if obj is not None:
            yield obj


# =============================================================================
# SPEED
# =============================================================================

def extract_speed_mps(
    ego_state: Any,
) -> Optional[float]:

    # Direct scalar.
    for container in nested_containers(
        ego_state
    ):

        for name in (
            "speed_mps",
            "speed",
            "vehicle_speed",
            "ego_speed",
            "current_speed",
        ):

            value = number(
                get_field(
                    container,
                    name,
                    None,
                )
            )

            if value is not None:
                return abs(
                    float(value)
                )

    # Velocity vectors.
    for container in nested_containers(
        ego_state
    ):

        for name in (
            "velocity",
            "linear_velocity",
            "velocity_mps",
            "vel",
            "ego_velocity",
            "rear_axle_velocity_2d",
        ):

            vec = vector_xy(
                get_field(
                    container,
                    name,
                    None,
                ),
                (
                    "x",
                    "vx",
                ),
                (
                    "y",
                    "vy",
                ),
            )

            if vec is not None:

                return float(
                    math.hypot(
                        vec[0],
                        vec[1],
                    )
                )

    return None


# =============================================================================
# POSE / HEADING
# =============================================================================

def quaternion_yaw(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> float:

    siny_cosp = 2.0 * (
        qw * qz
        +
        qx * qy
    )

    cosy_cosp = 1.0 - 2.0 * (
        qy * qy
        +
        qz * qz
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp,
    )


def extract_pose(
    ego_state: Any,
) -> Tuple[
    float,
    float,
    float,
]:

    pose = (
        get_field(
            ego_state,
            "pose",
            {},
        )
        or {}
    )

    # -------------------------------------------------------------------------
    # X / Y
    # -------------------------------------------------------------------------

    x = number(
        get_field(
            pose,
            "x",
            None,
        )
    )

    y = number(
        get_field(
            pose,
            "y",
            None,
        )
    )

    if (
        x is None
        or y is None
    ):

        try:

            arr = np.asarray(
                pose,
                dtype=np.float64,
            ).reshape(-1)

            if arr.size >= 2:

                x = float(
                    arr[0]
                )

                y = float(
                    arr[1]
                )

        except Exception:
            pass

    if (
        x is None
        or y is None
    ):

        raise ValueError(
            "Could not extract ego pose x/y"
        )

    # -------------------------------------------------------------------------
    # Heading / yaw
    # -------------------------------------------------------------------------

    yaw = None

    for key in (
        "yaw",
        "heading",
        "heading_rad",
        "theta",
        "psi",
    ):

        yaw = number(
            get_field(
                pose,
                key,
                None,
            )
        )

        if yaw is not None:
            break

    # Quaternion directly in pose.
    if yaw is None:

        qw = number(
            get_field(
                pose,
                "qw",
                None,
            )
        )

        qx = number(
            get_field(
                pose,
                "qx",
                None,
            )
        )

        qy = number(
            get_field(
                pose,
                "qy",
                None,
            )
        )

        qz = number(
            get_field(
                pose,
                "qz",
                None,
            )
        )

        if None not in (
            qw,
            qx,
            qy,
            qz,
        ):

            yaw = quaternion_yaw(
                float(qw),
                float(qx),
                float(qy),
                float(qz),
            )

    # Nested quaternion.
    if yaw is None:

        orientation = (
            get_field(
                pose,
                "orientation",
                None,
            )
            or
            get_field(
                pose,
                "quaternion",
                None,
            )
        )

        if orientation is not None:

            qw = number(
                get_field(
                    orientation,
                    "w",
                    None,
                )
            )

            if qw is None:

                qw = number(
                    get_field(
                        orientation,
                        "qw",
                        None,
                    )
                )

            qx = number(
                get_field(
                    orientation,
                    "x",
                    None,
                )
            )

            if qx is None:

                qx = number(
                    get_field(
                        orientation,
                        "qx",
                        None,
                    )
                )

            qy = number(
                get_field(
                    orientation,
                    "y",
                    None,
                )
            )

            if qy is None:

                qy = number(
                    get_field(
                        orientation,
                        "qy",
                        None,
                    )
                )

            qz = number(
                get_field(
                    orientation,
                    "z",
                    None,
                )
            )

            if qz is None:

                qz = number(
                    get_field(
                        orientation,
                        "qz",
                        None,
                    )
                )

            if None not in (
                qw,
                qx,
                qy,
                qz,
            ):

                yaw = quaternion_yaw(
                    float(qw),
                    float(qx),
                    float(qy),
                    float(qz),
                )

    if yaw is None:

        # Some nuPlan-style objects.
        rear_axle = get_field(
            ego_state,
            "rear_axle",
            None,
        )

        yaw = number(
            get_field(
                rear_axle,
                "heading",
                None,
            )
        )

    if yaw is None:

        raise ValueError(
            "Could not extract ego heading/yaw"
        )

    return (
        float(x),
        float(y),
        float(yaw),
    )


# =============================================================================
# ACCELERATION
# =============================================================================

def extract_acceleration_mps2(
    ego_state: Any,
    heading: float,
) -> Optional[float]:
    """
    Signed longitudinal acceleration.

    Positive:
        acceleration in ego forward direction.

    Negative:
        braking/deceleration.
    """

    for container in nested_containers(
        ego_state
    ):

        for name in (
            "longitudinal_acceleration_mps2",
            "longitudinal_acceleration",
            "acceleration_mps2",
            "accel_mps2",
        ):

            value = number(
                get_field(
                    container,
                    name,
                    None,
                )
            )

            if value is not None:
                return float(value)

    for container in nested_containers(
        ego_state
    ):

        for name in (
            "acceleration",
            "linear_acceleration",
            "accel",
            "ego_acceleration",
            "rear_axle_acceleration_2d",
        ):

            vec = vector_xy(
                get_field(
                    container,
                    name,
                    None,
                ),
                (
                    "x",
                    "ax",
                ),
                (
                    "y",
                    "ay",
                ),
            )

            if vec is not None:

                ax, ay = vec

                return float(
                    ax
                    *
                    math.cos(
                        heading
                    )
                    +
                    ay
                    *
                    math.sin(
                        heading
                    )
                )

    return None


# =============================================================================
# TRAJECTORY
#
# GT:
# trajectory_future
#
# Original:
# global frame / 10 Hz
#
# Convert:
# 0.5, 1.0, ... 5.0 sec
# -> current ego frame
# -> 10 x [x, y]
# =============================================================================

def future_array(
    value: Any,
) -> Optional[np.ndarray]:

    if value is None:
        return None

    try:

        arr = np.asarray(
            value,
            dtype=np.float64,
        )

        if (
            arr.ndim == 2
            and arr.shape[0] >= 2
            and arr.shape[1] >= 2
        ):
            return arr

    except Exception:
        pass

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        rows = []

        for point in value:

            if not isinstance(
                point,
                dict,
            ):
                continue

            x = number(
                get_ci(
                    point,
                    [
                        "x",
                        "position_x",
                        "px",
                    ],
                )
            )

            y = number(
                get_ci(
                    point,
                    [
                        "y",
                        "position_y",
                        "py",
                    ],
                )
            )

            yaw = number(
                get_ci(
                    point,
                    [
                        "yaw",
                        "heading",
                        "theta",
                    ],
                )
            )

            if (
                x is not None
                and y is not None
            ):

                rows.append(
                    [
                        x,
                        y,
                        0.0
                        if yaw is None
                        else yaw,
                    ]
                )

        if len(rows) >= 2:

            return np.asarray(
                rows,
                dtype=np.float64,
            )

    return None


def trajectory_to_ego_xy_5s(
    ego_state: Any,
) -> Optional[
    List[List[float]]
]:

    x0, y0, yaw0 = extract_pose(
        ego_state
    )

    future = get_field(
        ego_state,
        "trajectory_future",
        None,
    )

    arr = future_array(
        future
    )

    if arr is None:
        return None

    # -------------------------------------------------------------------------
    # nuReasoning ego trajectory: 10 Hz
    # -------------------------------------------------------------------------

    first_distance = np.linalg.norm(
        arr[0, :2]
        -
        np.array(
            [
                x0,
                y0,
            ],
            dtype=np.float64,
        )
    )

    first_is_current = (
        first_distance < 0.25
    )

    if first_is_current:

        t_raw = (
            np.arange(
                arr.shape[0],
                dtype=np.float64,
            )
            *
            0.1
        )

    else:

        t_raw = (
            np.arange(
                arr.shape[0],
                dtype=np.float64,
            )
            +
            1.0
        ) * 0.1

    if (
        t_raw[-1]
        <
        5.0 - 1e-6
    ):
        return None

    # -------------------------------------------------------------------------
    # 0.5 sec interval -> 10 points
    # -------------------------------------------------------------------------

    target_t = np.arange(
        0.5,
        5.0 + 1e-6,
        0.5,
    )

    gx = np.interp(
        target_t,
        t_raw,
        arr[:, 0],
    )

    gy = np.interp(
        target_t,
        t_raw,
        arr[:, 1],
    )

    # -------------------------------------------------------------------------
    # Global -> current ego frame
    # -------------------------------------------------------------------------

    dx = gx - x0
    dy = gy - y0

    c = math.cos(
        yaw0
    )

    s = math.sin(
        yaw0
    )

    # x = forward
    ex = (
        c * dx
        +
        s * dy
    )

    # y = left
    ey = (
        -s * dx
        +
        c * dy
    )

    trajectory = np.stack(
        [
            ex,
            ey,
        ],
        axis=1,
    )

    if trajectory.shape != (
        10,
        2,
    ):
        return None

    if not np.isfinite(
        trajectory
    ).all():
        return None

    return np.round(
        trajectory,
        4,
    ).tolist()


# =============================================================================
# LABEL HELPERS
# =============================================================================

def normalize_lon(
    value: str,
) -> str:

    mapping = {
        "Gently accelerate (speed up)":
        "Gently accelerate",
    }

    return mapping.get(
        value,
        value,
    )


def mission_command(
    frame: Dict[str, Any],
) -> str:

    mission = (
        get_ci(
            frame,
            [
                "mission_goal",
                "mission",
            ],
            {},
        )
        or {}
    )

    command = get_ci(
        mission,
        [
            "command",
            "mission_command",
        ],
        None,
    )

    if command is None:

        command = get_ci(
            frame,
            [
                "mission_command",
                "command",
            ],
            "UNKNOWN",
        )

    return (
        str(command).strip()
        or
        "UNKNOWN"
    )


# =============================================================================
# SPATIAL HELPERS
# =============================================================================

def short_category(
    obj: Dict[str, Any],
) -> str:

    label = str(
        get_ci(
            obj,
            [
                "detection_label",
                "label",
            ],
            "",
        )
    ).strip()

    if label:
        return label

    category = str(
        get_ci(
            obj,
            [
                "category",
                "object_category",
            ],
            "object",
        )
    ).strip()

    if category:

        return category.split(
            "."
        )[-1]

    return "object"


def spatial_center(
    obj: Dict[str, Any],
) -> Optional[
    Tuple[float, float]
]:

    bbox3d = (
        get_ci(
            obj,
            [
                "detection_bbox_3d",
                "bbox_3d",
            ],
            {},
        )
        or {}
    )

    center = (
        get_ci(
            bbox3d,
            [
                "center_3d_ego",
                "center",
                "position_3d_ego",
            ],
            {},
        )
        or {}
    )

    x = number(
        get_ci(
            center,
            [
                "x",
                "longitudinal",
                "ahead_m",
            ],
        )
    )

    y = number(
        get_ci(
            center,
            [
                "y",
                "lateral",
                "left_m",
            ],
        )
    )

    if (
        x is None
        or y is None
    ):
        return None

    return (
        float(x),
        float(y),
    )


def normalize_bbox_1000(
    bbox: Any,
    image_path: str,
) -> Optional[List[int]]:

    if (
        not isinstance(
            bbox,
            (
                list,
                tuple,
            ),
        )
        or
        len(bbox) != 4
    ):
        return None

    values = [
        number(x)
        for x in bbox
    ]

    if any(
        value is None
        for value in values
    ):
        return None

    try:

        with Image.open(
            image_path
        ) as image:

            width, height = (
                image.size
            )

    except Exception:
        return None

    if (
        width <= 0
        or height <= 0
    ):
        return None

    x1, y1, x2, y2 = [
        float(x)
        for x in values
    ]

    result = [
        int(
            round(
                max(
                    0.0,
                    min(
                        1000.0,
                        1000.0
                        *
                        x1
                        /
                        width,
                    ),
                )
            )
        ),

        int(
            round(
                max(
                    0.0,
                    min(
                        1000.0,
                        1000.0
                        *
                        y1
                        /
                        height,
                    ),
                )
            )
        ),

        int(
            round(
                max(
                    0.0,
                    min(
                        1000.0,
                        1000.0
                        *
                        x2
                        /
                        width,
                    ),
                )
            )
        ),

        int(
            round(
                max(
                    0.0,
                    min(
                        1000.0,
                        1000.0
                        *
                        y2
                        /
                        height,
                    ),
                )
            )
        ),
    ]

    if (
        result[2] <= result[0]
        or
        result[3] <= result[1]
    ):
        return None

    return result


# =============================================================================
# COUNTERFACTUAL
# =============================================================================

def counterfactual_cases(
    counterfactual: Any,
) -> List[Dict[str, Any]]:

    if not isinstance(
        counterfactual,
        dict,
    ):
        return []

    cases = []

    for key in (
        "Alternative actions",
        "alternative_actions",
        "Top safety-critical actions",
        "top_safety_critical_actions",
    ):

        items = get_ci(
            counterfactual,
            [key],
            None,
        )

        if not isinstance(
            items,
            list,
        ):
            continue

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            lon = get_ci(
                item,
                [
                    "Longitudinal",
                    "lon",
                ],
                None,
            )

            lat = get_ci(
                item,
                [
                    "Lateral",
                    "lat",
                ],
                None,
            )

            risk = get_ci(
                item,
                [
                    "Risk level",
                    "risk",
                    "risk_level",
                ],
                None,
            )

            if (
                lon is None
                or lat is None
                or risk is None
            ):
                continue

            risk = str(
                risk
            ).strip()

            if risk not in RISK_CLASSES:
                continue

            cases.append(
                {
                    "Longitudinal":
                    normalize_lon(
                        str(lon).strip()
                    ),

                    "Lateral":
                    str(lat).strip(),

                    "risk":
                    risk,
                }
            )

    return cases


# =============================================================================
# BASE RECORD
# =============================================================================

def base_record(
    clip_dir: Path,
    metadata: Dict[str, Any],
    frame: Dict[str, Any],
    images: List[str],
    speed: float,
    acceleration: float,
    heading: float,
) -> Dict[str, Any]:

    frame_idx = number(
        get_ci(
            frame,
            ["frame_index"],
            -1,
        )
    )

    return {
        "clip":
        clip_dir.name,

        "scenario_type":
        str(
            metadata.get(
                "scenario_type",
                "unknown",
            )
        ),

        "frame_index":
        int(
            -1
            if frame_idx is None
            else frame_idx
        ),

        "timestamp_us":
        frame_timestamp(
            frame
        ),

        "images":
        images,

        "speed_mps":
        float(speed),

        "acceleration_mps2":
        float(acceleration),

        "heading_rad":
        float(heading),

        "command":
        mission_command(
            frame
        ),
    }


# =============================================================================
# BUILD ALL 6 TASKS FROM ONE FRAME
# =============================================================================

def collect_rows_from_frame(
    clip_dir: Path,
    metadata: Dict[str, Any],
    frame: Dict[str, Any],
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Build at most ONE row per task from a frame.

    Maximum rows per valid frame = 6:
        reasoning, decision, counterfactual, spatial, spatial_obj, trajectory.

    This is the key anti-explosion change. The old version emitted every
    counterfactual case and many spatial categories / objects from one frame.
    """

    reasoning_path = resolve_reasoning_path(frame, clip_dir)
    ego_path = resolve_path(clip_dir, get_ci(frame, ["ego_state"]))
    images = resolve_three_front_images(frame, clip_dir)

    if reasoning_path is None:
        stats["missing_reasoning"] += 1
        return []
    if ego_path is None or not ego_path.is_file():
        stats["missing_ego"] += 1
        return []
    if images is None:
        stats["missing_images"] += 1
        return []

    try:
        reasoning_obj = read_json(reasoning_path)
        ego_state = load_pickle(ego_path)
    except Exception:
        stats["load_error"] += 1
        return []

    try:
        _, _, heading = extract_pose(ego_state)
        speed = extract_speed_mps(ego_state)
        acceleration = extract_acceleration_mps2(ego_state, heading)
        trajectory = trajectory_to_ego_xy_5s(ego_state)
    except Exception:
        stats["ego_parse_error"] += 1
        return []

    if speed is None:
        stats["missing_speed"] += 1
        return []
    if acceleration is None:
        stats["missing_acceleration"] += 1
        return []
    if not math.isfinite(heading):
        stats["missing_heading"] += 1
        return []

    base = base_record(
        clip_dir, metadata, frame, images, speed, acceleration, heading
    )
    token = str(
        get_ci(frame, ["token"], "")
        or f"{clip_dir.name}_{base['frame_index']}_{base['timestamp_us']}"
    )
    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Driving annotations
    # ------------------------------------------------------------------
    driving = get_ci(reasoning_obj, ["Driving"], {}) or {}
    decision = get_ci(
        driving, ["Driving decision", "driving_decision"], {}
    ) or {}
    reasoning_trace = str(
        get_ci(
            driving,
            ["Reasoning trace", "reasoning_trace", "Reasoning"],
            "",
        )
    ).strip()
    lon = normalize_lon(str(get_ci(decision, ["Longitudinal"], "")).strip())
    lat = str(get_ci(decision, ["Lateral"], "")).strip()

    if reasoning_trace:
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:reasoning",
                "task": "reasoning",
                "gt": {"reasoning": reasoning_trace},
            }
        )
        rows.append(rec)
        stats["task_reasoning"] += 1
    else:
        stats["empty_reasoning_trace"] += 1

    if lon in LON_CLASSES and lat in LAT_CLASSES:
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:decision",
                "task": "decision",
                "gt": {"lon": lon, "lat": lat},
            }
        )
        rows.append(rec)
        stats["task_decision"] += 1
    elif lon or lat:
        stats["decision_outside_classes"] += 1

    # ------------------------------------------------------------------
    # Counterfactual: ONE case only.
    # ------------------------------------------------------------------
    cf = get_ci(reasoning_obj, ["Counterfactual"], {}) or {}
    cf_cases = counterfactual_cases(cf)
    if cf_cases:
        case = cf_cases[0]
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:counterfactual:0",
                "task": "counterfactual",
                "action": {
                    "Longitudinal": case["Longitudinal"],
                    "Lateral": case["Lateral"],
                },
                "gt": {"risk": case["risk"]},
            }
        )
        rows.append(rec)
        stats["task_counterfactual"] += 1
        stats["counterfactual_dropped_by_frame_cap"] += max(0, len(cf_cases) - 1)

    # ------------------------------------------------------------------
    # Trajectory: always retain when valid.
    # ------------------------------------------------------------------
    if trajectory is not None and len(trajectory) == 10:
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:trajectory",
                "task": "trajectory",
                "gt": {"waypoints": trajectory},
            }
        )
        rows.append(rec)
        stats["task_trajectory"] += 1
    else:
        stats["bad_trajectory"] += 1

    # ------------------------------------------------------------------
    # Spatial: build candidates, then keep ONE spatial and ONE spatial_obj.
    # For generic spatial, each candidate is already the closest object of
    # that category within that camera; we then keep the closest candidate
    # overall. For spatial_obj we keep the closest object with a valid bbox.
    # ------------------------------------------------------------------
    spatial = get_ci(reasoning_obj, ["Spatial"], {}) or {}
    per_camera = get_ci(
        spatial,
        ["per_camera_results", "per camera results"],
        {},
    ) or {}

    spatial_candidates = []
    spatial_obj_candidates = []

    if isinstance(per_camera, dict):
        for cam_idx, cam_name in enumerate(CAMERA_ORDER):
            cam_data = get_ci(per_camera, [cam_name], {}) or {}
            objects = get_ci(cam_data, ["objects"], []) or []
            if not isinstance(objects, list):
                continue

            grouped = defaultdict(list)
            for obj_idx, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                center = spatial_center(obj)
                if center is None:
                    continue
                ahead, left = center
                category = short_category(obj)
                distance = math.hypot(ahead, left)
                grouped[category].append((distance, ahead, left))

                bbox = get_ci(
                    obj,
                    ["detection_bbox_2d", "bbox_2d", "bbox"],
                    None,
                )
                bbox_1000 = normalize_bbox_1000(bbox, images[cam_idx])
                if bbox_1000 is not None:
                    spatial_obj_candidates.append(
                        (
                            distance,
                            cam_name,
                            obj_idx,
                            category,
                            bbox_1000,
                            ahead,
                            left,
                        )
                    )

            for category, candidates in grouped.items():
                distance, ahead, left = min(candidates, key=lambda x: x[0])
                spatial_candidates.append(
                    (distance, cam_name, category, ahead, left)
                )

    if spatial_candidates:
        distance, cam_name, category, ahead, left = min(
            spatial_candidates, key=lambda x: x[0]
        )
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:spatial:{cam_name}:{category}",
                "task": "spatial",
                "ref_cam": cam_name.replace("_", "-"),
                "category": category,
                "gt": {"ahead_m": float(ahead), "left_m": float(left)},
            }
        )
        rows.append(rec)
        stats["task_spatial"] += 1
        stats["spatial_dropped_by_frame_cap"] += max(
            0, len(spatial_candidates) - 1
        )

    if spatial_obj_candidates:
        (
            distance,
            cam_name,
            obj_idx,
            category,
            bbox_1000,
            ahead,
            left,
        ) = min(spatial_obj_candidates, key=lambda x: x[0])
        rec = dict(base)
        rec.update(
            {
                "id": f"{token}:spatial_obj:{cam_name}:{obj_idx}",
                "task": "spatial_obj",
                "ref_cam": cam_name.replace("_", "-"),
                "category": category,
                "bbox_1000": bbox_1000,
                "gt": {"ahead_m": float(ahead), "left_m": float(left)},
            }
        )
        rows.append(rec)
        stats["task_spatial_obj"] += 1
        stats["spatial_obj_dropped_by_frame_cap"] += max(
            0, len(spatial_obj_candidates) - 1
        )

    return rows



# =============================================================================
# COLLECT ONE CLIP
# =============================================================================

def collect_rows_from_clip(
    clip_dir: Path,
    max_frames_per_clip: Optional[int] = DEFAULT_MAX_FRAMES_PER_CLIP,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Collect rows from actually usable frames, then apply the clip cap.

    Correct order:
        1. scan every metadata frame
        2. build task rows and keep frames that generate >= 1 row
        3. evenly choose at most max_frames_per_clip VALID frames
        4. flatten the selected frame rows

    This prevents the old failure mode where arbitrary metadata frames were
    selected before checking reasoning / ego-state / camera validity.
    """

    stats = defaultdict(int)
    metadata_path = clip_dir / "metadata.json"

    try:
        metadata = read_json(metadata_path)
    except Exception:
        stats["bad_metadata"] += 1
        return [], dict(stats)

    frames = get_ci(
        metadata,
        ["frames"],
        [],
    )

    if not isinstance(
        frames,
        list,
    ):
        stats["bad_frames"] += 1
        return [], dict(stats)

    stats["frames_in_metadata"] = len(frames)

    # -------------------------------------------------------------------------
    # FIRST: scan ALL frames and determine which ones are genuinely usable.
    # -------------------------------------------------------------------------
    valid_frames: List[
        Tuple[
            int,
            List[Dict[str, Any]],
        ]
    ] = []

    for frame_idx, frame in enumerate(
        frames
    ):
        stats["frames_scanned"] += 1
        stats["frames_seen"] += 1

        if not isinstance(
            frame,
            dict,
        ):
            stats["bad_frame"] += 1
            continue

        try:
            frame_rows = collect_rows_from_frame(
                clip_dir,
                metadata,
                frame,
                stats,
            )
        except Exception:
            stats["frame_exception"] += 1
            continue

        if not frame_rows:
            continue

        valid_frames.append(
            (
                frame_idx,
                frame_rows,
            )
        )

    stats["valid_frames_before_cap"] = len(
        valid_frames
    )

    stats["rows_before_clip_cap"] = sum(
        len(frame_rows)
        for _, frame_rows in valid_frames
    )

    # -------------------------------------------------------------------------
    # SECOND: apply cap only AFTER validity is known.
    # -------------------------------------------------------------------------
    selected_frames = valid_frames

    if (
        max_frames_per_clip is not None
        and
        max_frames_per_clip > 0
        and
        len(valid_frames) > max_frames_per_clip
    ):
        picked_positions = np.linspace(
            0,
            len(valid_frames) - 1,
            num=max_frames_per_clip,
            dtype=np.int64,
        )

        picked_positions = sorted(
            {
                int(position)
                for position in picked_positions
            }
        )

        # Defensive fill in case integer conversion creates duplicate positions.
        if len(picked_positions) < max_frames_per_clip:
            used = set(
                picked_positions
            )

            for position in range(
                len(valid_frames)
            ):
                if position in used:
                    continue

                picked_positions.append(
                    position
                )

                used.add(
                    position
                )

                if len(picked_positions) >= max_frames_per_clip:
                    break

            picked_positions.sort()

        selected_frames = [
            valid_frames[position]
            for position in picked_positions[:max_frames_per_clip]
        ]

        stats["frames_dropped_by_clip_cap"] += (
            len(valid_frames)
            -
            len(selected_frames)
        )

    stats["valid_frames_after_cap"] = len(
        selected_frames
    )

    # -------------------------------------------------------------------------
    # Flatten selected frame rows.
    # -------------------------------------------------------------------------
    rows: List[
        Dict[str, Any]
    ] = []

    for _, frame_rows in selected_frames:
        rows.extend(
            frame_rows
        )

    stats["rows"] = len(
        rows
    )

    return (
        rows,
        dict(
            stats
        ),
    )


# =============================================================================
# SPLIT
# =============================================================================

def split_clips(
    clip_rows: Dict[
        str,
        List[Dict[str, Any]],
    ],
    seed: int,
) -> Dict[str, List[str]]:
    """
    Clip-disjoint 70/20/10.

    Balance by generated row count instead of simply clip count.
    """

    clips = list(
        clip_rows.keys()
    )

    if len(clips) < 3:

        raise RuntimeError(
            f"Need >=3 valid clips, "
            f"got {len(clips)}"
        )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        clips
    )

    total_rows = sum(
        len(
            clip_rows[
                clip
            ]
        )
        for clip in clips
    )

    target = {
        "train":
        total_rows
        *
        TRAIN_RATIO,

        "val":
        total_rows
        *
        VAL_RATIO,

        "test":
        total_rows
        *
        TEST_RATIO,
    }

    assigned = {
        "train": [],
        "val": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # Large clips first.
    clips = sorted(
        clips,
        key=lambda clip:
        len(
            clip_rows[
                clip
            ]
        ),
        reverse=True,
    )

    for clip in clips:

        n_rows = len(
            clip_rows[
                clip
            ]
        )

        def normalized_deficit(
            split: str,
        ) -> float:

            return (
                target[
                    split
                ]
                -
                counts[
                    split
                ]
            ) / max(
                1.0,
                target[
                    split
                ],
            )

        split_name = max(
            (
                "train",
                "val",
                "test",
            ),
            key=normalized_deficit,
        )

        assigned[
            split_name
        ].append(
            clip
        )

        counts[
            split_name
        ] += n_rows

    return assigned


# =============================================================================
# DATA PREPARATION
# =============================================================================

def count_tasks(
    rows: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, int]:

    counts = Counter(
        str(
            row.get(
                "task",
                "",
            )
        )
        for row in rows
    )

    return {
        task:
        int(
            counts.get(
                task,
                0,
            )
        )
        for task in TASKS
    }


def cap_rows_by_task(
    rows: Sequence[Dict[str, Any]],
    caps: Dict[str, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Deterministically cap each task without oversampling.

    A cap <= 0 means no cap for that task. Rows from different clips are never
    moved across train/val/test, so clip-disjointness remains intact.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task", ""))].append(row)

    rng = random.Random(seed)
    kept = []
    dropped = {}

    for task in TASKS:
        task_rows = list(grouped.get(task, []))
        cap = int(caps.get(task, 0) or 0)
        if cap > 0 and len(task_rows) > cap:
            rng.shuffle(task_rows)
            dropped[task] = len(task_rows) - cap
            task_rows = task_rows[:cap]
        else:
            dropped[task] = 0
        kept.extend(task_rows)

    kept.sort(key=lambda row: str(row["id"]))
    return kept, dropped


def prepare_dataset(args: argparse.Namespace) -> None:
    install_pickle_aliases()

    root = args.part1_root
    if not root.is_dir():
        raise FileNotFoundError(root)

    clip_dirs = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        ]
    )
    if getattr(args, "max_clips", None) is not None:
        clip_dirs = clip_dirs[: args.max_clips]

    max_frames_per_clip = getattr(
        args, "max_frames_per_clip", DEFAULT_MAX_FRAMES_PER_CLIP
    )
    task_caps = getattr(args, "task_caps", DEFAULT_PREP_TASK_CAPS)

    print()
    print("=" * 110)
    print("PREPARE NUREASONING PART1 | COMPACT MULTITASK")
    print("=" * 110)
    print("Part1              :", root)
    print("Clips              :", len(clip_dirs))
    print("Max frames / clip  :", max_frames_per_clip)
    print("Max rows / frame   : 6 (one per task)")
    print("Task caps          :", task_caps)

    clip_rows = {}
    total_stats = defaultdict(int)

    for index, clip_dir in enumerate(clip_dirs, 1):
        rows, stats = collect_rows_from_clip(
            clip_dir,
            max_frames_per_clip=max_frames_per_clip,
        )
        if rows:
            clip_rows[clip_dir.name] = rows
        for key, value in stats.items():
            total_stats[key] += value

        if index % 25 == 0 or index == len(clip_dirs):
            candidate_rows = sum(len(x) for x in clip_rows.values())
            print(
                f"[PREP] {index:5d}/{len(clip_dirs):5d} "
                f"| valid clips={len(clip_rows):5d} "
                f"| candidate rows={candidate_rows:7d}"
            )

    if not clip_rows:
        raise RuntimeError("No valid dataset rows generated.")

    assignment = split_clips(clip_rows, args.seed)
    split_rows = {}
    pre_cap_counts = {}
    dropped_by_cap = {}

    for split_index, split_name in enumerate(("train", "val", "test")):
        rows = []
        for clip in assignment[split_name]:
            rows.extend(clip_rows[clip])

        pre_cap_counts[split_name] = count_tasks(rows)
        rows, dropped = cap_rows_by_task(
            rows,
            task_caps.get(split_name, {}),
            seed=args.seed + 1009 * (split_index + 1),
        )
        dropped_by_cap[split_name] = dropped
        split_rows[split_name] = rows

    train_clips = set(assignment["train"])
    val_clips = set(assignment["val"])
    test_clips = set(assignment["test"])
    assert not (train_clips & val_clips)
    assert not (train_clips & test_clips)
    assert not (val_clips & test_clips)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        write_jsonl(args.data_dir / f"{split_name}.jsonl", split_rows[split_name])

    manifest = {
        "seed": args.seed,
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "compact_policy": {
            "max_frames_per_clip": max_frames_per_clip,
            "max_rows_per_frame": 6,
            "counterfactual_per_frame": 1,
            "spatial_per_frame": 1,
            "spatial_obj_per_frame": 1,
            "task_caps": task_caps,
        },
        "task_definition": list(TASKS),
        "legacy_tasks": list(LEGACY_TASKS),
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
            "horizon_sec": 5.0,
            "interval_sec": 0.5,
            "num_waypoints": 10,
            "frame": "current_ego",
            "axes": {"x": "forward", "y": "left"},
        },
        "clips": {split: assignment[split] for split in ("train", "val", "test")},
        "rows_before_task_cap": {
            split: sum(pre_cap_counts[split].values())
            for split in ("train", "val", "test")
        },
        "task_counts_before_cap": pre_cap_counts,
        "dropped_by_task_cap": dropped_by_cap,
        "rows": {
            split: len(split_rows[split]) for split in ("train", "val", "test")
        },
        "task_counts": {
            split: count_tasks(split_rows[split])
            for split in ("train", "val", "test")
        },
        "prepare_stats": dict(total_stats),
    }
    save_json(args.data_dir / "manifest.json", manifest)

    print()
    print("=" * 110)
    print("DATASET READY")
    print("=" * 110)
    for split_name in ("train", "val", "test"):
        print()
        print(split_name.upper())
        print("  clips       :", len(assignment[split_name]))
        print("  before cap  :", pre_cap_counts[split_name])
        print("  after cap   :", count_tasks(split_rows[split_name]))
        print("  total rows  :", len(split_rows[split_name]))
        print("  dropped     :", dropped_by_cap[split_name])



# =============================================================================
# PROMPT
# =============================================================================

def build_context(
    rec: Dict[str, Any],
) -> str:

    return (
        "You are an autonomous-driving assistant.\n"
        "Three synchronized camera views are provided in this order: "
        "front-left, front, front-right.\n\n"

        "Driving context:\n"
        f"- Mission command: {rec['command']}\n"
        f"- Current speed: {float(rec['speed_mps']):.3f} m/s\n"
        f"- Current signed longitudinal acceleration: "
        f"{float(rec['acceleration_mps2']):.3f} m/s^2 "
        "(positive means accelerating forward, negative means decelerating)\n"
        f"- Current heading/yaw from the ego state: "
        f"{float(rec['heading_rad']):.6f} rad\n\n"

        "Use the camera observations together with the mission command and "
        "ego-state dynamics when they are relevant. "
        "Treat heading/yaw as the current orientation value, not by itself "
        "as evidence that the vehicle is turning.\n\n"
    )


def build_prompt(
    rec: Dict[str, Any],
) -> str:

    task = rec[
        "task"
    ]

    # =========================================================================
    # NEW REASONING
    # =========================================================================

    if task == "reasoning":

        return (
            build_context(
                rec
            )
            +
            "Generate the driving reasoning for the current situation. "
            "Use the three camera views, mission command, current speed, "
            "signed longitudinal acceleration, and heading/yaw when relevant. "
            "Explain the important scene evidence, safety constraints, route "
            "intention, and ego-motion context that justify the appropriate "
            "driving behavior. "
            "Do not output a separate final action label, JSON, or trajectory. "
            "Return only the natural-language reasoning trace."
        )

    # =========================================================================
    # ORIGINAL DECISION
    # =========================================================================

    if task == "decision":

        return (
            build_context(
                rec
            )
            +
            "Decide the driving action for this moment.\n"
            +
            "Longitudinal must be exactly one of: "
            +
            ", ".join(
                LON_CLASSES
            )
            +
            ".\n"
            +
            "Lateral must be exactly one of: "
            +
            ", ".join(
                LAT_CLASSES
            )
            +
            ".\n"
            +
            'Answer with JSON only: '
            '{"Longitudinal": "...", "Lateral": "..."}'
        )

    # =========================================================================
    # ORIGINAL COUNTERFACTUAL
    # =========================================================================

    if task == "counterfactual":

        action = rec[
            "action"
        ]

        return (
            build_context(
                rec
            )
            +
            f'Consider this alternative action: '
            f'Longitudinal = "{action["Longitudinal"]}", '
            f'Lateral = "{action["Lateral"]}".\n'
            +
            "How risky would executing this action be right now? "
            +
            "Answer with exactly one word out of: "
            +
            ", ".join(
                RISK_CLASSES
            )
            +
            "."
        )

    # =========================================================================
    # ORIGINAL SPATIAL
    # =========================================================================

    if task == "spatial":

        return (
            build_context(
                rec
            )
            +
            f"Look at the "
            f"{rec['ref_cam']} view. "
            f"What is the closest "
            f"{rec['category']} "
            f"in that view? "
            +
            "Report its distance ahead and lateral offset in meters, "
            +
            'as JSON only: '
            '{"ahead_m": <number>, "left_m": <number>} '
            +
            "(left_m is positive to the left, negative to the right)."
        )

    # =========================================================================
    # ORIGINAL SPATIAL OBJECT
    # =========================================================================

    if task == "spatial_obj":

        bbox = rec[
            "bbox_1000"
        ]

        return (
            build_context(
                rec
            )
            +
            f"In the "
            f"{rec['ref_cam']} view, "
            f"consider the "
            f"{rec['category']} inside the "
            f"2D bounding box "
            f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] "
            "(pixel coordinates normalized to 0-1000, "
            "format [x1, y1, x2, y2]).\n"
            +
            "Report that object's position in the ego frame: "
            "distance ahead and lateral offset in meters, "
            +
            'as JSON only: '
            '{"ahead_m": <number>, "left_m": <number>} '
            +
            "(left_m is positive to the left, negative to the right)."
        )

    # =========================================================================
    # ORIGINAL DIRECT TRAJECTORY
    # =========================================================================

    if task == "trajectory":

        return (
            build_context(
                rec
            )
            +
            "Predict the vehicle's trajectory for the next 5 seconds "
            "as 10 waypoints at 0.5 s intervals, "
            "in the ego frame "
            "(x forward, y left, meters).\n"
            +
            "Answer with exactly 10 comma-separated pairs like: "
            +
            "(x1,y1), (x2,y2), ... "
            "with one decimal place."
        )

    raise ValueError(
        f"Unknown task: {task}"
    )


# =============================================================================
# TARGET FORMAT
# =============================================================================

def format_target(
    rec: Dict[str, Any],
) -> str:

    task = rec[
        "task"
    ]

    if task == "reasoning":

        return str(
            rec[
                "gt"
            ][
                "reasoning"
            ]
        ).strip()

    if task == "decision":

        return json.dumps(
            {
                "Longitudinal":
                normalize_lon(
                    str(
                        rec[
                            "gt"
                        ][
                            "lon"
                        ]
                    ).strip()
                ),

                "Lateral":
                str(
                    rec[
                        "gt"
                    ][
                        "lat"
                    ]
                ).strip(),
            },

            ensure_ascii=False,
        )

    if task == "counterfactual":

        return str(
            rec[
                "gt"
            ][
                "risk"
            ]
        ).strip()

    if task in (
        "spatial",
        "spatial_obj",
    ):

        return json.dumps(
            {
                "ahead_m":
                rec[
                    "gt"
                ][
                    "ahead_m"
                ],

                "left_m":
                rec[
                    "gt"
                ][
                    "left_m"
                ],
            },

            ensure_ascii=False,
        )

    if task == "trajectory":

        waypoints = rec[
            "gt"
        ][
            "waypoints"
        ]

        if len(
            waypoints
        ) != 10:

            raise ValueError(
                "Trajectory must have exactly 10 points."
            )

        return ", ".join(
            f"({float(x):.1f},{float(y):.1f})"
            for x, y
            in waypoints
        )

    raise ValueError(
        task
    )


# =============================================================================
# DATASET
# =============================================================================

class JsonlDataset(
    Dataset
):

    def __init__(
        self,
        rows: Sequence[
            Dict[str, Any]
        ],
    ):

        self.rows = list(
            rows
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        return self.rows[
            index
        ]


# =============================================================================
# IMAGE LOADING
# =============================================================================

def open_three_images(
    rec: Dict[str, Any],
) -> List[Image.Image]:

    paths = rec.get(
        "images"
    )

    if (
        not isinstance(
            paths,
            list,
        )
        or
        len(paths) != 3
    ):

        raise ValueError(
            f"Exactly 3 images required: "
            f"{rec.get('id')}"
        )

    result = []

    for value in paths:

        path = Path(
            str(value)
        )

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

        with Image.open(
            path
        ) as image:

            result.append(
                image
                .convert(
                    "RGB"
                )
                .copy()
            )

    return result


# =============================================================================
# COLLATOR
# =============================================================================

class QwenVLCollator:

    def __init__(
        self,
        processor,
        max_seq_len: int,
    ):

        self.processor = processor

        self.max_seq_len = int(
            max_seq_len
        )

    @staticmethod
    def common_prefix_len(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> int:

        n = min(
            a.numel(),
            b.numel(),
        )

        if n <= 0:
            return 0

        equal = (
            a[:n]
            ==
            b[:n]
        )

        if bool(
            equal.all()
        ):
            return n

        mismatch = (
            ~equal
        ).nonzero(
            as_tuple=False
        )

        return int(
            mismatch[
                0
            ].item()
        )

    def __call__(
        self,
        batch: Sequence[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        if len(batch) != 1:

            raise ValueError(
                "micro batch must be 1"
            )

        rec = batch[
            0
        ]

        prompt = build_prompt(
            rec
        )

        target = format_target(
            rec
        )

        images = open_three_images(
            rec
        )

        # ---------------------------------------------------------------------
        # User content:
        # exactly three image placeholders.
        # ---------------------------------------------------------------------

        user_content = [
            {
                "type":
                "text",

                "text":
                "Front-left camera:",
            },

            {
                "type":
                "image",
            },

            {
                "type":
                "text",

                "text":
                "Front camera:",
            },

            {
                "type":
                "image",
            },

            {
                "type":
                "text",

                "text":
                "Front-right camera:",
            },

            {
                "type":
                "image",
            },

            {
                "type":
                "text",

                "text":
                prompt,
            },
        ]

        # ---------------------------------------------------------------------
        # Prompt only
        # ---------------------------------------------------------------------

        prompt_messages = [
            {
                "role":
                "user",

                "content":
                user_content,
            }
        ]

        prompt_text = (
            self.processor
            .apply_chat_template(
                prompt_messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )

        # ---------------------------------------------------------------------
        # Prompt + GT assistant output
        # ---------------------------------------------------------------------

        full_messages = [
            {
                "role":
                "user",

                "content":
                user_content,
            },

            {
                "role":
                "assistant",

                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        target,
                    }
                ],
            },
        ]

        full_text = (
            self.processor
            .apply_chat_template(
                full_messages,

                tokenize=False,

                add_generation_prompt=False,
            )
        )

        # ---------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------
#
# Qwen3-VL IMPORTANT:
#
# Never truncate a multimodal sequence inside processor().
# If truncation cuts the repeated image tokens while the image features
# are still present, Transformers raises:
#
#   Mismatch in `image` token count between text and `input_ids`
#
# Control sequence length through the image pixel budget instead.
# ---------------------------------------------------------------------

        prompt_inputs = self.processor(
            text=[
                prompt_text
            ],

            images=images,

            return_tensors="pt",

            padding=False,

            truncation=False,
        )

        full_inputs = self.processor(
            text=[
                full_text
            ],

            images=images,

            return_tensors="pt",

            padding=False,

            truncation=False,
        )

        prompt_seq_len = int(
            prompt_inputs[
                "input_ids"
            ].shape[1]
        )

        full_seq_len = int(
            full_inputs[
                "input_ids"
            ].shape[1]
        )

        # Multimodal token streams must NEVER be blindly truncated.
        # If this fires, reduce image resolution instead.
        if (
            prompt_seq_len > self.max_seq_len
            or
            full_seq_len > self.max_seq_len
        ):

            raise RuntimeError(
                "\n"
                "Multimodal sequence exceeds configured max_seq_len.\n"
                f"id={rec.get('id')}\n"
                f"task={rec['task']}\n"
                f"prompt_seq_len={prompt_seq_len}\n"
                f"full_seq_len={full_seq_len}\n"
                f"max_seq_len={self.max_seq_len}\n"
                f"min_pixels={MIN_PIXELS}\n"
                f"max_pixels={MAX_PIXELS}\n"
                "\n"
                "Do NOT enable truncation for Qwen3-VL.\n"
                "Reduce the image pixel budget instead."
            )

        prompt_ids = (
            prompt_inputs[
                "input_ids"
            ][0]
        )

        full_ids = (
            full_inputs[
                "input_ids"
            ][0]
        )

        prefix_len = (
            self.common_prefix_len(
                prompt_ids,
                full_ids,
            )
        )

        labels = (
            full_inputs[
                "input_ids"
            ]
            .clone()
        )

        if (
            prefix_len <= 0
            or
            prefix_len
            >=
            labels.shape[1]
        ):

            raise RuntimeError(
                "\nInvalid assistant boundary\n"
                f"id={rec.get('id')}\n"
                f"task={rec['task']}\n"
                f"prefix_len={prefix_len}\n"
                f"seq_len={labels.shape[1]}\n"
            )

        # ---------------------------------------------------------------------
        # ASSISTANT-ONLY CE
        # ---------------------------------------------------------------------

        labels[
            :,
            :prefix_len,
        ] = -100

        if "attention_mask" in full_inputs:

            labels[
                full_inputs[
                    "attention_mask"
                ]
                ==
                0
            ] = -100

        supervised = int(
            (
                labels
                !=
                -100
            )
            .sum()
            .item()
        )

        if supervised <= 0:

            raise RuntimeError(
                f"No supervised assistant tokens: "
                f"{rec.get('id')}"
            )

        full_inputs[
            "labels"
        ] = labels

        # Metadata for training loop.
        full_inputs[
            "_task"
        ] = rec[
            "task"
        ]

        full_inputs[
            "_id"
        ] = rec[
            "id"
        ]

        full_inputs[
            "_supervised_tokens"
        ] = supervised

        return full_inputs


# =============================================================================
# PROCESSOR
# =============================================================================

def load_processor(
    model_path: Path,
):

    sources = [
        model_path,
        PROCESSOR_FALLBACK_PATH,
    ]

    last_error = None

    for source in sources:

        if not source.exists():
            continue

        try:

            # IMPORTANT:
            # Qwen3-VL image token budget must be supplied when the
            # processor is constructed.
            processor = (
                AutoProcessor
                .from_pretrained(
                    str(source),

                    trust_remote_code=True,
                    local_files_only=True,

                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
            )

            image_processor = getattr(
                processor,
                "image_processor",
                None,
            )

            print(
                "Processor:",
                source,
            )

            print(
                "Image min/max pixels:",
                f"{MIN_PIXELS:,}",
                "/",
                f"{MAX_PIXELS:,}",
            )

            if image_processor is not None:

                print(
                    "Image processor size:",
                    getattr(
                        image_processor,
                        "size",
                        None,
                    ),
                )

            return processor

        except Exception as exc:

            last_error = exc

            print(
                f"[WARN] Processor load failed: "
                f"{source}"
            )

            print(
                f"       {type(exc).__name__}: "
                f"{exc}"
            )

    raise RuntimeError(
        f"Processor load failed: "
        f"{last_error}"
    )


# =============================================================================
# MODEL
# =============================================================================

def load_base_model(
    model_path: Path,
    device: Optional[
        torch.device
    ] = None,
):

    kwargs = {
        "torch_dtype":
        DTYPE,

        "attn_implementation":
        ATTN_IMPLEMENTATION,

        "trust_remote_code":
        True,

        "local_files_only":
        True,

        "low_cpu_mem_usage":
        True,
    }

    model = (
        AutoModelForImageTextToText
        .from_pretrained(
            str(model_path),
            **kwargs,
        )
    )

    model.config.use_cache = False

    if device is not None:

        model.to(
            device
        )

    return model


# =============================================================================
# LORA
# =============================================================================

def attach_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
):

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    config = LoraConfig(
        r=rank,

        lora_alpha=alpha,

        lora_dropout=dropout,

        bias="none",

        task_type=(
            TaskType.CAUSAL_LM
        ),

        target_modules=(
            target_modules
        ),
    )

    model = get_peft_model(
        model,
        config,
    )

    # Gradient checkpointing.
    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):

        try:

            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant":
                    False,
                }
            )

        except TypeError:

            model.gradient_checkpointing_enable()

    if hasattr(
        model,
        "enable_input_require_grads",
    ):

        try:
            model.enable_input_require_grads()

        except Exception:
            pass

    return (
        model,
        target_modules,
    )


def print_trainable(
    model,
) -> None:

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Trainable:",
        f"{trainable:,}",
    )

    print(
        "Total    :",
        f"{total:,}",
    )

    print(
        "Percent  :",
        f"{100.0 * trainable / total:.4f}%",
    )

    if hasattr(
        model,
        "print_trainable_parameters",
    ):

        model.print_trainable_parameters()


# =============================================================================
# TASK WEIGHTS
# =============================================================================

def inverse_sqrt_task_weights(
    counts: Dict[str, int],
) -> Dict[str, float]:
    """
    No oversampling.

    weight_t ∝ 1 / sqrt(N_t)

    Scale weights so expected per-sample weight
    under natural distribution is approximately 1.
    """

    raw = {}

    for task in TASKS:

        count = counts.get(
            task,
            0,
        )

        if count > 0:

            raw[
                task
            ] = 1.0 / math.sqrt(
                count
            )

    if not raw:
        raise RuntimeError(
            "No task weights."
        )

    total_count = sum(
        counts[
            task
        ]
        for task in raw
    )

    weighted_sum = sum(
        counts[
            task
        ]
        *
        raw[
            task
        ]
        for task in raw
    )

    scale = (
        total_count
        /
        weighted_sum
    )

    return {
        task:
        raw[
            task
        ]
        *
        scale

        for task in raw
    }


# =============================================================================
# MOVE BATCH
# =============================================================================

def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
):

    task = batch.pop(
        "_task"
    )

    sample_id = batch.pop(
        "_id"
    )

    supervised_tokens = (
        batch.pop(
            "_supervised_tokens"
        )
    )

    moved = {}

    for key, value in batch.items():

        if torch.is_tensor(
            value
        ):

            value = value.to(
                device,
                non_blocking=True,
            )

            if value.is_floating_point():

                value = value.to(
                    DTYPE
                )

        moved[
            key
        ] = value

    return (
        moved,
        task,
        sample_id,
        supervised_tokens,
    )


def format_duration(
    seconds: float,
) -> str:
    """Format seconds as HH:MM:SS."""

    try:
        seconds = max(
            0,
            int(
                round(
                    float(seconds)
                )
            ),
        )
    except Exception:
        return "--:--:--"

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate_task_losses(
    model,
    loader: DataLoader,
    device: torch.device,
    phase: str = "VAL",
    log_every: int = 100,
) -> Dict[str, Any]:

    model.eval()

    sums = defaultdict(
        float
    )

    counts = defaultdict(
        int
    )

    raw_sum = 0.0

    start = time.perf_counter()

    total_batches = len(
        loader
    )

    for index, batch in enumerate(
        loader,
        1,
    ):

        (
            inputs,
            task,
            _,
            _,
        ) = move_batch(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=DTYPE,
        ):

            outputs = model(
                **inputs
            )

            loss = outputs.loss

        value = float(
            loss
            .detach()
            .float()
            .item()
        )

        sums[
            task
        ] += value

        counts[
            task
        ] += 1

        raw_sum += value

        del (
            inputs,
            outputs,
            loss,
        )

        if (
            index
            %
            max(
                1,
                int(log_every),
            )
            ==
            0
            or
            index
            ==
            total_batches
        ):
            elapsed = (
                time.perf_counter()
                -
                start
            )

            sec_per_sample = (
                elapsed
                /
                max(
                    1,
                    index,
                )
            )

            remaining = (
                sec_per_sample
                *
                max(
                    0,
                    total_batches
                    -
                    index,
                )
            )

            print(
                f"[{phase}] "
                f"{index:5d}/"
                f"{total_batches:5d} "
                f"| elapsed="
                f"{format_duration(elapsed)} "
                f"| eta="
                f"{format_duration(remaining)} "
                f"| {sec_per_sample:.3f}s/sample"
            )

    task_mean = {}

    for task in TASKS:

        if counts[
            task
        ]:

            task_mean[
                task
            ] = (
                sums[
                    task
                ]
                /
                counts[
                    task
                ]
            )

        else:

            task_mean[
                task
            ] = float(
                "nan"
            )

    valid_losses = [
        loss
        for loss in task_mean.values()
        if not math.isnan(
            loss
        )
    ]

    balanced = (
        sum(
            valid_losses
        )
        /
        len(
            valid_losses
        )
        if valid_losses
        else float(
            "inf"
        )
    )

    elapsed_total = (
        time.perf_counter()
        -
        start
    )

    model.train()

    return {
        "balanced":
        float(
            balanced
        ),

        "task_mean":
        task_mean,

        "task_count":
        dict(
            counts
        ),

        "raw_mean":
        (
            raw_sum
            /
            max(
                1,
                sum(
                    counts.values()
                ),
            )
        ),

        "seconds":
        elapsed_total,
    }


# =============================================================================
# LEGACY REGRESSION GUARD
# =============================================================================

def legacy_regression_guard(
    current_task_mean: Dict[
        str,
        float,
    ],
    baseline_task_mean: Dict[
        str,
        float,
    ],
) -> Tuple[
    float,
    Dict[str, float],
]:

    penalties = {}

    for task in LEGACY_TASKS:

        current = (
            current_task_mean
            .get(
                task,
                float("nan"),
            )
        )

        baseline = (
            baseline_task_mean
            .get(
                task,
                float("nan"),
            )
        )

        if (
            math.isnan(
                current
            )
            or
            math.isnan(
                baseline
            )
            or
            baseline <= 0
        ):

            continue

        relative = (
            current
            /
            baseline
            -
            1.0
        )

        penalty = max(
            0.0,
            relative
            -
            LEGACY_RELATIVE_TOLERANCE,
        )

        penalties[
            task
        ] = penalty

    guard = (
        float(
            np.mean(
                list(
                    penalties.values()
                )
            )
        )
        if penalties
        else 0.0
    )

    return (
        guard,
        penalties,
    )


def selection_score(
    val_result: Dict[str, Any],
    baseline_val: Dict[str, Any],
):

    (
        guard,
        penalties,
    ) = legacy_regression_guard(
        val_result[
            "task_mean"
        ],
        baseline_val[
            "task_mean"
        ],
    )

    score = (
        float(
            val_result[
                "balanced"
            ]
        )
        +
        LEGACY_REGRESSION_PENALTY
        *
        guard
    )

    return (
        score,
        {
            "balanced_task_ce":
            val_result[
                "balanced"
            ],

            "legacy_regression_guard":
            guard,

            "legacy_task_penalties":
            penalties,

            "score":
            score,
        },
    )


# =============================================================================
# SAVE ADAPTER
# =============================================================================

def save_adapter(
    model,
    processor,
    path: Path,
    state: Dict[str, Any],
) -> None:

    if path.exists():

        shutil.rmtree(
            path
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        str(path),
        safe_serialization=True,
    )

    processor.save_pretrained(
        str(path)
    )

    save_json(
        path
        /
        "training_state.json",

        state,
    )


# =============================================================================
# MERGE
# =============================================================================

def merge_best(
    best_dir: Path,
    processor,
    base_model_path: Path,
    final_model_dir: Path,
) -> None:

    print()
    print("=" * 110)
    print("MERGE BEST LoRA -> FINAL MODEL")
    print("=" * 110)

    if final_model_dir.exists():

        shutil.rmtree(
            final_model_dir
        )

    base = load_base_model(
        base_model_path,
        device=None,
    )

    peft_model = (
        PeftModel
        .from_pretrained(
            base,
            str(
                best_dir
            ),
            is_trainable=False,
        )
    )

    merged = (
        peft_model
        .merge_and_unload()
    )

    final_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.save_pretrained(
        str(
            final_model_dir
        ),
        safe_serialization=True,
    )

    processor.save_pretrained(
        str(
            final_model_dir
        )
    )

    print(
        "[OK] Final model:",
        final_model_dir,
    )

    del (
        merged,
        peft_model,
        base,
    )

    cleanup_cuda()


# =============================================================================
# TRAIN
# =============================================================================

def train(
    args: argparse.Namespace,
) -> None:

    train_path = (
        args.data_dir
        /
        "train.jsonl"
    )

    val_path = (
        args.data_dir
        /
        "val.jsonl"
    )

    test_path = (
        args.data_dir
        /
        "test.jsonl"
    )

    for path in (
        args.base_model,
        train_path,
        val_path,
        test_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required."
        )

    torch.cuda.set_device(
        args.gpu_id
    )

    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    if not torch.cuda.is_bf16_supported():

        raise RuntimeError(
            "BF16 required. "
            "Use RTX 3080 Ti/A5000."
        )

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    set_seed(
        args.seed
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    train_rows = read_jsonl(
        train_path
    )

    val_rows = read_jsonl(
        val_path
    )

    test_rows = read_jsonl(
        test_path
    )

    train_counts = count_tasks(
        train_rows
    )

    val_counts = count_tasks(
        val_rows
    )

    test_counts = count_tasks(
        test_rows
    )

    task_weights = (
        inverse_sqrt_task_weights(
            train_counts
        )
    )

    gpu_info = (
        torch.cuda
        .get_device_properties(
            args.gpu_id
        )
    )

    print()
    print("=" * 110)
    print("VLM_BASELINE -> REASONING_VLM_V2")
    print("=" * 110)

    print(
        "GPU             :",
        gpu_info.name,
    )

    print(
        "VRAM            :",
        f"{gpu_info.total_memory / 1024**3:.2f} GB",
    )

    print(
        "Precision       : BF16"
    )

    print(
        "Attention       :",
        ATTN_IMPLEMENTATION,
    )

    print(
        "Base model      :",
        args.base_model,
    )

    print(
        "Train tasks     :",
        train_counts,
    )

    print(
        "Val tasks       :",
        val_counts,
    )

    print(
        "Test tasks      :",
        test_counts,
    )

    print(
        "Task sampling   : NONE"
    )

    print(
        "Task weights    :",
        {
            key:
            round(
                value,
                4,
            )
            for key, value
            in task_weights.items()
        },
    )

    print(
        "Tasks           : reasoning + original 5 VLM_Baseline tasks"
    )

    print(
        "Trajectory      : direct VLM output, 10 x (x,y), 0.5 s, 5 s"
    )

    print(
        "Loss            : assistant-token CE x task weight"
    )

    print(
        "LoRA            :",
        (
            f"r={args.lora_r}, "
            f"alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}"
        ),
    )

    print(
        "LR              :",
        f"{args.lr:.2e}",
    )

    print(
        "Epochs          :",
        args.epochs,
    )

    print(
        "Batch           :",
        (
            f"1 x accum "
            f"{args.grad_accum} "
            f"= effective "
            f"{args.grad_accum}"
        ),
    )

    # -------------------------------------------------------------------------
    # Processor
    # -------------------------------------------------------------------------

    processor = load_processor(
        args.base_model
    )

    collator = QwenVLCollator(
        processor,
        args.max_seq_len,
    )

    # -------------------------------------------------------------------------
    # Loader
    # -------------------------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        JsonlDataset(
            train_rows
        ),

        batch_size=1,

        shuffle=True,

        generator=generator,

        num_workers=0,

        pin_memory=True,

        collate_fn=collator,
    )

    val_loader = DataLoader(
        JsonlDataset(
            val_rows
        ),

        batch_size=1,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        collate_fn=collator,
    )

    test_loader = DataLoader(
        JsonlDataset(
            test_rows
        ),

        batch_size=1,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        collate_fn=collator,
    )

    # =========================================================================
    # INITIAL VLM_BASELINE VALIDATION
    #
    # This explicitly measures the OLD tasks before continued training.
    # =========================================================================

    print()
    print("=" * 110)
    print("[1/3] INITIAL VLM_BASELINE VALIDATION")
    print("=" * 110)

    base_model = load_base_model(
        args.base_model,
        device,
    )

    base_model.eval()

    baseline_val = evaluate_task_losses(
        base_model,
        val_loader,
        device,
        phase="BASE-VAL",
        log_every=100,
    )

    print(
        "Baseline balanced CE:",
        baseline_val[
            "balanced"
        ],
    )

    print(
        "Baseline per-task CE:",
        baseline_val[
            "task_mean"
        ],
    )

    save_json(
        args.run_dir
        /
        "baseline_validation.json",

        baseline_val,
    )

    # =========================================================================
    # ADD LoRA
    # =========================================================================

    print()
    print("=" * 110)
    print("[2/3] ATTACH LoRA")
    print("=" * 110)

    model, targets = attach_lora(
        base_model,

        args.lora_r,

        args.lora_alpha,

        args.lora_dropout,
    )

    print(
        "LoRA targets:",
        targets,
    )

    print_trainable(
        model
    )

    model.train()

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,

        lr=args.lr,

        weight_decay=(
            args.weight_decay
        ),

        betas=(
            0.9,
            0.999,
        ),
    )

    optimizer_steps_per_epoch = math.ceil(
        len(
            train_loader
        )
        /
        args.grad_accum
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        *
        args.epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        *
        args.warmup_ratio
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,

            num_warmup_steps=(
                warmup_steps
            ),

            num_training_steps=(
                total_optimizer_steps
            ),
        )
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_dir = (
        args.run_dir
        /
        "best_adapter"
    )

    best_score = float(
        "inf"
    )

    best_epoch = 0

    bad_epochs = 0

    global_step = 0

    history = []

    # =========================================================================
    # ETA STATE
    # =========================================================================

    cumulative_train_seconds = 0.0
    cumulative_train_samples = 0

    # The initial VLM_Baseline validation is used as the first estimate
    # for each later validation pass.
    val_seconds_estimate = float(
        baseline_val.get(
            "seconds",
            0.0,
        )
    )

    test_seconds_estimate = (
        val_seconds_estimate
        *
        len(test_loader)
        /
        max(
            1,
            len(val_loader),
        )
    )

    print()
    print(
        "[ETA] Validation reference :",
        format_duration(
            val_seconds_estimate
        ),
    )

    print(
        "[ETA] Test reference       :",
        format_duration(
            test_seconds_estimate
        ),
    )

    print(
        "[ETA] Train ETA appears after the first logging interval "
        "and stabilizes as more samples are processed."
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("=" * 110)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 110)

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_raw_sum = 0.0

        epoch_weighted_sum = 0.0

        epoch_count = 0

        task_raw_sum = defaultdict(
            float
        )

        task_count = defaultdict(
            int
        )

        start = time.perf_counter()

        for micro_step, batch in enumerate(
            train_loader,
            1,
        ):

            (
                inputs,
                task,
                sample_id,
                supervised_tokens,
            ) = move_batch(
                batch,
                device,
            )

            task_weight = task_weights.get(
                task,
                1.0,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=DTYPE,
            ):

                output = model(
                    **inputs
                )

                raw_loss = (
                    output.loss
                )

                weighted_loss = (
                    raw_loss
                    *
                    task_weight
                )

                backward_loss = (
                    weighted_loss
                    /
                    args.grad_accum
                )

            backward_loss.backward()

            raw_value = float(
                raw_loss
                .detach()
                .float()
                .item()
            )

            weighted_value = (
                raw_value
                *
                task_weight
            )

            epoch_raw_sum += (
                raw_value
            )

            epoch_weighted_sum += (
                weighted_value
            )

            epoch_count += 1

            task_raw_sum[
                task
            ] += raw_value

            task_count[
                task
            ] += 1

            do_step = (
                micro_step
                %
                args.grad_accum
                ==
                0
            )

            last_step = (
                micro_step
                ==
                len(
                    train_loader
                )
            )

            if (
                do_step
                or last_step
            ):

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            if (
                micro_step
                %
                args.log_every
                ==
                0
                or last_step
            ):

                elapsed = (
                    time.perf_counter()
                    -
                    start
                )

                current_lr = (
                    scheduler
                    .get_last_lr()[0]
                )

                allocated = (
                    torch.cuda
                    .memory_allocated(
                        device
                    )
                    /
                    1024**3
                )

                completed_train_samples = (
                    cumulative_train_samples
                    +
                    micro_step
                )

                measured_train_seconds = (
                    cumulative_train_seconds
                    +
                    elapsed
                )

                avg_train_sec_per_sample = (
                    measured_train_seconds
                    /
                    max(
                        1,
                        completed_train_samples,
                    )
                )

                epoch_remaining_samples = max(
                    0,
                    len(train_loader)
                    -
                    micro_step,
                )

                future_epoch_samples = (
                    max(
                        0,
                        args.epochs
                        -
                        epoch,
                    )
                    *
                    len(train_loader)
                )

                total_remaining_train_samples = (
                    epoch_remaining_samples
                    +
                    future_epoch_samples
                )

                epoch_eta_seconds = (
                    avg_train_sec_per_sample
                    *
                    epoch_remaining_samples
                )

                # Approximate ETA through all requested epochs + validation
                # passes + final held-out test. Early stopping may finish sooner.
                remaining_val_runs = max(
                    0,
                    args.epochs
                    -
                    epoch
                    +
                    1,
                )

                run_eta_seconds = (
                    avg_train_sec_per_sample
                    *
                    total_remaining_train_samples
                    +
                    val_seconds_estimate
                    *
                    remaining_val_runs
                    +
                    test_seconds_estimate
                )

                print(
                    f"[TRAIN] "
                    f"epoch={epoch:02d} "
                    f"micro="
                    f"{micro_step:6d}/"
                    f"{len(train_loader):6d} "
                    f"raw="
                    f"{epoch_raw_sum / epoch_count:.5f} "
                    f"weighted="
                    f"{epoch_weighted_sum / epoch_count:.5f} "
                    f"task={task:14s} "
                    f"tokens={supervised_tokens:4d} "
                    f"lr={current_lr:.3e} "
                    f"gpu={allocated:.2f}GB "
                    f"elapsed={format_duration(elapsed)} "
                    f"epoch_eta={format_duration(epoch_eta_seconds)} "
                    f"run_eta~={format_duration(run_eta_seconds)}"
                )

            del (
                inputs,
                output,
                raw_loss,
                weighted_loss,
                backward_loss,
            )

        epoch_train_seconds = (
            time.perf_counter()
            -
            start
        )

        cumulative_train_seconds += (
            epoch_train_seconds
        )

        cumulative_train_samples += len(
            train_loader
        )

        print()
        print(
            f"[EPOCH {epoch:02d} TRAIN DONE] "
            f"time={format_duration(epoch_train_seconds)} "
            f"| avg="
            f"{epoch_train_seconds / max(1, len(train_loader)):.3f}s/sample"
        )

        # ---------------------------------------------------------------------
        # Epoch train summary
        # ---------------------------------------------------------------------

        train_task_mean = {}

        for task in TASKS:

            count = task_count.get(
                task,
                0,
            )

            if count:

                train_task_mean[
                    task
                ] = (
                    task_raw_sum[
                        task
                    ]
                    /
                    count
                )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        print()
        print(
            "[VAL] evaluating..."
        )

        val_result = evaluate_task_losses(
            model,
            val_loader,
            device,
            phase=f"VAL-E{epoch}",
            log_every=100,
        )

        # Update future validation ETA using a smoothed measured duration.
        val_seconds_estimate = (
            0.7
            *
            val_seconds_estimate
            +
            0.3
            *
            float(
                val_result.get(
                    "seconds",
                    val_seconds_estimate,
                )
            )
        )

        (
            score,
            score_info,
        ) = selection_score(
            val_result,
            baseline_val,
        )

        print()
        print(
            "Train per-task CE:",
            train_task_mean,
        )

        print(
            "Val per-task CE  :",
            val_result[
                "task_mean"
            ],
        )

        print(
            "Balanced val CE  :",
            f"{val_result['balanced']:.6f}",
        )

        print(
            "Legacy guard     :",
            f"{score_info['legacy_regression_guard']:.6f}",
        )

        print(
            "Selection score  :",
            f"{score:.6f}",
        )

        epoch_state = {
            "epoch":
            epoch,

            "global_step":
            global_step,

            "train_raw_mean":
            epoch_raw_sum
            /
            max(
                1,
                epoch_count,
            ),

            "train_weighted_mean":
            epoch_weighted_sum
            /
            max(
                1,
                epoch_count,
            ),

            "train_task_mean":
            train_task_mean,

            "val":
            val_result,

            "selection":
            score_info,

            "task_weights":
            task_weights,
        }

        history.append(
            epoch_state
        )

        save_json(
            args.run_dir
            /
            "history.json",

            history,
        )

        # ---------------------------------------------------------------------
        # Best checkpoint
        # ---------------------------------------------------------------------

        improved = (
            score
            <
            (
                best_score
                -
                args.min_delta
            )
        )

        if improved:

            best_score = score

            best_epoch = epoch

            bad_epochs = 0

            save_adapter(
                model,
                processor,
                best_dir,
                epoch_state,
            )

            print(
                f"[BEST] epoch={epoch} "
                f"score={score:.6f}"
            )

        else:

            bad_epochs += 1

            print(
                f"[NO IMPROVEMENT] "
                f"{bad_epochs}/"
                f"{args.patience}"
            )

        cleanup_cuda()

        if bad_epochs >= args.patience:

            print()
            print(
                "[EARLY STOP]"
            )

            break

    # =========================================================================
    # DELETE TRAINING MODEL
    # =========================================================================

    del (
        model,
        base_model,
        optimizer,
        scheduler,
    )

    cleanup_cuda()

    if best_epoch <= 0:

        raise RuntimeError(
            "No best checkpoint created."
        )

    # =========================================================================
    # TEST BEST
    # =========================================================================

    print()
    print("=" * 110)
    print("[3/3] HELD-OUT TEST")
    print("=" * 110)

    test_base = load_base_model(
        args.base_model,
        device,
    )

    best_model = (
        PeftModel
        .from_pretrained(
            test_base,

            str(
                best_dir
            ),

            is_trainable=False,
        )
    )

    best_model.to(
        device
    )

    best_model.eval()

    test_result = evaluate_task_losses(
        best_model,
        test_loader,
        device,
        phase="TEST",
        log_every=100,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Test balanced CE:",
        test_result[
            "balanced"
        ],
    )

    print(
        "Test per-task CE:",
        test_result[
            "task_mean"
        ],
    )

    save_json(
        args.run_dir
        /
        "test_result.json",

        {
            "best_epoch":
            best_epoch,

            "best_selection_score":
            best_score,

            "test":
            test_result,
        },
    )

    del (
        best_model,
        test_base,
    )

    cleanup_cuda()

    # =========================================================================
    # MERGE IS A SEPARATE STAGE IN THE SPLIT WORKFLOW
    # =========================================================================

    if not getattr(args, "skip_merge", False):
        merge_best(
            best_dir=best_dir,
            processor=processor,
            base_model_path=args.base_model,
            final_model_dir=args.final_model_dir,
        )
    else:
        print()
        print("[SKIP] merge disabled; run 03_merge_reasoning_v2.py separately.")

    print()
    print("=" * 110)
    print("TRAINING COMPLETE")
    print("=" * 110)

    print(
        "Base:",
        args.base_model,
    )

    print(
        "Best adapter:",
        best_dir,
    )

    print(
        "Final merged:",
        (args.final_model_dir if not getattr(args, "skip_merge", False) else "NOT MERGED YET"),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Tasks:",
        TASKS,
    )

    print(
        "Trajectory capability:",
        "PRESERVED + TRAINED"
    )

    print(
        "Reasoning capability:",
        "ADDED"
    )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        "--part1-root",
        type=Path,
        default=PART1_ROOT,
    )

    parser.add_argument(
        "--base-model",
        type=Path,
        default=BASE_MODEL_PATH,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
    )

    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=FINAL_MODEL_DIR,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
    )

    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Debug only.",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.prepare_only
        and
        args.train_only
    ):

        parser.error(
            "--prepare-only and "
            "--train-only cannot be used together."
        )

    # Resolve paths.
    args.part1_root = (
        args.part1_root
        .expanduser()
        .resolve()
    )

    args.base_model = (
        args.base_model
        .expanduser()
        .resolve()
    )

    args.data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    args.run_dir = (
        args.run_dir
        .expanduser()
        .resolve()
    )

    args.final_model_dir = (
        args.final_model_dir
        .expanduser()
        .resolve()
    )

    return args


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    print()
    print("=" * 110)
    print("VLM_BASELINE -> REASONING_VLM_V2")
    print("=" * 110)

    print(
        "Base model:",
        args.base_model,
    )

    print(
        "Dataset:",
        args.part1_root,
    )

    print(
        "Tasks:",
        ", ".join(
            TASKS
        ),
    )

    print(
        "Trajectory:",
        "10 x (x,y), 0.5s interval, 5s horizon, ego frame",
    )

    print(
        "New capability:",
        "natural-language reasoning trace",
    )

    print(
        "Input dynamics:",
        "speed + acceleration + heading/yaw",
    )

    print(
        "Split:",
        "70 / 20 / 10 clip-disjoint",
    )

    print(
        "LoRA:",
        (
            f"r={args.lora_r}, "
            f"alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}"
        ),
    )

    # -------------------------------------------------------------------------
    # PREPARE
    # -------------------------------------------------------------------------

    if not args.train_only:

        prepare_dataset(
            args
        )

    if args.prepare_only:

        print()
        print(
            "[DONE] prepare-only"
        )

        return

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    train(
        args
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()