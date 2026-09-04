#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning ↔ Decision ↔ VLM Trajectory ↔ Action Expert causal-diagnostic analysis
for Reasoning_VLM_v2_fixedsplit + RAW-KV Flow/DiT Action Experts.

GOAL
----
Diagnose WHY reasoning-conditioned Action Expert planning improves (or fails),
not merely whether mean ADE improves.

The same held-out fixed-split sample is evaluated through four synchronized views:

  1) VLM reasoning prompt
       -> generated reasoning
       -> reasoning quality vs GT reasoning

  2) VLM decision prompt
       -> explicit Longitudinal / Lateral driving decision
       -> decision accuracy vs GT decision

  3) VLM trajectory prompt
       -> directly generated 10 x (x,y) trajectory
       -> direct VLM trajectory ADE/FDE vs GT

  4) Existing RAW-KV Flow/DiT Action Experts
       direct AE    : reasoning-prompt boundary RAW K/V
       reasoning AE : same prompt RAW K/V + generated reasoning RAW K/V
       -> 10 x (x,y,yaw)

The script then tests the following relationships:

A. Reasoning quality -> direct VLM trajectory quality
B. Direct VLM trajectory quality -> direct/reasoning AE quality
C. Reasoning quality -> reasoning-AE gain over direct AE
D. Explicit VLM decision direction -> VLM/direct-AE/reasoning-AE trajectory direction
E. Generated-reasoning intent -> explicit VLM decision -> trajectory direction
F. Partial correlations controlling for VLM trajectory quality / reasoning length
G. Standardized multivariable regression for reasoning-AE gain
H. 2x2 stratification: reasoning quality x VLM trajectory quality
I. Counterexamples that separate semantic-quality effects from generic VLM/latent quality

IMPORTANT CALIBRATION
---------------------
Trajectory-to-decision direction is necessarily a heuristic because a continuous
5-second path is richer than a discrete decision label. Therefore the SAME
trajectory-direction extractor is first calibrated on:

    GT decision vs GT trajectory

If that calibration is weak, the report explicitly warns against over-interpreting
prediction-side direction agreement.

FAIRNESS / CONTROL
------------------
- exact same ActionExpert fixed-split test rows
- exact same images / mission / speed / acceleration / heading per VLM task
- exact training prompts for reasoning / decision / trajectory
- greedy VLM generation (do_sample=False)
- same Reasoning_VLM_v2_fixedsplit backbone for all VLM outputs
- direct/reasoning Action Experts use identical per-sample Gaussian x0
- Action Expert checkpoint normalizers/configs are hard-checked for compatibility
- existing trajectory/reasoning RAW-KV ablation cache is reused when compatible,
  so this diagnostic can analyze the EXACT generated reasoning/trajectory used in
  the previous controlled experiment

DEFAULT PATHS
-------------
VLM:
  /home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit

Dataset:
  /home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl

Existing controlled trace cache (optional but preferred):
  /home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit

Action Experts:
  /home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/direct_flow_dit/best.pt
  /home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt

Output:
  /home/lhh/lab/Action_Expert/results/reasoning_trajectory_causality_fixedsplit

OUTPUT FILES
------------
report.md
summary.json
samples.jsonl
samples.csv
counterexamples.json
attention_diagnostics.json  (overall/layer/Euler-step metrics, collapse, gain correlations)
attention_vectors.pt        (sample x layer x Euler-step attended V-context vectors)
plots/*.png                 (if matplotlib is installed)

Example
-------
cd ~/lab/Action_Expert/scripts
python3 analyze_reasoning_trajectory_causality_fixedsplit.py --overwrite

Debug first 10:
python3 analyze_reasoning_trajectory_causality_fixedsplit.py \
    --limit 10 --overwrite

Force fresh reasoning/trajectory RAW-KV + text generation instead of reusing the
previous trace cache:
python3 analyze_reasoning_trajectory_causality_fixedsplit.py \
    --force-regenerate-trace-cache --overwrite
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# DEFAULTS
# =============================================================================

VLM_PATH = Path("/home/lhh/lab/models/vlm/Reasoning_VLM_v2_fixedsplit")
TEST_JSONL = Path("/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl")

PREFERRED_TRACE_CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/trace_rawkv_ablation_fixedsplit"
)
WORK_CACHE_ROOT = Path(
    "/home/lhh/lab/Dataset/ActionExpert/reasoning_trajectory_causality_fixedsplit"
)

DIRECT_CKPT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/direct_flow_dit/best.pt"
)
REASONING_CKPT = Path(
    "/home/lhh/lab/models/action_expert/"
    "reasoning_vlm_v2_fixedsplit_rawkv_dit/flow/reasoning_flow_dit/best.pt"
)

RESULT_ROOT = Path(
    "/home/lhh/lab/Action_Expert/results/reasoning_trajectory_causality_fixedsplit"
)

NUM_STEPS = 10
DT = 0.5
SEED = 20260823
FLOW_NOISE_SEED = 20260841
MIN_PIXELS = 200_704
MAX_PIXELS = 200_704
CACHE_DTYPE = torch.bfloat16

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

TASKS = ("reasoning", "trajectory", "decision")


# =============================================================================
# EXACT VLM PROMPTS FROM Reasoning_VLM_v2 TRAINING
# =============================================================================

def build_context(rec: Dict[str, Any]) -> str:
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


def build_task_prompt(rec: Dict[str, Any], task: str) -> str:
    if task == "reasoning":
        return (
            build_context(rec)
            + "Generate the driving reasoning for the current situation. "
            "Use the three camera views, mission command, current speed, "
            "signed longitudinal acceleration, and heading/yaw when relevant. "
            "Explain the important scene evidence, safety constraints, route "
            "intention, and ego-motion context that justify the appropriate "
            "driving behavior. "
            "Do not output a separate final action label, JSON, or trajectory. "
            "Return only the natural-language reasoning trace."
        )

    if task == "trajectory":
        return (
            build_context(rec)
            + "Predict the vehicle's trajectory for the next 5 seconds "
            "as 10 waypoints at 0.5 s intervals, "
            "in the ego frame "
            "(x forward, y left, meters).\n"
            "Answer with exactly 10 comma-separated pairs like: "
            "(x1,y1), (x2,y2), ... "
            "with one decimal place."
        )

    if task == "decision":
        return (
            build_context(rec)
            + "Decide the driving action for this moment.\n"
            + "Longitudinal must be exactly one of: "
            + ", ".join(LON_CLASSES)
            + ".\n"
            + "Lateral must be exactly one of: "
            + ", ".join(LAT_CLASSES)
            + ".\n"
            + 'Answer with JSON only: {"Longitudinal": "...", "Lateral": "..."}'
        )

    raise ValueError(f"Unknown task: {task}")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"JSONL parse error {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_dir_signature(path: Path) -> str:
    patterns = ("*.safetensors", "*.bin", "*.json", "*.model", "*.txt")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise RuntimeError(f"No model/config files found under {path}")
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(str(p.relative_to(path)).encode("utf-8"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(st.st_mtime_ns).encode("ascii"))
    return h.hexdigest()


def stable_sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((value + int(base_seed)) % (2**63 - 1))


def make_noise(sample_id: str, base_seed: int, num_steps: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_sample_seed(sample_id, base_seed))
    return torch.randn((1, num_steps, 3), generator=g, dtype=torch.float32)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    sid = str(row.get("id", "")).strip()
    if not sid:
        raise ValueError("Empty sample id")
    row["id"] = sid

    command = str(row.get("command") or row.get("mission_command") or "").strip()
    if not command:
        raise ValueError(f"Missing command id={sid}")
    row["command"] = command
    row["mission_command"] = command

    images = row.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"Expected exactly 3 images id={sid}")
    images = [str(x) for x in images]
    for p in images:
        if not Path(p).is_file():
            raise FileNotFoundError(f"id={sid}: {p}")
    row["images"] = images

    for key in ("speed_mps", "acceleration_mps2", "heading_rad"):
        value = finite_float(row.get(key))
        if not math.isfinite(value):
            raise ValueError(f"Missing/non-finite {key} id={sid}")
        row[key] = value

    gt = np.asarray(row.get("trajectory"), dtype=np.float32)
    if gt.shape != (NUM_STEPS, 3) or not np.isfinite(gt).all():
        raise ValueError(f"trajectory must be finite [10,3] id={sid}, got={gt.shape}")
    row["trajectory"] = gt.tolist()

    # These fields are part of prepare_action_expert_fixedsplit.py.
    row["gt_reasoning"] = str(row.get("gt_reasoning") or "").strip()
    row["longitudinal"] = normalize_lon_label(str(row.get("longitudinal") or "").strip())
    row["lateral"] = normalize_lat_label(str(row.get("lateral") or "").strip())
    row["scene_description"] = str(row.get("scene_description") or "").strip()
    return row


def row_fingerprint(row: Dict[str, Any]) -> str:
    payload = {
        "id": row["id"],
        "images": row["images"],
        "command": row["command"],
        "speed_mps": float(row["speed_mps"]),
        "acceleration_mps2": float(row["acceleration_mps2"]),
        "heading_rad": float(row["heading_rad"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


# =============================================================================
# LABEL NORMALIZATION / PARSING
# =============================================================================

def _canonical_match(value: str, classes: Sequence[str]) -> str:
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    for cls in classes:
        if text == cls.lower():
            return cls
    for cls in classes:
        if cls.lower() in text:
            return cls
    return ""


def normalize_lon_label(value: str) -> str:
    replacements = {
        "gently accelerate (speed up)": "Gently accelerate",
        "maintain current speed": "Maintain speed",
    }
    text = re.sub(r"\s+", " ", str(value).strip())
    text = replacements.get(text.lower(), text)
    return _canonical_match(text, LON_CLASSES)


def normalize_lat_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return _canonical_match(text, LAT_CLASSES)


def parse_decision_text(text: str) -> Dict[str, Any]:
    raw = str(text).strip()
    obj: Any = None

    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                obj = None

    lon = ""
    lat = ""
    if isinstance(obj, dict):
        lon = normalize_lon_label(
            str(obj.get("Longitudinal") or obj.get("longitudinal") or "")
        )
        lat = normalize_lat_label(
            str(obj.get("Lateral") or obj.get("lateral") or "")
        )

    if not lon:
        lon = normalize_lon_label(raw)
    if not lat:
        lat = normalize_lat_label(raw)

    return {
        "raw": raw,
        "longitudinal": lon,
        "lateral": lat,
        "parse_ok": bool(lon and lat),
    }


def reasoning_intent(text: str) -> Dict[str, str]:
    """Heuristic intent extraction from natural-language reasoning.

    This is auxiliary only. The explicit VLM decision prompt is the primary
    semantic action probe.
    """
    s = re.sub(r"\s+", " ", str(text).lower())

    # Longitudinal: specific high-priority intents first.
    if re.search(r"\b(remain stopped|stay stopped|remain stationary)\b", s):
        lon = "Remain stopped"
    elif re.search(r"\b(quickly come to a stop|hard stop|emergency stop)\b", s):
        lon = "Quickly come to a stop"
    elif re.search(r"\b(gently come to a stop|come to a complete stop|come to a stop|stop behind|stop before|stop at|yield .* stop)\b", s):
        lon = "Gently come to a stop"
    elif re.search(r"\b(slow down quickly|decelerate quickly|brake hard)\b", s):
        lon = "Slow down quickly"
    elif re.search(r"\b(slow down gently|slow down|gently decelerate|decelerate)\b", s):
        lon = "Slow down gently"
    elif re.search(r"\b(gently accelerate|accelerate|speed up)\b", s):
        lon = "Gently accelerate"
    elif re.search(r"\b(maintain speed|maintain current speed|keep current speed|continue at the current speed)\b", s):
        lon = "Maintain speed"
    else:
        lon = ""

    # Lateral: turn/lane-change > slight shift > no action.
    if re.search(r"\b(turn left|left turn)\b", s):
        lat = "Turn left"
    elif re.search(r"\b(turn right|right turn)\b", s):
        lat = "Turn right"
    elif re.search(r"\b(left lane change|change to the left lane|change lanes? to the left|move into the left lane)\b", s):
        lat = "Left lane change"
    elif re.search(r"\b(right lane change|change to the right lane|change lanes? to the right|move into the right lane)\b", s):
        lat = "Right lane change"
    elif re.search(r"\b(slightly move left|move slightly left|shift slightly left|keep left)\b", s):
        lat = "Slightly move left in the lane"
    elif re.search(r"\b(slightly move right|move slightly right|shift slightly right|keep right)\b", s):
        lat = "Slightly move right in the lane"
    elif re.search(r"\b(stay centered|stay in (?:the )?current lane|maintain (?:the )?current lane|no lateral action|keep the lane|continue straight)\b", s):
        lat = "No lateral action"
    else:
        lat = ""

    return {"longitudinal": lon, "lateral": lat}


def longitudinal_coarse(label: str) -> str:
    label = normalize_lon_label(label)
    if label == "Gently accelerate":
        return "accelerate"
    if label == "Maintain speed":
        return "maintain"
    if label in ("Slow down gently", "Slow down quickly"):
        return "decelerate"
    if label in ("Gently come to a stop", "Quickly come to a stop", "Remain stopped"):
        return "stop"
    return "unknown"


def longitudinal_sign(label: str) -> str:
    coarse = longitudinal_coarse(label)
    if coarse == "accelerate":
        return "positive"
    if coarse in ("decelerate", "stop"):
        return "negative"
    if coarse == "maintain":
        return "neutral"
    return "unknown"


def lateral_coarse(label: str) -> str:
    label = normalize_lat_label(label)
    if label in ("Slightly move left in the lane", "Left lane change", "Turn left"):
        return "left"
    if label in ("Slightly move right in the lane", "Right lane change", "Turn right"):
        return "right"
    if label == "No lateral action":
        return "straight"
    return "unknown"


# =============================================================================
# TEXT QUALITY
# =============================================================================

def text_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower())


def token_f1(pred: str, gt: str) -> float:
    p = text_tokens(pred)
    g = text_tokens(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc_ = Counter(g)
    overlap = sum((pc & gc_).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2.0 * precision * recall / (precision + recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, gt: str) -> float:
    p = text_tokens(pred)
    g = text_tokens(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    lcs = lcs_length(p, g)
    precision = lcs / len(p)
    recall = lcs / len(g)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# =============================================================================
# VLM TRAJECTORY PARSING + METRICS
# =============================================================================

PAIR_RE = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)"
)


def parse_vlm_trajectory(text: str) -> Optional[np.ndarray]:
    pairs = [(float(a), float(b)) for a, b in PAIR_RE.findall(str(text))]
    if len(pairs) >= NUM_STEPS:
        arr = np.asarray(pairs[:NUM_STEPS], dtype=np.float64)
        if arr.shape == (NUM_STEPS, 2) and np.isfinite(arr).all():
            return arr

    # Fallback for formats without parentheses; only accept if exactly 20 usable numbers.
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(text))
    try:
        vals = [float(x) for x in nums]
    except Exception:
        return None
    if len(vals) == 20:
        arr = np.asarray(vals, dtype=np.float64).reshape(NUM_STEPS, 2)
        if np.isfinite(arr).all():
            return arr
    return None


def xy_metrics(pred_xy: np.ndarray, gt_xyz: np.ndarray) -> Dict[str, float]:
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    if pred_xy.shape != (NUM_STEPS, 2) or gt_xyz.shape != (NUM_STEPS, 3):
        raise ValueError(f"Bad xy metric shapes {pred_xy.shape}, {gt_xyz.shape}")
    err = np.linalg.norm(pred_xy - gt_xyz[:, :2], axis=-1)
    return {"ade_m": float(err.mean()), "fde_m": float(err[-1])}


def wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def xyz_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_xyz, dtype=np.float64)
    gt = np.asarray(gt_xyz, dtype=np.float64)
    if pred.shape != (NUM_STEPS, 3) or gt.shape != (NUM_STEPS, 3):
        raise ValueError(f"Bad xyz metric shapes {pred.shape}, {gt.shape}")
    err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
    heading = np.abs(wrap_angle_np(pred[:, 2] - gt[:, 2]))
    return {
        "ade_m": float(err.mean()),
        "fde_m": float(err[-1]),
        "heading_mae_rad": float(heading.mean()),
    }


def trajectory_pair_distance(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape[-1] == 3:
        aa = aa[:, :2]
    if bb.shape[-1] == 3:
        bb = bb[:, :2]
    if aa.shape != (NUM_STEPS, 2) or bb.shape != (NUM_STEPS, 2):
        return {"mean_xy_m": float("nan"), "final_xy_m": float("nan")}
    d = np.linalg.norm(aa - bb, axis=-1)
    return {"mean_xy_m": float(d.mean()), "final_xy_m": float(d[-1])}


# =============================================================================
# TRAJECTORY -> MOTION/DIRECTION DIAGNOSTIC
# =============================================================================

def inferred_xy_heading(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    # Robust final direction from the last up-to-3 segments.
    pts = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    diffs = np.diff(pts, axis=0)
    tail = diffs[-3:]
    vec = tail.mean(axis=0)
    if np.linalg.norm(vec) < 1.0e-6:
        return 0.0
    return float(math.atan2(float(vec[1]), float(vec[0])))


def trajectory_motion(
    trajectory: np.ndarray,
    current_speed_mps: float,
) -> Dict[str, Any]:
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.shape not in ((NUM_STEPS, 2), (NUM_STEPS, 3)):
        raise ValueError(f"trajectory_motion expects [10,2/3], got={arr.shape}")
    xy = arr[:, :2]
    pts = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    seg_speeds = np.linalg.norm(np.diff(pts, axis=0), axis=-1) / DT
    early_speed = float(seg_speeds[:3].mean())
    late_speed = float(seg_speeds[-3:].mean())
    current = float(max(0.0, current_speed_mps))
    final_progress = float(np.linalg.norm(xy[-1]))

    accel_threshold = max(0.8, 0.12 * max(current, 1.0))
    decel_threshold = accel_threshold

    if current < 0.5 and final_progress < 2.0 and late_speed < 0.8:
        lon = "stop"
    elif current >= 0.8 and late_speed < 0.6:
        lon = "stop"
    elif late_speed - current > accel_threshold:
        lon = "accelerate"
    elif late_speed - current < -decel_threshold:
        lon = "decelerate"
    else:
        lon = "maintain"

    if arr.shape[1] == 3:
        final_heading = float(arr[-1, 2])
    else:
        final_heading = inferred_xy_heading(xy)

    final_y = float(xy[-1, 1])
    lateral_threshold_m = 0.75
    heading_threshold_rad = 0.10

    left_signal = final_y > lateral_threshold_m or final_heading > heading_threshold_rad
    right_signal = final_y < -lateral_threshold_m or final_heading < -heading_threshold_rad
    if left_signal and not right_signal:
        lat = "left"
    elif right_signal and not left_signal:
        lat = "right"
    else:
        lat = "straight"

    return {
        "early_speed_est_mps": early_speed,
        "late_speed_est_mps": late_speed,
        "current_speed_mps": current,
        "delta_late_vs_current_mps": late_speed - current,
        "final_x_m": float(xy[-1, 0]),
        "final_y_m": final_y,
        "final_heading_rad": final_heading,
        "longitudinal": lon,
        "lateral": lat,
    }


def decision_vs_motion(
    longitudinal_label: str,
    lateral_label: str,
    motion: Dict[str, Any],
) -> Dict[str, Any]:
    dec_lon4 = longitudinal_coarse(longitudinal_label)
    dec_lon_sign = longitudinal_sign(longitudinal_label)
    dec_lat = lateral_coarse(lateral_label)

    mot_lon4 = str(motion.get("longitudinal", "unknown"))
    mot_lon_sign = (
        "positive" if mot_lon4 == "accelerate"
        else "negative" if mot_lon4 in ("decelerate", "stop")
        else "neutral" if mot_lon4 == "maintain"
        else "unknown"
    )
    mot_lat = str(motion.get("lateral", "unknown"))

    lon4_eval = dec_lon4 != "unknown" and mot_lon4 != "unknown"
    lonsign_eval = dec_lon_sign != "unknown" and mot_lon_sign != "unknown"
    lat_eval = dec_lat != "unknown" and mot_lat != "unknown"

    lon4_match = bool(lon4_eval and dec_lon4 == mot_lon4)
    lonsign_match = bool(lonsign_eval and dec_lon_sign == mot_lon_sign)
    lat_match = bool(lat_eval and dec_lat == mot_lat)

    signed_axes: List[bool] = []
    if lonsign_eval:
        signed_axes.append(lonsign_match)
    if lat_eval:
        signed_axes.append(lat_match)

    exact_axes: List[bool] = []
    if lon4_eval:
        exact_axes.append(lon4_match)
    if lat_eval:
        exact_axes.append(lat_match)

    return {
        "decision_longitudinal_coarse": dec_lon4,
        "trajectory_longitudinal_coarse": mot_lon4,
        "decision_longitudinal_sign": dec_lon_sign,
        "trajectory_longitudinal_sign": mot_lon_sign,
        "decision_lateral": dec_lat,
        "trajectory_lateral": mot_lat,
        "longitudinal_exact_match": lon4_match if lon4_eval else None,
        "longitudinal_sign_match": lonsign_match if lonsign_eval else None,
        "lateral_match": lat_match if lat_eval else None,
        "signed_score": float(np.mean(signed_axes)) if signed_axes else None,
        "exact_score": float(np.mean(exact_axes)) if exact_axes else None,
        "signed_joint_match": bool(all(signed_axes)) if len(signed_axes) == 2 else None,
        "exact_joint_match": bool(all(exact_axes)) if len(exact_axes) == 2 else None,
    }


def decision_pair_consistency(a_lon: str, a_lat: str, b_lon: str, b_lat: str) -> Dict[str, Any]:
    a_lon_sign = longitudinal_sign(a_lon)
    b_lon_sign = longitudinal_sign(b_lon)
    a_lat_c = lateral_coarse(a_lat)
    b_lat_c = lateral_coarse(b_lat)
    lon_eval = a_lon_sign != "unknown" and b_lon_sign != "unknown"
    lat_eval = a_lat_c != "unknown" and b_lat_c != "unknown"
    lon_match = bool(lon_eval and a_lon_sign == b_lon_sign)
    lat_match = bool(lat_eval and a_lat_c == b_lat_c)
    vals = []
    if lon_eval:
        vals.append(lon_match)
    if lat_eval:
        vals.append(lat_match)
    return {
        "longitudinal_sign_match": lon_match if lon_eval else None,
        "lateral_match": lat_match if lat_eval else None,
        "score": float(np.mean(vals)) if vals else None,
        "joint_match": bool(all(vals)) if len(vals) == 2 else None,
    }


# =============================================================================
# VLM RUNTIME + RAW-KV EXTRACTION
# =============================================================================

def load_target_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
):
    kwargs = {
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), torch_dtype=dtype, **kwargs
        )
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return model, processor


def reset_multimodal_rope_state(model) -> None:
    candidates = [model, getattr(model, "model", None)]
    inner = getattr(getattr(model, "model", None), "language_model", None)
    if inner is not None:
        candidates.append(inner)
    for obj in candidates:
        if obj is not None and hasattr(obj, "rope_deltas"):
            obj.rope_deltas = None


def open_three_images(row: Dict[str, Any]) -> List[Image.Image]:
    result: List[Image.Image] = []
    for p in row["images"]:
        with Image.open(p) as im:
            result.append(im.convert("RGB").copy())
    return result


def build_batch(
    processor,
    row: Dict[str, Any],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], str]:
    prompt = build_task_prompt(row, task)
    images = open_three_images(row)
    user_content = [
        {"type": "text", "text": "Front-left camera:"},
        {"type": "image"},
        {"type": "text", "text": "Front camera:"},
        {"type": "image"},
        {"type": "text", "text": "Front-right camera:"},
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": user_content}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    batch = dict(batch)
    batch.pop("token_type_ids", None)
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved, prompt_text


def normalize_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(x) for x in value if x is not None}
    return {int(value)}


def content_token_ids(processor, model, new_ids: torch.Tensor) -> Tuple[List[int], bool]:
    eos_ids: set[int] = set()
    eos_ids |= normalize_token_id_set(getattr(processor.tokenizer, "eos_token_id", None))
    eos_ids |= normalize_token_id_set(
        getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    )
    for token_text in ("<|im_end|>", "<|endoftext|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token_text)
        if (
            token_id is not None
            and token_id != getattr(processor.tokenizer, "unk_token_id", None)
            and int(token_id) >= 0
        ):
            eos_ids.add(int(token_id))

    pad_ids: set[int] = set()
    pad_ids |= normalize_token_id_set(getattr(processor.tokenizer, "pad_token_id", None))
    pad_ids |= normalize_token_id_set(
        getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    )

    content: List[int] = []
    ended = False
    for token in new_ids.tolist():
        token = int(token)
        if token in eos_ids:
            ended = True
            break
        if token in pad_ids and content:
            ended = True
            break
        content.append(token)
    return content, ended


def get_last_layer_kv(past_key_values):
    if past_key_values is None:
        raise RuntimeError("past_key_values is None")
    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        if layers:
            layer = layers[-1]
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None and value is not None:
                return key, value
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if past_key_values.key_cache and past_key_values.value_cache:
            return past_key_values.key_cache[-1], past_key_values.value_cache[-1]
    if isinstance(past_key_values, (tuple, list)):
        last = past_key_values[-1]
        if isinstance(last, (tuple, list)) and len(last) >= 2:
            return last[0], last[1]
    raise RuntimeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def raw_kv_to_cpu(key: torch.Tensor, value: torch.Tensor, max_length: Optional[int] = None):
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
        raise ValueError(f"Expected equal K/V [1,H,T,D], got={key.shape}/{value.shape}")
    if max_length is not None:
        key = key[..., :max_length, :]
        value = value[..., :max_length, :]
    return (
        key[0].detach().to(dtype=CACHE_DTYPE).cpu().contiguous(),
        value[0].detach().to(dtype=CACHE_DTYPE).cpu().contiguous(),
    )


@torch.inference_mode()
def generate_text_only(
    model,
    processor,
    row: Dict[str, Any],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> Dict[str, Any]:
    reset_multimodal_rope_state(model)
    batch, prompt_text = build_batch(processor, row, task, device, dtype)
    prompt_len = int(batch["input_ids"].shape[1])
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        kwargs["pad_token_id"] = int(pad_id)

    sync_cuda(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        out = model.generate(**batch, **kwargs)
    sync_cuda(device)
    sec = time.perf_counter() - t0
    seq = out.sequences[0, prompt_len:]
    ids, ended = content_token_ids(processor, model, seq)
    text = processor.tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return {
        "text": text,
        "token_ids": [int(x) for x in ids],
        "tokens": len(ids),
        "ended": bool(ended),
        "generation_ms": float(sec * 1000.0),
        "prompt_text": prompt_text,
    }


@torch.inference_mode()
def extract_task_cache(
    model,
    processor,
    row: Dict[str, Any],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    allow_truncated: bool,
    max_prefix_diff: float,
) -> Dict[str, Any]:
    reset_multimodal_rope_state(model)
    batch, prompt_text = build_batch(processor, row, task, device, dtype)
    prompt_token_length = int(batch["input_ids"].shape[1])

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        prompt_out = model(**batch, use_cache=True, return_dict=True)
    prompt_key, prompt_value = get_last_layer_kv(prompt_out.past_key_values)
    prompt_cache_length = int(prompt_key.shape[-2])
    direct_key, direct_value = raw_kv_to_cpu(
        prompt_key, prompt_value, max_length=prompt_cache_length
    )
    text_cfg = getattr(model.config, "text_config", model.config)
    hq = int(getattr(text_cfg, "num_attention_heads", direct_key.shape[0]))
    hkv = int(direct_key.shape[0])
    head_dim = int(direct_key.shape[-1])
    del prompt_out, prompt_key, prompt_value

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_id)

    sync_cuda(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        generation = model.generate(**batch, **generation_kwargs)
    sync_cuda(device)
    generation_ms = (time.perf_counter() - t0) * 1000.0

    sequences = getattr(generation, "sequences", None)
    generation_cache = getattr(generation, "past_key_values", None)
    if sequences is None or generation_cache is None:
        raise RuntimeError("generate() did not return sequences/past_key_values")

    new_ids = sequences[0, prompt_token_length:]
    content_ids, ended = content_token_ids(processor, model, new_ids)
    if not content_ids:
        raise RuntimeError(f"Empty generated {task}: id={row['id']}")
    if not ended and not allow_truncated:
        raise RuntimeError(
            f"{task} did not terminate within {max_new_tokens} tokens id={row['id']}"
        )
    text = processor.tokenizer.decode(
        content_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()

    gen_key, gen_value = get_last_layer_kv(generation_cache)
    generation_cache_length = int(gen_key.shape[-2])
    desired_end = prompt_cache_length + len(content_ids)
    usable_end = min(generation_cache_length, desired_end)
    missing = max(0, desired_end - generation_cache_length)
    if ended and missing > 0:
        raise RuntimeError(f"Generation cache missing {task} tokens id={row['id']} missing={missing}")
    if missing > 1:
        raise RuntimeError(f"Generation cache too short id={row['id']} missing={missing}")

    prefix_len = min(prompt_cache_length, generation_cache_length)
    gen_prefix_key = gen_key[0, :, :prefix_len, :].detach().float().cpu()
    gen_prefix_value = gen_value[0, :, :prefix_len, :].detach().float().cpu()
    key_diff = (gen_prefix_key - direct_key[:, :prefix_len, :].float()).abs().max().item()
    val_diff = (gen_prefix_value - direct_value[:, :prefix_len, :].float()).abs().max().item()
    prefix_diff = float(max(key_diff, val_diff))
    if prefix_diff > max_prefix_diff:
        raise RuntimeError(
            f"Prefill/generate prefix mismatch id={row['id']} task={task} "
            f"diff={prefix_diff:.6g} > {max_prefix_diff:.6g}"
        )

    delta_key = (
        gen_key[0, :, prompt_cache_length:usable_end, :]
        .detach().to(dtype=CACHE_DTYPE).cpu().contiguous()
    )
    delta_value = (
        gen_value[0, :, prompt_cache_length:usable_end, :]
        .detach().to(dtype=CACHE_DTYPE).cpu().contiguous()
    )
    if delta_key.shape != delta_value.shape or delta_key.ndim != 3 or delta_key.shape[1] <= 0:
        raise RuntimeError(f"Bad delta K/V id={row['id']} task={task}")

    return {
        "task": task,
        "prompt_text": prompt_text,
        "direct_key": direct_key,
        "direct_value": direct_value,
        "delta_key": delta_key,
        "delta_value": delta_value,
        "generated_text": text,
        "generated_token_ids": [int(x) for x in content_ids],
        "prompt_cache_length": prompt_cache_length,
        "generated_tokens": len(content_ids),
        "cached_generated_tokens": int(delta_key.shape[1]),
        "generation_ended": bool(ended),
        "generation_ms": float(generation_ms),
        "prefix_max_abs_diff": prefix_diff,
        "vlm_num_attention_heads": hq,
        "vlm_num_kv_heads": hkv,
        "vlm_head_dim": head_dim,
    }


# =============================================================================
# TRACE CACHE: reuse prior 2x2 cache when possible, otherwise rebuild
# =============================================================================

def cache_filename(index: int, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}_{digest}.pt"


def load_trace_manifest(root: Path) -> Optional[List[Dict[str, Any]]]:
    path = root / "test" / "manifest.jsonl"
    if not path.is_file():
        return None
    try:
        rows = read_jsonl(path)
    except Exception:
        return None
    return rows or None


def validate_trace_cache(
    root: Path,
    rows: Sequence[Dict[str, Any]],
    vlm_path: Path,
) -> Optional[List[Dict[str, Any]]]:
    manifest = load_trace_manifest(root)
    if manifest is None or len(manifest) < len(rows):
        return None
    manifest = manifest[:len(rows)]
    if [str(x.get("id")) for x in manifest] != [r["id"] for r in rows]:
        return None

    for item, row in zip(manifest, rows):
        p = Path(str(item.get("cache_file", "")))
        if not p.is_file():
            return None
        if str(item.get("input_fingerprint", "")) != row_fingerprint(row):
            return None

    meta_path = root / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved_vlm = Path(str(meta.get("vlm", ""))).expanduser()
            if saved_vlm and saved_vlm.resolve() != vlm_path.resolve():
                return None
        except Exception:
            return None
    return manifest


def build_trace_cache(
    root: Path,
    rows: Sequence[Dict[str, Any]],
    model,
    processor,
    vlm_path: Path,
    dataset_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if root.exists():
        shutil.rmtree(root)
    out_dir = root / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for index, row in enumerate(rows, 1):
        reasoning = extract_task_cache(
            model, processor, row, "reasoning", device, dtype,
            args.reasoning_max_new_tokens, args.allow_truncated, args.max_prefix_diff,
        )
        trajectory = extract_task_cache(
            model, processor, row, "trajectory", device, dtype,
            args.trajectory_max_new_tokens, args.allow_truncated, args.max_prefix_diff,
        )

        geom_r = (
            reasoning["vlm_num_attention_heads"],
            reasoning["vlm_num_kv_heads"],
            reasoning["vlm_head_dim"],
        )
        geom_t = (
            trajectory["vlm_num_attention_heads"],
            trajectory["vlm_num_kv_heads"],
            trajectory["vlm_head_dim"],
        )
        if geom_r != geom_t:
            raise RuntimeError(f"Reasoning/trajectory KV geometry mismatch id={row['id']}")

        cache_path = out_dir / cache_filename(index, row["id"])
        payload = {
            "cache_version": "reasoning_trajectory_causality_fixedsplit_v1",
            "id": row["id"],
            "clip": str(row.get("clip", "")),
            "input_fingerprint": row_fingerprint(row),
            "images": list(row["images"]),
            "command": row["command"],
            "speed_mps": float(row["speed_mps"]),
            "acceleration_mps2": float(row["acceleration_mps2"]),
            "heading_rad": float(row["heading_rad"]),
            "trajectory_gt": torch.as_tensor(row["trajectory"], dtype=torch.float32),
            "vlm": str(vlm_path),
            "vlm_num_attention_heads": geom_r[0],
            "vlm_num_kv_heads": geom_r[1],
            "vlm_head_dim": geom_r[2],
            "reasoning_direct_key": reasoning["direct_key"],
            "reasoning_direct_value": reasoning["direct_value"],
            "reasoning_delta_key": reasoning["delta_key"],
            "reasoning_delta_value": reasoning["delta_value"],
            "reasoning_text": reasoning["generated_text"],
            "reasoning_token_ids": reasoning["generated_token_ids"],
            "reasoning_prompt_cache_length": reasoning["prompt_cache_length"],
            "reasoning_cached_tokens": reasoning["cached_generated_tokens"],
            "reasoning_generation_ms": reasoning["generation_ms"],
            "trajectory_direct_key": trajectory["direct_key"],
            "trajectory_direct_value": trajectory["direct_value"],
            "trajectory_delta_key": trajectory["delta_key"],
            "trajectory_delta_value": trajectory["delta_value"],
            "trajectory_text": trajectory["generated_text"],
            "trajectory_token_ids": trajectory["generated_token_ids"],
            "trajectory_prompt_cache_length": trajectory["prompt_cache_length"],
            "trajectory_cached_tokens": trajectory["cached_generated_tokens"],
            "trajectory_generation_ms": trajectory["generation_ms"],
        }
        torch.save(payload, cache_path)

        manifest.append({
            "id": row["id"],
            "clip": str(row.get("clip", "")),
            "cache_file": str(cache_path),
            "input_fingerprint": row_fingerprint(row),
            "reasoning_text": reasoning["generated_text"],
            "trajectory_text": trajectory["generated_text"],
            "reasoning_trace_tokens": int(reasoning["cached_generated_tokens"]),
            "trajectory_trace_tokens": int(trajectory["cached_generated_tokens"]),
        })

        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(rows) - index)
        print(
            f"[TRACE CACHE {index:04d}/{len(rows):04d}] id={row['id']} "
            f"R={reasoning['cached_generated_tokens']} T={trajectory['cached_generated_tokens']} "
            f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
            flush=True,
        )
        del payload, reasoning, trajectory
        if index % 20 == 0:
            cleanup_cuda()

    write_jsonl(out_dir / "manifest.jsonl", manifest)
    save_json(root / "meta.json", {
        "vlm": str(vlm_path.resolve()),
        "vlm_signature": model_dir_signature(vlm_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "samples": len(rows),
        "decode": "greedy",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    })
    return manifest


# =============================================================================
# DECISION CACHE
# =============================================================================

def decision_cache_request(vlm_path: Path, dataset_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "vlm": str(vlm_path.resolve()),
        "vlm_signature": model_dir_signature(vlm_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "max_new_tokens": int(args.decision_max_new_tokens),
        "attn": args.attn,
        "dtype": args.dtype,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "decode": "greedy",
    }


def load_decision_cache(path: Path, meta_path: Path, rows: Sequence[Dict[str, Any]], request: Dict[str, Any]):
    if not path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("request") != request:
            return None
        data = read_jsonl(path)
    except Exception:
        return None
    if len(data) < len(rows):
        return None
    data = data[:len(rows)]
    if [str(x.get("id")) for x in data] != [r["id"] for r in rows]:
        return None
    if any(str(x.get("input_fingerprint")) != row_fingerprint(r) for x, r in zip(data, rows)):
        return None
    return data


def build_decision_cache(
    path: Path,
    meta_path: Path,
    rows: Sequence[Dict[str, Any]],
    model,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    vlm_path: Path,
    dataset_path: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for index, row in enumerate(rows, 1):
        generated = generate_text_only(
            model, processor, row, "decision", device, dtype, args.decision_max_new_tokens
        )
        parsed = parse_decision_text(generated["text"])
        records.append({
            "id": row["id"],
            "clip": str(row.get("clip", "")),
            "input_fingerprint": row_fingerprint(row),
            "decision_text": generated["text"],
            "decision_tokens": generated["tokens"],
            "decision_generation_ms": generated["generation_ms"],
            "decision_parse_ok": parsed["parse_ok"],
            "decision_longitudinal": parsed["longitudinal"],
            "decision_lateral": parsed["lateral"],
        })
        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(rows) - index)
        if index % 10 == 0 or index == len(rows):
            print(
                f"[DECISION {index:04d}/{len(rows):04d}] id={row['id']} "
                f"parse={'Y' if parsed['parse_ok'] else 'N'} "
                f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
                flush=True,
            )
    write_jsonl(path, records)
    save_json(meta_path, {"request": decision_cache_request(vlm_path, dataset_path, args)})
    return records


# =============================================================================
# RAW-KV FLOW / DiT ARCHITECTURE (exact checkpoint-compatible standalone copy)
# =============================================================================


ATTENTION_DIAGNOSTIC_METRICS = (
    "vlm_attention_mass",
    "self_attention_mass",
    "vlm_attention_entropy_nats",
    "vlm_effective_token_fraction",
    "reasoning_attention_mass",
    "reasoning_attention_share_within_vlm",
    "reasoning_attention_enrichment",
    "reasoning_attention_entropy_nats",
    "reasoning_attention_entropy_normalized",
    "reasoning_effective_token_fraction",
    "top1_reasoning_token_share",
    "top1_reasoning_token_share_within_vlm",
    "attended_context_uniform_mean_v_cosine",
)


class AttentionDiagnosticRecorder:
    """Non-parametric recorder for RAW-KV Action-Expert attention diagnostics.

    The recorder never changes logits, attention weights, model parameters, or
    the checkpoint state_dict.  It only observes the pre-dropout softmax
    attention tensor produced by RawKVPrefixFusionAttention.

    Token regions for the reasoning-conditioned AE are:
        [0, direct_tokens)                         : VLM prompt/prefill
        [direct_tokens, direct_tokens+reasoning_tokens) : generated reasoning

    All scalar metrics are averaged over AE query heads and 10 trajectory-query
    tokens for a single (sample, layer, Euler step) cell.

    The stored attended-context vector keeps the full attention-head structure:
    attention over valid VLM tokens -> V vectors -> mean over trajectory queries
    -> flatten [Hq, head_dim] to [Hq * head_dim].
    """

    def __init__(
        self,
        role: str,
        num_layers: int,
        solver_steps: int,
        store_vectors: bool = True,
    ) -> None:
        self.role = str(role)
        self.num_layers = int(num_layers)
        self.solver_steps = int(solver_steps)
        self.store_vectors = bool(store_vectors)

        self.sample_ids: List[str] = []
        self.records: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
        self.vectors: Dict[str, Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = {}
        self.sample_token_regions: Dict[str, Dict[str, int]] = {}

        self._sample_id: Optional[str] = None
        self._euler_step: Optional[int] = None
        self._euler_t: Optional[float] = None
        self._direct_tokens: int = 0
        self._reasoning_tokens: int = 0

    def begin_sample(
        self,
        sample_id: str,
        direct_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        sid = str(sample_id)
        if self._sample_id is not None:
            raise RuntimeError(
                f"AttentionDiagnosticRecorder sample already active: {self._sample_id}"
            )
        if sid in self.records:
            raise RuntimeError(f"Duplicate attention diagnostic sample: {sid}")
        self._sample_id = sid
        self._euler_step = None
        self._euler_t = None
        self._direct_tokens = int(direct_tokens)
        self._reasoning_tokens = int(reasoning_tokens)
        self.sample_ids.append(sid)
        self.records[sid] = {}
        self.vectors[sid] = {}
        self.sample_token_regions[sid] = {
            "direct_tokens": int(direct_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "total_vlm_tokens": int(direct_tokens + reasoning_tokens),
        }

    def set_euler_step(self, step: int, t: float) -> None:
        if self._sample_id is None:
            raise RuntimeError("set_euler_step() called without active sample")
        step = int(step)
        if step < 0 or step >= self.solver_steps:
            raise ValueError(f"Euler step out of range: {step}")
        self._euler_step = step
        self._euler_t = float(t)

    @staticmethod
    def _safe_mean(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return float("nan")
        value = float(x.detach().float().mean().cpu().item())
        return value if math.isfinite(value) else float("nan")

    @staticmethod
    def _entropy_from_normalized(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
        eps = 1.0e-12
        return -(p * torch.log(p.clamp_min(eps))).sum(dim=dim)

    def record(
        self,
        layer_index: int,
        attn: torch.Tensor,
        expanded_vlm_value: torch.Tensor,
        memory_mask: torch.Tensor,
        prefix_len: int,
    ) -> None:
        """Record one layer at the currently active Euler step.

        Parameters
        ----------
        attn:
            Softmax attention [B, Hq, Nquery, prefix_len + Nquery], before dropout.
        expanded_vlm_value:
            VLM prefix values after GQA repeat [B, Hq, prefix_len, head_dim].
        memory_mask:
            Valid VLM prefix mask [B, prefix_len].
        """
        if self._sample_id is None or self._euler_step is None:
            raise RuntimeError("Attention diagnostic record called outside sample/Euler step")
        if attn.ndim != 4 or expanded_vlm_value.ndim != 4:
            raise ValueError(
                f"Bad attention/value ranks: attn={attn.shape}, V={expanded_vlm_value.shape}"
            )
        if int(attn.shape[0]) != 1:
            raise ValueError("AttentionDiagnosticRecorder currently expects batch size 1")

        layer_index = int(layer_index)
        step = int(self._euler_step)
        cell = (layer_index, step)
        if cell in self.records[self._sample_id]:
            raise RuntimeError(
                f"Duplicate attention diagnostic cell sid={self._sample_id} "
                f"layer={layer_index} step={step}"
            )

        prefix_len = int(prefix_len)
        if prefix_len <= 0:
            raise ValueError("prefix_len must be > 0")
        if int(expanded_vlm_value.shape[2]) != prefix_len:
            raise ValueError(
                f"VLM V length mismatch: {expanded_vlm_value.shape[2]} vs {prefix_len}"
            )

        # Float32 diagnostics only; this does not feed back into model inference.
        a = attn[..., :prefix_len].detach().float()
        v = expanded_vlm_value.detach().float()
        valid = memory_mask[:, None, None, :prefix_len].bool()
        valid_f = valid.float()

        # The AE attention also includes AE self tokens.  Therefore this is the
        # absolute mass allocated to all VLM memory versus AE self-attention.
        masked_a = a * valid_f
        vlm_mass = masked_a.sum(dim=-1)
        self_mass = (1.0 - vlm_mass).clamp(min=0.0, max=1.0)

        eps = 1.0e-12
        p_vlm = masked_a / vlm_mass.unsqueeze(-1).clamp_min(eps)
        vlm_entropy = self._entropy_from_normalized(p_vlm)
        valid_tokens = valid_f[:, 0, 0, :].sum(dim=-1).clamp_min(1.0)
        vlm_effective_fraction = (
            torch.exp(vlm_entropy)
            / valid_tokens[:, None, None]
        ).clamp(min=0.0, max=1.0)

        direct_end = min(max(self._direct_tokens, 0), prefix_len)
        reasoning_start = direct_end
        reasoning_end = min(
            reasoning_start + max(self._reasoning_tokens, 0),
            prefix_len,
        )
        reasoning_count = max(0, reasoning_end - reasoning_start)

        metrics: Dict[str, float] = {
            "euler_t": float(self._euler_t),
            "vlm_attention_mass": self._safe_mean(vlm_mass),
            "self_attention_mass": self._safe_mean(self_mass),
            "vlm_attention_entropy_nats": self._safe_mean(vlm_entropy),
            "vlm_effective_token_fraction": self._safe_mean(vlm_effective_fraction),
            "reasoning_attention_mass": float("nan"),
            "reasoning_attention_share_within_vlm": float("nan"),
            "reasoning_attention_enrichment": float("nan"),
            "reasoning_attention_entropy_nats": float("nan"),
            "reasoning_attention_entropy_normalized": float("nan"),
            "reasoning_effective_token_fraction": float("nan"),
            "top1_reasoning_token_share": float("nan"),
            "top1_reasoning_token_share_within_vlm": float("nan"),
            "attended_context_uniform_mean_v_cosine": float("nan"),
        }

        # VLM attended context and uniform-mean V are computed in exactly the
        # same repeated-head value space used by the AE attention.
        attended = torch.matmul(p_vlm, v)                 # [1,Hq,Nq,D]
        attended_vec = attended.mean(dim=2).reshape(1, -1)
        valid_v = valid[:, :, 0, :].transpose(-2, -1)    # [1,T,1] only for mask
        del valid_v
        token_valid = memory_mask[:, :prefix_len].bool()
        token_valid_f = token_valid[:, None, :, None].float()
        uniform_v = (v * token_valid_f).sum(dim=2) / (
            token_valid_f.sum(dim=2).clamp_min(1.0)
        )                                                 # [1,Hq,D]
        uniform_vec = uniform_v.reshape(1, -1)
        context_cos = F.cosine_similarity(
            attended_vec, uniform_vec, dim=-1, eps=1.0e-8
        )
        metrics["attended_context_uniform_mean_v_cosine"] = self._safe_mean(context_cos)

        vector_entry: Dict[str, torch.Tensor] = {}
        if self.store_vectors:
            vector_entry["attended_context"] = (
                attended_vec[0].to(dtype=torch.float16).cpu().contiguous()
            )
            # Uniform V is constant across AE layers/Euler steps for a sample,
            # but storing it per sample later keeps the .pt payload reproducible.
            vector_entry["uniform_vlm_mean"] = (
                uniform_vec[0].to(dtype=torch.float16).cpu().contiguous()
            )

        if reasoning_count > 0:
            r = masked_a[..., reasoning_start:reasoning_end]
            r_mass = r.sum(dim=-1)
            r_share_vlm = r_mass / vlm_mass.clamp_min(eps)

            valid_count = float(valid_tokens[0].item())
            r_valid = token_valid[:, reasoning_start:reasoning_end]
            r_valid_count = int(r_valid.sum().item())
            token_fraction = (
                float(r_valid_count) / valid_count if valid_count > 0.0 else float("nan")
            )
            if token_fraction > 0.0:
                enrichment = r_share_vlm / token_fraction
            else:
                enrichment = torch.full_like(r_share_vlm, float("nan"))

            p_r = r / r_mass.unsqueeze(-1).clamp_min(eps)
            r_entropy = self._entropy_from_normalized(p_r)
            if r_valid_count > 1:
                r_entropy_norm = r_entropy / math.log(float(r_valid_count))
            elif r_valid_count == 1:
                r_entropy_norm = torch.zeros_like(r_entropy)
            else:
                r_entropy_norm = torch.full_like(r_entropy, float("nan"))
            if r_valid_count > 0:
                r_eff_frac = (
                    torch.exp(r_entropy) / float(r_valid_count)
                ).clamp(min=0.0, max=1.0)
            else:
                r_eff_frac = torch.full_like(r_entropy, float("nan"))

            top1_r_abs = r.max(dim=-1).values
            top1_r_share = top1_r_abs / r_mass.clamp_min(eps)
            top1_r_vlm_share = top1_r_abs / vlm_mass.clamp_min(eps)

            metrics.update({
                "reasoning_attention_mass": self._safe_mean(r_mass),
                "reasoning_attention_share_within_vlm": self._safe_mean(r_share_vlm),
                "reasoning_attention_enrichment": self._safe_mean(enrichment),
                "reasoning_attention_entropy_nats": self._safe_mean(r_entropy),
                "reasoning_attention_entropy_normalized": self._safe_mean(r_entropy_norm),
                "reasoning_effective_token_fraction": self._safe_mean(r_eff_frac),
                "top1_reasoning_token_share": self._safe_mean(top1_r_share),
                "top1_reasoning_token_share_within_vlm": self._safe_mean(top1_r_vlm_share),
            })

            # Reasoning-only attended value context.  This is useful for
            # cross-scene collapse analysis independent of prompt tokens.
            r_v = v[:, :, reasoning_start:reasoning_end, :]
            r_context = torch.matmul(p_r, r_v)
            r_context_vec = r_context.mean(dim=2).reshape(1, -1)
            if self.store_vectors:
                vector_entry["reasoning_context"] = (
                    r_context_vec[0].to(dtype=torch.float16).cpu().contiguous()
                )

        self.records[self._sample_id][cell] = metrics
        if self.store_vectors:
            self.vectors[self._sample_id][cell] = vector_entry

    def end_sample(self) -> Dict[str, float]:
        if self._sample_id is None:
            raise RuntimeError("end_sample() without active sample")
        sid = self._sample_id
        expected = self.num_layers * self.solver_steps
        actual = len(self.records[sid])
        if actual != expected:
            raise RuntimeError(
                f"Incomplete attention diagnostics sid={sid}: {actual}/{expected} cells"
            )
        summary = self.sample_summary(sid)
        self._sample_id = None
        self._euler_step = None
        self._euler_t = None
        self._direct_tokens = 0
        self._reasoning_tokens = 0
        return summary

    @staticmethod
    def _aggregate_cells(cells: Iterable[Dict[str, float]]) -> Dict[str, float]:
        cells = list(cells)
        out: Dict[str, float] = {}
        for metric in ATTENTION_DIAGNOSTIC_METRICS:
            vals = np.asarray(
                [finite_float(c.get(metric)) for c in cells],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            out[metric] = float(vals.mean()) if len(vals) else float("nan")
        return out

    def sample_summary(self, sample_id: str) -> Dict[str, float]:
        return self._aggregate_cells(self.records[str(sample_id)].values())

    def sample_layer_summary(self, sample_id: str, layer: int) -> Dict[str, float]:
        cells = [
            rec for (l, _), rec in self.records[str(sample_id)].items()
            if int(l) == int(layer)
        ]
        return self._aggregate_cells(cells)

    def sample_step_summary(self, sample_id: str, step: int) -> Dict[str, float]:
        cells = [
            rec for (_, s), rec in self.records[str(sample_id)].items()
            if int(s) == int(step)
        ]
        return self._aggregate_cells(cells)

    def overall_summary(self) -> Dict[str, float]:
        return self._aggregate_cells(
            rec
            for sid in self.sample_ids
            for rec in self.records[sid].values()
        )

    def layer_summary(self, layer: int) -> Dict[str, float]:
        return self._aggregate_cells(
            rec
            for sid in self.sample_ids
            for (l, _), rec in self.records[sid].items()
            if int(l) == int(layer)
        )

    def step_summary(self, step: int) -> Dict[str, float]:
        return self._aggregate_cells(
            rec
            for sid in self.sample_ids
            for (_, s), rec in self.records[sid].items()
            if int(s) == int(step)
        )

    def layer_step_summary(self, layer: int, step: int) -> Dict[str, float]:
        return self._aggregate_cells(
            self.records[sid][(int(layer), int(step))]
            for sid in self.sample_ids
        )

    def vector_tensor(self, vector_name: str) -> Optional[torch.Tensor]:
        if not self.store_vectors or not self.sample_ids:
            return None
        sample_tensors: List[torch.Tensor] = []
        for sid in self.sample_ids:
            cells: List[torch.Tensor] = []
            for layer in range(self.num_layers):
                for step in range(self.solver_steps):
                    entry = self.vectors[sid].get((layer, step), {})
                    vec = entry.get(vector_name)
                    if vec is None:
                        return None
                    cells.append(vec)
            stacked = torch.stack(cells, dim=0)
            sample_tensors.append(
                stacked.view(self.num_layers, self.solver_steps, -1)
            )
        return torch.stack(sample_tensors, dim=0).contiguous()

    def uniform_v_tensor(self) -> Optional[torch.Tensor]:
        """Return one uniform-mean-V vector per sample.

        The VLM memory is fixed during all AE layers/Euler steps, so the first
        recorded cell is sufficient and avoids [L,S] duplication.
        """
        if not self.store_vectors or not self.sample_ids:
            return None
        result: List[torch.Tensor] = []
        for sid in self.sample_ids:
            entry = self.vectors[sid].get((0, 0), {})
            vec = entry.get("uniform_vlm_mean")
            if vec is None:
                return None
            result.append(vec)
        return torch.stack(result, dim=0).contiguous()


@dataclass
class TrajectoryNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32)
        self.std = torch.as_tensor(self.std, dtype=torch.float32)
        if self.mean.shape != self.std.shape:
            raise ValueError("Normalizer mean/std mismatch")
        if torch.any(self.std <= 0):
            raise ValueError("Normalizer std must be >0")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
        )

    def to(self, device: torch.device | str) -> "TrajectoryNormalizer":
        return TrajectoryNormalizer(self.mean.to(device), self.std.to(device))

    def _stats_for(self, trajectory: torch.Tensor):
        if self.mean.ndim == 1:
            if trajectory.ndim == 2:
                return self.mean.view(1, 3), self.std.view(1, 3)
            return self.mean.view(1, 1, 3), self.std.view(1, 1, 3)
        if trajectory.ndim == 2:
            return self.mean, self.std
        return self.mean.unsqueeze(0), self.std.unsqueeze(0)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(trajectory)
        return trajectory * std.to(trajectory.device) + mean.to(trajectory.device)


class RawKVPrefixFusionAttention(nn.Module):
    def __init__(self, hidden_dim, vlm_num_attention_heads, vlm_num_kv_heads, vlm_head_dim, dropout, causal_queries):
        super().__init__()
        self.vlm_num_attention_heads = int(vlm_num_attention_heads)
        self.vlm_num_kv_heads = int(vlm_num_kv_heads)
        self.vlm_head_dim = int(vlm_head_dim)
        if self.vlm_num_attention_heads % self.vlm_num_kv_heads != 0:
            raise ValueError("VLM Hq must be divisible by Hkv")
        self.kv_groups = self.vlm_num_attention_heads // self.vlm_num_kv_heads
        self.q_attn_dim = self.vlm_num_attention_heads * self.vlm_head_dim
        self.kv_attn_dim = self.vlm_num_kv_heads * self.vlm_head_dim
        self.scale = self.vlm_head_dim ** -0.5
        self.causal_queries = bool(causal_queries)
        self.dropout_p = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, self.q_attn_dim, bias=False)
        self.self_k_proj = nn.Linear(hidden_dim, self.kv_attn_dim, bias=False)
        self.self_v_proj = nn.Linear(hidden_dim, self.kv_attn_dim, bias=False)
        self.out_proj = nn.Linear(self.q_attn_dim, hidden_dim, bias=False)

        # Non-parametric runtime-only diagnostics.  These attributes are not
        # part of state_dict, preserving exact checkpoint compatibility.
        self.diagnostic_recorder: Optional[AttentionDiagnosticRecorder] = None
        self.diagnostic_layer_index: int = -1

    def forward(self, query_tokens, vlm_key, vlm_value, memory_mask):
        b, n, _ = query_tokens.shape
        hq = self.vlm_num_attention_heads
        hkv = self.vlm_num_kv_heads
        d = self.vlm_head_dim
        q = self.q_proj(query_tokens).view(b, n, hq, d).transpose(1, 2)
        self_k = self.self_k_proj(query_tokens).view(b, n, hkv, d).transpose(1, 2)
        self_v = self.self_v_proj(query_tokens).view(b, n, hkv, d).transpose(1, 2)
        raw_k = vlm_key.to(device=q.device, dtype=q.dtype)
        raw_v = vlm_value.to(device=q.device, dtype=q.dtype)
        k = torch.cat([raw_k, self_k], dim=2)
        v = torch.cat([raw_v, self_v], dim=2)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        prefix_len = int(raw_k.shape[2])
        invalid_prefix = (~memory_mask.bool())[:, None, None, :].expand(b, 1, n, prefix_len)
        if self.causal_queries:
            blocked_self = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=scores.device), diagonal=1
            )[None, None, :, :].expand(b, 1, n, n)
        else:
            blocked_self = torch.zeros((b, 1, n, n), dtype=torch.bool, device=scores.device)
        blocked = torch.cat([invalid_prefix, blocked_self], dim=-1)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)

        # Record PRE-DROPOUT attention.  model.eval() makes dropout zero in this
        # experiment, but recording here guarantees the metric definition.
        if self.diagnostic_recorder is not None:
            self.diagnostic_recorder.record(
                layer_index=self.diagnostic_layer_index,
                attn=attn,
                expanded_vlm_value=v[:, :, :prefix_len, :],
                memory_mask=memory_mask,
                prefix_len=prefix_len,
            )

        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.q_attn_dim)
        return self.out_proj(out)

class TrajectoryOutputHeadRawKV(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.head(self.norm(x))


class RawKVEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, vlm_num_attention_heads, vlm_num_kv_heads, vlm_head_dim, ff_dim, dropout, causal_queries):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = RawKVPrefixFusionAttention(
            hidden_dim, vlm_num_attention_heads, vlm_num_kv_heads,
            vlm_head_dim, dropout, causal_queries
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, vlm_key, vlm_value, memory_mask):
        x = self.norm_attn(tokens)
        tokens = tokens + self.dropout(self.attn(x, vlm_key, vlm_value, memory_mask))
        tokens = tokens + self.dropout(self.ffn(self.norm_ff(tokens)))
        return tokens


class FourierEncoderRawKV(nn.Module):
    def __init__(self, dim=20, max_freq=100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0.0, math.log10(float(max_freq)), steps=half, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x):
        arg = x.float().unsqueeze(-1) * self.freqs.float() * (2.0 * math.pi)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2.0)


class FlowActionInputProjectionRawKV(nn.Module):
    def __init__(self, action_dim, hidden_dim, time_fourier_dim=20, time_mlp_hidden=512):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.time_encoder = FourierEncoderRawKV(time_fourier_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_fourier_dim, time_mlp_hidden), nn.LayerNorm(time_mlp_hidden),
            nn.GELU(), nn.Linear(time_mlp_hidden, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, x_t, t):
        b = x_t.shape[0]
        action_hidden = self.action_proj(x_t.to(dtype=self.action_proj[0].weight.dtype))
        t_scalar = t.reshape(b, -1)[:, 0]
        time_hidden = self.time_proj(
            self.time_encoder(t_scalar).to(dtype=self.time_proj[0].weight.dtype)
        ).unsqueeze(1)
        return action_hidden + time_hidden


class RawKVFlowMatchingActionExpert(nn.Module):
    def __init__(self, hidden_dim, num_steps, num_layers, vlm_num_attention_heads, vlm_num_kv_heads, vlm_head_dim, ff_dim, dropout):
        super().__init__()
        self.num_steps = int(num_steps)
        self.action_in = FlowActionInputProjectionRawKV(3, hidden_dim)
        self.horizon_embedding = nn.Parameter(torch.randn(1, self.num_steps, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            RawKVEncoderBlock(
                hidden_dim, vlm_num_attention_heads, vlm_num_kv_heads,
                vlm_head_dim, ff_dim, dropout, False
            )
            for _ in range(num_layers)
        ])
        self.output = TrajectoryOutputHeadRawKV(hidden_dim)

        for layer_index, layer in enumerate(self.layers):
            layer.attn.diagnostic_layer_index = int(layer_index)

    def attach_attention_diagnostic_recorder(
        self,
        recorder: Optional[AttentionDiagnosticRecorder],
    ) -> None:
        for layer_index, layer in enumerate(self.layers):
            layer.attn.diagnostic_layer_index = int(layer_index)
            layer.attn.diagnostic_recorder = recorder

    def forward(self, x_t, t, vlm_key, vlm_value, memory_mask):
        tokens = self.action_in(x_t, t)
        tokens = tokens + self.horizon_embedding.to(device=tokens.device, dtype=tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, vlm_key, vlm_value, memory_mask)
        return self.output(tokens)

def load_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model_state_dict", "action_config", "normalizer",
        "vlm_num_attention_heads", "vlm_num_kv_heads", "vlm_head_dim",
    ):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key}: {path}")
    return ckpt


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device):
    cfg = ckpt["action_config"]
    model = RawKVFlowMatchingActionExpert(
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        vlm_num_attention_heads=int(ckpt["vlm_num_attention_heads"]),
        vlm_num_kv_heads=int(ckpt["vlm_num_kv_heads"]),
        vlm_head_dim=int(ckpt["vlm_head_dim"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    normalizer = TrajectoryNormalizer.from_dict(ckpt["normalizer"]).to(device)
    solver_steps = int(cfg.get("solver_steps", 10))
    return model, normalizer, solver_steps


def validate_checkpoint_pair(direct: Dict[str, Any], reasoning: Dict[str, Any]) -> None:
    for key in ("vlm_num_attention_heads", "vlm_num_kv_heads", "vlm_head_dim"):
        if int(direct[key]) != int(reasoning[key]):
            raise RuntimeError(f"Checkpoint geometry mismatch: {key}")
    dc, rc = direct["action_config"], reasoning["action_config"]
    for key in ("hidden_dim", "num_steps", "num_layers", "ff_dim", "dropout", "solver_steps"):
        if dc.get(key) != rc.get(key):
            raise RuntimeError(f"Checkpoint action config mismatch {key}: {dc.get(key)} vs {rc.get(key)}")
    dn = TrajectoryNormalizer.from_dict(direct["normalizer"])
    rn = TrajectoryNormalizer.from_dict(reasoning["normalizer"])
    if not torch.equal(dn.mean, rn.mean) or not torch.equal(dn.std, rn.std):
        raise RuntimeError("Direct/reasoning normalizer mismatch")


@torch.inference_mode()
def euler_sample_rawkv_controlled(
    model,
    vlm_key,
    vlm_value,
    memory_mask,
    normalizer,
    solver_steps,
    noise_cpu,
    diagnostic_recorder: Optional[AttentionDiagnosticRecorder] = None,
):
    device = vlm_key.device
    x = noise_cpu.to(device=device, dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, solver_steps + 1, device=device)
    for i in range(solver_steps):
        t_now = float(times[i].item())
        dt = float((times[i + 1] - times[i]).item())
        if diagnostic_recorder is not None:
            diagnostic_recorder.set_euler_step(i, t_now)
        t = torch.full((1, 1, 1), t_now, device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            velocity = model(
                x_t=x, t=t, vlm_key=vlm_key, vlm_value=vlm_value, memory_mask=memory_mask
            )
        x = x + dt * velocity.float()
    return normalizer.denormalize(x)

def reasoning_memory_from_cache(cache: Dict[str, Any], reuse: bool):
    key = cache["reasoning_direct_key"]
    value = cache["reasoning_direct_value"]
    if reuse:
        dk = cache["reasoning_delta_key"]
        dv = cache["reasoning_delta_value"]
        key = torch.cat([key, dk], dim=1)
        value = torch.cat([value, dv], dim=1)
    mask = torch.ones((1, int(key.shape[1])), dtype=torch.bool)
    return key.unsqueeze(0), value.unsqueeze(0), mask


def evaluate_action_expert(
    checkpoint_path: Path,
    role: str,
    rows: Sequence[Dict[str, Any]],
    trace_manifest: Sequence[Dict[str, Any]],
    device: torch.device,
    noise_seed: int,
    solver_steps_override: Optional[int],
    store_attention_vectors: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], AttentionDiagnosticRecorder]:
    ckpt = load_checkpoint(checkpoint_path)
    model, normalizer, ckpt_steps = build_model_from_checkpoint(ckpt, device)
    solver_steps = int(solver_steps_override) if solver_steps_override else int(ckpt_steps)
    reuse = role == "reasoning"

    recorder = AttentionDiagnosticRecorder(
        role=role,
        num_layers=len(model.layers),
        solver_steps=solver_steps,
        store_vectors=store_attention_vectors,
    )
    model.attach_attention_diagnostic_recorder(recorder)

    output: Dict[str, Dict[str, Any]] = {}
    start = time.perf_counter()

    for index, (row, item) in enumerate(zip(rows, trace_manifest), 1):
        if row["id"] != str(item["id"]):
            raise RuntimeError(f"ID mismatch at {index}")
        cache = torch.load(item["cache_file"], map_location="cpu", weights_only=False)
        if str(cache["id"]) != row["id"]:
            raise RuntimeError(f"Cache ID mismatch id={row['id']}")

        direct_tokens = int(cache["reasoning_direct_key"].shape[1])
        available_reasoning_tokens = int(cache["reasoning_delta_key"].shape[1])
        active_reasoning_tokens = available_reasoning_tokens if reuse else 0

        key, value, mask = reasoning_memory_from_cache(cache, reuse=reuse)
        key = key.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        value = value.to(device=device, dtype=CACHE_DTYPE, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)
        noise = make_noise(row["id"], noise_seed, NUM_STEPS)

        recorder.begin_sample(
            sample_id=row["id"],
            direct_tokens=direct_tokens,
            reasoning_tokens=active_reasoning_tokens,
        )

        sync_cuda(device)
        t0 = time.perf_counter()
        pred = euler_sample_rawkv_controlled(
            model,
            key,
            value,
            mask,
            normalizer,
            solver_steps,
            noise,
            diagnostic_recorder=recorder,
        )
        sync_cuda(device)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        attention_summary = recorder.end_sample()

        pred_np = pred[0].float().cpu().numpy()
        gt_np = np.asarray(row["trajectory"], dtype=np.float64)
        m = xyz_metrics(pred_np, gt_np)
        output[row["id"]] = {
            "trajectory": pred_np.tolist(),
            "ade_m": m["ade_m"],
            "fde_m": m["fde_m"],
            "heading_mae_rad": m["heading_mae_rad"],
            "infer_ms": float(infer_ms),
            "memory_tokens": int(key.shape[2]),
            "direct_memory_tokens": direct_tokens,
            "reasoning_memory_tokens": active_reasoning_tokens,
            "solver_steps": solver_steps,
            "attention_summary": attention_summary,
        }
        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(rows) - index)
        if index % 10 == 0 or index == len(rows):
            extra = ""
            if reuse:
                extra = (
                    f" Rshare={fmt(attention_summary.get('reasoning_attention_share_within_vlm'))}"
                    f" enrich={fmt(attention_summary.get('reasoning_attention_enrichment'))}"
                )
            print(
                f"[AE {role.upper():9s} {index:04d}/{len(rows):04d}] "
                f"ADE={m['ade_m']:.3f} memT={key.shape[2]}{extra} "
                f"elapsed={format_eta(elapsed)} eta={format_eta(eta)}",
                flush=True,
            )
        del cache, key, value, mask, pred

    model.attach_attention_diagnostic_recorder(None)
    del model, normalizer, ckpt
    cleanup_cuda()
    return output, recorder


# =============================================================================
# STATISTICS
# =============================================================================

def clean_pair(x: Iterable[Any], y: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa = np.asarray([finite_float(v) for v in x], dtype=np.float64)
    ya = np.asarray([finite_float(v) for v in y], dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def pearson_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    a, b = clean_pair(x, y)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: Iterable[Any], y: Iterable[Any]) -> float:
    a, b = clean_pair(x, y)
    if len(a) < 3:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def corr_pair(x: Iterable[Any], y: Iterable[Any]) -> Dict[str, Any]:
    a, b = clean_pair(x, y)
    return {
        "n": int(len(a)),
        "pearson": pearson_corr(a, b),
        "spearman": spearman_corr(a, b),
    }


def partial_corr(x: Iterable[Any], y: Iterable[Any], controls: Sequence[Iterable[Any]]) -> Dict[str, Any]:
    arrays = [np.asarray([finite_float(v) for v in x], dtype=np.float64),
              np.asarray([finite_float(v) for v in y], dtype=np.float64)]
    arrays += [np.asarray([finite_float(v) for v in c], dtype=np.float64) for c in controls]
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    arr = [a[mask] for a in arrays]
    if len(arr[0]) < max(5, len(controls) + 3):
        return {"n": int(len(arr[0])), "pearson": float("nan"), "spearman": float("nan")}
    if not controls:
        return corr_pair(arr[0], arr[1])
    C = np.column_stack(arr[2:])
    C = np.column_stack([np.ones(len(C)), C])
    bx, *_ = np.linalg.lstsq(C, arr[0], rcond=None)
    by, *_ = np.linalg.lstsq(C, arr[1], rcond=None)
    rx = arr[0] - C @ bx
    ry = arr[1] - C @ by
    return {"n": int(len(rx)), "pearson": pearson_corr(rx, ry), "spearman": spearman_corr(rx, ry)}


def bootstrap_mean_ci(values: Iterable[Any], iterations: int, seed: int) -> Dict[str, float]:
    v = np.asarray([finite_float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan"), "n": 0}
    if iterations <= 0:
        return {"mean": float(v.mean()), "lo95": float("nan"), "hi95": float("nan"), "n": int(len(v))}
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        means[i] = v[rng.integers(0, len(v), size=len(v))].mean()
    return {
        "mean": float(v.mean()),
        "lo95": float(np.percentile(means, 2.5)),
        "hi95": float(np.percentile(means, 97.5)),
        "n": int(len(v)),
    }


def group_metric(samples: Sequence[Dict[str, Any]], predicate, metric: str, bootstrap: int, seed: int):
    vals = [s[metric] for s in samples if predicate(s) and math.isfinite(finite_float(s.get(metric)))]
    return bootstrap_mean_ci(vals, bootstrap, seed)


def standardized_regression(samples: Sequence[Dict[str, Any]], target: str, predictors: Sequence[str]) -> Dict[str, Any]:
    rows = []
    ys = []
    for s in samples:
        y = finite_float(s.get(target))
        xs = [finite_float(s.get(k)) for k in predictors]
        if math.isfinite(y) and all(math.isfinite(x) for x in xs):
            ys.append(y)
            rows.append(xs)
    if len(rows) < len(predictors) + 5:
        return {"n": len(rows), "r2": float("nan"), "coefficients": {k: float("nan") for k in predictors}}
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xmean, xstd = X.mean(axis=0), X.std(axis=0)
    ymean, ystd = y.mean(), y.std()
    keep = xstd > 1e-12
    Xs = np.zeros_like(X)
    Xs[:, keep] = (X[:, keep] - xmean[keep]) / xstd[keep]
    ys_std = (y - ymean) / max(ystd, 1e-12)
    design = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(design, ys_std, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((ys_std - pred) ** 2))
    ss_tot = float(np.sum((ys_std - ys_std.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(X)),
        "r2": float(r2),
        "target": target,
        "coefficients": {name: float(beta[i + 1]) for i, name in enumerate(predictors)},
        "note": "standardized OLS coefficients; association diagnostic, not causal identification",
    }


# =============================================================================
# ATTENTION DIAGNOSTIC AGGREGATION / COLLAPSE / CORRELATION
# =============================================================================

COMMON_ATTENTION_METRICS = (
    "vlm_attention_mass",
    "self_attention_mass",
    "vlm_attention_entropy_nats",
    "vlm_effective_token_fraction",
    "attended_context_uniform_mean_v_cosine",
)

REASONING_ATTENTION_METRICS = (
    "reasoning_attention_mass",
    "reasoning_attention_share_within_vlm",
    "reasoning_attention_enrichment",
    "reasoning_attention_entropy_nats",
    "reasoning_attention_entropy_normalized",
    "reasoning_effective_token_fraction",
    "top1_reasoning_token_share",
    "top1_reasoning_token_share_within_vlm",
)


def _mean_dicts(dicts: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([finite_float(d.get(key)) for d in dicts], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        result[key] = float(vals.mean()) if len(vals) else float("nan")
    return result


def cross_scene_collapse_stats(vectors: Optional[torch.Tensor]) -> Dict[str, float]:
    """Measure whether scene vectors collapse toward the same direction.

    `cross_scene_mean_cosine` is the exact mean cosine over all ordered
    off-diagonal scene pairs, computed without materializing NxN similarities.
    A value near 1 means strong directional collapse; near 0 means diverse/
    approximately orthogonal scene representations.

    `centroid_norm_ratio` = ||mean(v)|| / mean(||v||).  This is another
    scale-aware collapse indicator in [approximately 0, 1].
    """
    if vectors is None:
        return {
            "n": 0,
            "cross_scene_mean_cosine": float("nan"),
            "centroid_norm_ratio": float("nan"),
            "mean_vector_norm": float("nan"),
            "mean_feature_std": float("nan"),
        }

    x = vectors.detach().float().cpu()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    finite_rows = torch.isfinite(x).all(dim=1)
    x = x[finite_rows]
    norms = torch.linalg.vector_norm(x, dim=1)
    keep = norms > 1.0e-12
    x = x[keep]
    norms = norms[keep]
    n = int(x.shape[0])
    if n == 0:
        return {
            "n": 0,
            "cross_scene_mean_cosine": float("nan"),
            "centroid_norm_ratio": float("nan"),
            "mean_vector_norm": float("nan"),
            "mean_feature_std": float("nan"),
        }

    unit = x / norms[:, None]
    if n >= 2:
        sum_unit = unit.sum(dim=0)
        mean_cos = (
            float(torch.dot(sum_unit, sum_unit).item()) - float(n)
        ) / float(n * (n - 1))
    else:
        mean_cos = float("nan")

    mean_norm = float(norms.mean().item())
    centroid_norm = float(torch.linalg.vector_norm(x.mean(dim=0)).item())
    centroid_ratio = centroid_norm / max(mean_norm, 1.0e-12)
    feature_std = float(x.std(dim=0, unbiased=False).mean().item())

    return {
        "n": n,
        "cross_scene_mean_cosine": float(mean_cos),
        "centroid_norm_ratio": float(centroid_ratio),
        "mean_vector_norm": float(mean_norm),
        "mean_feature_std": float(feature_std),
    }


def recorder_cell_vectors(
    recorder: AttentionDiagnosticRecorder,
    vector_name: str,
    layer: int,
    step: int,
) -> Optional[torch.Tensor]:
    rows: List[torch.Tensor] = []
    for sid in recorder.sample_ids:
        vec = recorder.vectors.get(sid, {}).get((int(layer), int(step)), {}).get(vector_name)
        if vec is None:
            return None
        rows.append(vec)
    return torch.stack(rows, dim=0) if rows else None


def build_recorder_role_diagnostics(
    recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    by_layer: Dict[str, Any] = {}
    by_step: Dict[str, Any] = {}
    by_layer_step: Dict[str, Any] = {}

    for layer in range(recorder.num_layers):
        by_layer[str(layer)] = recorder.layer_summary(layer)
    for step in range(recorder.solver_steps):
        summary = recorder.step_summary(step)
        summary["euler_t"] = float(step / recorder.solver_steps)
        by_step[str(step)] = summary
    for layer in range(recorder.num_layers):
        layer_obj: Dict[str, Any] = {}
        for step in range(recorder.solver_steps):
            cell = recorder.layer_step_summary(layer, step)
            cell["euler_t"] = float(step / recorder.solver_steps)
            layer_obj[str(step)] = cell
        by_layer_step[str(layer)] = layer_obj

    return {
        "role": recorder.role,
        "num_layers": recorder.num_layers,
        "solver_steps": recorder.solver_steps,
        "overall": recorder.overall_summary(),
        "by_layer": by_layer,
        "by_euler_step": by_step,
        "by_layer_step": by_layer_step,
        "per_sample_overall": {
            sid: recorder.sample_summary(sid) for sid in recorder.sample_ids
        },
        "sample_token_regions": recorder.sample_token_regions,
    }


def build_cross_scene_collapse(
    recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    vector_names = ["attended_context"]
    if recorder.role == "reasoning":
        vector_names.append("reasoning_context")

    result: Dict[str, Any] = {}
    for vector_name in vector_names:
        layer_step: Dict[str, Any] = {}
        layer_stats_lists: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        step_stats_lists: Dict[int, List[Dict[str, float]]] = defaultdict(list)

        for layer in range(recorder.num_layers):
            layer_obj: Dict[str, Any] = {}
            for step in range(recorder.solver_steps):
                stats = cross_scene_collapse_stats(
                    recorder_cell_vectors(recorder, vector_name, layer, step)
                )
                stats["euler_t"] = float(step / recorder.solver_steps)
                layer_obj[str(step)] = stats
                layer_stats_lists[layer].append(stats)
                step_stats_lists[step].append(stats)
            layer_step[str(layer)] = layer_obj

        collapse_keys = (
            "cross_scene_mean_cosine",
            "centroid_norm_ratio",
            "mean_vector_norm",
            "mean_feature_std",
        )
        by_layer = {
            str(layer): _mean_dicts(stats, collapse_keys)
            for layer, stats in layer_stats_lists.items()
        }
        by_step = {
            str(step): {
                **_mean_dicts(stats, collapse_keys),
                "euler_t": float(step / recorder.solver_steps),
            }
            for step, stats in step_stats_lists.items()
        }
        overall = _mean_dicts(
            [s for stats in layer_stats_lists.values() for s in stats],
            collapse_keys,
        )
        overall["n_samples"] = len(recorder.sample_ids)

        result[vector_name] = {
            "overall": overall,
            "by_layer": by_layer,
            "by_euler_step": by_step,
            "by_layer_step": layer_step,
        }
    return result


def paired_direct_reasoning_vector_cosine(
    direct_recorder: AttentionDiagnosticRecorder,
    reasoning_recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    if direct_recorder.sample_ids != reasoning_recorder.sample_ids:
        raise RuntimeError("Direct/reasoning attention recorder sample order mismatch")
    if (
        direct_recorder.num_layers != reasoning_recorder.num_layers
        or direct_recorder.solver_steps != reasoning_recorder.solver_steps
    ):
        raise RuntimeError("Direct/reasoning attention recorder geometry mismatch")

    by_layer_step: Dict[str, Any] = {}
    all_values: List[float] = []
    layer_values: Dict[int, List[float]] = defaultdict(list)
    step_values: Dict[int, List[float]] = defaultdict(list)

    for layer in range(direct_recorder.num_layers):
        layer_obj: Dict[str, Any] = {}
        for step in range(direct_recorder.solver_steps):
            dv = recorder_cell_vectors(direct_recorder, "attended_context", layer, step)
            rv = recorder_cell_vectors(reasoning_recorder, "attended_context", layer, step)
            if dv is None or rv is None:
                vals = np.asarray([], dtype=np.float64)
            else:
                cos = F.cosine_similarity(
                    dv.float(), rv.float(), dim=-1, eps=1.0e-8
                )
                vals = cos.detach().cpu().numpy().astype(np.float64)
                vals = vals[np.isfinite(vals)]
            mean_value = float(vals.mean()) if len(vals) else float("nan")
            layer_obj[str(step)] = {
                "n": int(len(vals)),
                "mean_paired_cosine": mean_value,
                "euler_t": float(step / direct_recorder.solver_steps),
            }
            if math.isfinite(mean_value):
                all_values.append(mean_value)
                layer_values[layer].append(mean_value)
                step_values[step].append(mean_value)
        by_layer_step[str(layer)] = layer_obj

    return {
        "overall_mean_paired_cosine": float(np.mean(all_values)) if all_values else float("nan"),
        "by_layer": {
            str(k): float(np.mean(v)) if v else float("nan")
            for k, v in layer_values.items()
        },
        "by_euler_step": {
            str(k): {
                "mean_paired_cosine": float(np.mean(v)) if v else float("nan"),
                "euler_t": float(k / direct_recorder.solver_steps),
            }
            for k, v in step_values.items()
        },
        "by_layer_step": by_layer_step,
    }


def _corr_metrics_with_gain(
    metric_dicts: Sequence[Dict[str, Any]],
    gains: Sequence[float],
    metrics: Sequence[str],
) -> Dict[str, Any]:
    return {
        metric: corr_pair(
            [d.get(metric, float("nan")) for d in metric_dicts],
            gains,
        )
        for metric in metrics
    }


def build_attention_gain_correlations(
    samples: Sequence[Dict[str, Any]],
    direct_recorder: AttentionDiagnosticRecorder,
    reasoning_recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    if [s["id"] for s in samples] != reasoning_recorder.sample_ids:
        raise RuntimeError("Sample order mismatch for attention/gain correlation")
    if direct_recorder.sample_ids != reasoning_recorder.sample_ids:
        raise RuntimeError("Direct/reasoning recorder order mismatch")

    gains = [finite_float(s["reasoning_ae_gain_ade_m"]) for s in samples]
    reasoning_metrics = COMMON_ATTENTION_METRICS + REASONING_ATTENTION_METRICS

    reasoning_overall_dicts = [
        reasoning_recorder.sample_summary(sid)
        for sid in reasoning_recorder.sample_ids
    ]

    by_layer: Dict[str, Any] = {}
    for layer in range(reasoning_recorder.num_layers):
        metric_dicts = [
            reasoning_recorder.sample_layer_summary(sid, layer)
            for sid in reasoning_recorder.sample_ids
        ]
        by_layer[str(layer)] = _corr_metrics_with_gain(
            metric_dicts, gains, reasoning_metrics
        )

    by_step: Dict[str, Any] = {}
    for step in range(reasoning_recorder.solver_steps):
        metric_dicts = [
            reasoning_recorder.sample_step_summary(sid, step)
            for sid in reasoning_recorder.sample_ids
        ]
        by_step[str(step)] = {
            "euler_t": float(step / reasoning_recorder.solver_steps),
            "correlations": _corr_metrics_with_gain(
                metric_dicts, gains, reasoning_metrics
            ),
        }

    # Direct-vs-reasoning deltas for metrics that exist in both conditions.
    delta_dicts: List[Dict[str, float]] = []
    for sid in reasoning_recorder.sample_ids:
        d = direct_recorder.sample_summary(sid)
        r = reasoning_recorder.sample_summary(sid)
        delta_dicts.append({
            metric: finite_float(r.get(metric)) - finite_float(d.get(metric))
            for metric in COMMON_ATTENTION_METRICS
        })

    return {
        "target": "reasoning_ae_gain_ade_m = direct_AE_ADE - reasoning_AE_ADE; higher is better",
        "reasoning_attention_overall": _corr_metrics_with_gain(
            reasoning_overall_dicts, gains, reasoning_metrics
        ),
        "reasoning_attention_by_layer": by_layer,
        "reasoning_attention_by_euler_step": by_step,
        "reasoning_minus_direct_overall": _corr_metrics_with_gain(
            delta_dicts, gains, COMMON_ATTENTION_METRICS
        ),
    }


def build_attention_diagnostics(
    samples: Sequence[Dict[str, Any]],
    direct_recorder: AttentionDiagnosticRecorder,
    reasoning_recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    direct_role = build_recorder_role_diagnostics(direct_recorder)
    reasoning_role = build_recorder_role_diagnostics(reasoning_recorder)
    direct_collapse = build_cross_scene_collapse(direct_recorder)
    reasoning_collapse = build_cross_scene_collapse(reasoning_recorder)

    # Explicit reasoning-minus-direct collapse delta for attended context.
    collapse_delta: Dict[str, Any] = {
        "by_layer_step": {},
        "by_layer": {},
        "by_euler_step": {},
    }
    d_att = direct_collapse["attended_context"]
    r_att = reasoning_collapse["attended_context"]

    for layer in range(reasoning_recorder.num_layers):
        layer_obj: Dict[str, Any] = {}
        for step in range(reasoning_recorder.solver_steps):
            d = d_att["by_layer_step"][str(layer)][str(step)]
            r = r_att["by_layer_step"][str(layer)][str(step)]
            layer_obj[str(step)] = {
                "cross_scene_mean_cosine_delta": (
                    finite_float(r.get("cross_scene_mean_cosine"))
                    - finite_float(d.get("cross_scene_mean_cosine"))
                ),
                "centroid_norm_ratio_delta": (
                    finite_float(r.get("centroid_norm_ratio"))
                    - finite_float(d.get("centroid_norm_ratio"))
                ),
                "euler_t": float(step / reasoning_recorder.solver_steps),
            }
        collapse_delta["by_layer_step"][str(layer)] = layer_obj

    for layer in range(reasoning_recorder.num_layers):
        d = d_att["by_layer"][str(layer)]
        r = r_att["by_layer"][str(layer)]
        collapse_delta["by_layer"][str(layer)] = {
            "cross_scene_mean_cosine_delta": (
                finite_float(r.get("cross_scene_mean_cosine"))
                - finite_float(d.get("cross_scene_mean_cosine"))
            ),
            "centroid_norm_ratio_delta": (
                finite_float(r.get("centroid_norm_ratio"))
                - finite_float(d.get("centroid_norm_ratio"))
            ),
        }

    for step in range(reasoning_recorder.solver_steps):
        d = d_att["by_euler_step"][str(step)]
        r = r_att["by_euler_step"][str(step)]
        collapse_delta["by_euler_step"][str(step)] = {
            "cross_scene_mean_cosine_delta": (
                finite_float(r.get("cross_scene_mean_cosine"))
                - finite_float(d.get("cross_scene_mean_cosine"))
            ),
            "centroid_norm_ratio_delta": (
                finite_float(r.get("centroid_norm_ratio"))
                - finite_float(d.get("centroid_norm_ratio"))
            ),
            "euler_t": float(step / reasoning_recorder.solver_steps),
        }

    return {
        "version": "attention_diagnostics_v1",
        "metric_definitions": {
            "vlm_attention_mass":
                "absolute AE attention probability mass assigned to valid VLM prefix tokens; self-attention occupies the remainder",
            "reasoning_attention_mass":
                "absolute AE attention probability mass assigned to generated reasoning-token positions",
            "reasoning_attention_share_within_vlm":
                "reasoning_attention_mass / total VLM-prefix attention mass",
            "reasoning_attention_enrichment":
                "reasoning_attention_share_within_vlm / (reasoning-token count / valid VLM-token count); >1 means selective enrichment beyond token quantity",
            "reasoning_attention_entropy_nats":
                "entropy of attention renormalized only across reasoning-token positions",
            "reasoning_attention_entropy_normalized":
                "reasoning entropy divided by log(number of reasoning tokens)",
            "reasoning_effective_token_fraction":
                "exp(reasoning entropy) / number of reasoning tokens; 1 means broad/uniform use, small values mean concentration",
            "top1_reasoning_token_share":
                "largest single reasoning-token attention divided by total reasoning-token attention",
            "top1_reasoning_token_share_within_vlm":
                "largest single reasoning-token attention divided by total VLM-prefix attention",
            "attended_context_uniform_mean_v_cosine":
                "cosine between the AE attention-weighted VLM V-context vector and the uniform-mean V vector over valid VLM tokens",
            "cross_scene_mean_cosine":
                "mean off-diagonal cosine among scene attended-context vectors at a fixed layer/Euler step; near 1 indicates directional representation collapse",
            "centroid_norm_ratio":
                "||mean(scene vector)|| / mean(||scene vector||); higher means stronger shared-direction collapse",
        },
        "vector_definition":
            "per sample/layer/Euler step: attention over valid VLM tokens is renormalized, multiplied by repeated-head V values, averaged over the 10 AE trajectory queries, then flattened as [Hq * head_dim]",
        "direct": direct_role,
        "reasoning": reasoning_role,
        "cross_scene_collapse": {
            "direct": direct_collapse,
            "reasoning": reasoning_collapse,
            "reasoning_minus_direct": collapse_delta,
            "paired_direct_reasoning_context_cosine":
                paired_direct_reasoning_vector_cosine(
                    direct_recorder, reasoning_recorder
                ),
        },
        "attention_metric_vs_ae_gain_correlation":
            build_attention_gain_correlations(
                samples, direct_recorder, reasoning_recorder
            ),
    }


def build_attention_vector_payload(
    direct_recorder: AttentionDiagnosticRecorder,
    reasoning_recorder: AttentionDiagnosticRecorder,
) -> Dict[str, Any]:
    if direct_recorder.sample_ids != reasoning_recorder.sample_ids:
        raise RuntimeError("Cannot save attention vectors: sample order mismatch")

    direct_attended = direct_recorder.vector_tensor("attended_context")
    reasoning_attended = reasoning_recorder.vector_tensor("attended_context")
    reasoning_context = reasoning_recorder.vector_tensor("reasoning_context")
    direct_uniform = direct_recorder.uniform_v_tensor()
    reasoning_uniform = reasoning_recorder.uniform_v_tensor()

    if direct_attended is None or reasoning_attended is None or reasoning_context is None:
        raise RuntimeError("Attention vectors were not recorded")

    return {
        "version": "attention_vectors_v1",
        "sample_ids": list(direct_recorder.sample_ids),
        "layout": {
            "attended_context": "[sample, layer, euler_step, Hq*head_dim]",
            "reasoning_context": "[sample, layer, euler_step, Hq*head_dim]",
            "uniform_vlm_mean": "[sample, Hq*head_dim]",
        },
        "dtype": "float16",
        "num_layers": int(direct_recorder.num_layers),
        "solver_steps": int(direct_recorder.solver_steps),
        "vector_definition":
            "VLM-attended context uses VLM-prefix-normalized AE attention, mean over 10 trajectory queries, full repeated Hq head structure flattened",
        "direct": {
            "attended_context": direct_attended,
            "uniform_vlm_mean": direct_uniform,
        },
        "reasoning": {
            "attended_context": reasoning_attended,
            "reasoning_context": reasoning_context,
            "uniform_vlm_mean": reasoning_uniform,
        },
    }


# =============================================================================
# SAMPLE ANALYSIS
# =============================================================================

def bool_float(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return float("nan")


def analyze_samples(
    rows: Sequence[Dict[str, Any]],
    trace_manifest: Sequence[Dict[str, Any]],
    decision_records: Sequence[Dict[str, Any]],
    direct_ae: Dict[str, Dict[str, Any]],
    reasoning_ae: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, (row, trace_item, dec_rec) in enumerate(zip(rows, trace_manifest, decision_records)):
        sid = row["id"]
        if sid != str(trace_item["id"]) or sid != str(dec_rec["id"]):
            raise RuntimeError(f"Paired ID mismatch at index={index}")
        cache = torch.load(trace_item["cache_file"], map_location="cpu", weights_only=False)
        reasoning_text = str(cache.get("reasoning_text", trace_item.get("reasoning_text", ""))).strip()
        trajectory_text = str(cache.get("trajectory_text", trace_item.get("trajectory_text", ""))).strip()
        reasoning_tokens = int(cache.get("reasoning_cached_tokens", trace_item.get("reasoning_trace_tokens", 0)))
        trajectory_tokens = int(cache.get("trajectory_cached_tokens", trace_item.get("trajectory_trace_tokens", 0)))

        gt = np.asarray(row["trajectory"], dtype=np.float64)
        vlm_xy = parse_vlm_trajectory(trajectory_text)
        vlm_metrics = xy_metrics(vlm_xy, gt) if vlm_xy is not None else {"ade_m": float("nan"), "fde_m": float("nan")}

        d = direct_ae[sid]
        r = reasoning_ae[sid]
        dtraj = np.asarray(d["trajectory"], dtype=np.float64)
        rtraj = np.asarray(r["trajectory"], dtype=np.float64)

        gt_motion = trajectory_motion(gt, row["speed_mps"])
        direct_motion = trajectory_motion(dtraj, row["speed_mps"])
        reasoning_motion = trajectory_motion(rtraj, row["speed_mps"])
        vlm_motion = trajectory_motion(vlm_xy, row["speed_mps"]) if vlm_xy is not None else None

        pred_lon = normalize_lon_label(str(dec_rec.get("decision_longitudinal", "")))
        pred_lat = normalize_lat_label(str(dec_rec.get("decision_lateral", "")))
        gt_lon = normalize_lon_label(row["longitudinal"])
        gt_lat = normalize_lat_label(row["lateral"])

        decision_lon_correct = bool(pred_lon and gt_lon and pred_lon == gt_lon)
        decision_lat_correct = bool(pred_lat and gt_lat and pred_lat == gt_lat)
        decision_joint_correct = bool(pred_lon and pred_lat and gt_lon and gt_lat and decision_lon_correct and decision_lat_correct)

        r_intent = reasoning_intent(reasoning_text)
        reasoning_vs_decision = decision_pair_consistency(
            r_intent["longitudinal"], r_intent["lateral"], pred_lon, pred_lat
        )
        reasoning_vs_gt_decision = decision_pair_consistency(
            r_intent["longitudinal"], r_intent["lateral"], gt_lon, gt_lat
        )

        gt_dec_vs_gt_traj = decision_vs_motion(gt_lon, gt_lat, gt_motion)
        pred_dec_vs_direct = decision_vs_motion(pred_lon, pred_lat, direct_motion)
        pred_dec_vs_reasoning = decision_vs_motion(pred_lon, pred_lat, reasoning_motion)
        pred_dec_vs_vlm = decision_vs_motion(pred_lon, pred_lat, vlm_motion) if vlm_motion is not None else {}
        reasoning_vs_direct = decision_vs_motion(
            r_intent["longitudinal"], r_intent["lateral"], direct_motion
        )
        reasoning_vs_reasoning = decision_vs_motion(
            r_intent["longitudinal"], r_intent["lateral"], reasoning_motion
        )
        reasoning_vs_vlm = decision_vs_motion(
            r_intent["longitudinal"], r_intent["lateral"], vlm_motion
        ) if vlm_motion is not None else {}

        f1 = token_f1(reasoning_text, row["gt_reasoning"]) if row["gt_reasoning"] else float("nan")
        rouge = rouge_l_f1(reasoning_text, row["gt_reasoning"]) if row["gt_reasoning"] else float("nan")
        rq = float(np.nanmean([f1, rouge])) if np.isfinite([f1, rouge]).any() else float("nan")

        vlm_vs_direct = trajectory_pair_distance(vlm_xy, dtraj) if vlm_xy is not None else {"mean_xy_m": float("nan"), "final_xy_m": float("nan")}
        vlm_vs_reasoning = trajectory_pair_distance(vlm_xy, rtraj) if vlm_xy is not None else {"mean_xy_m": float("nan"), "final_xy_m": float("nan")}

        gt_xy = gt[:, :2]
        gt_path_len = float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], gt_xy]), axis=0), axis=-1).sum())
        gt_final_lat = float(abs(gt[-1, 1]))
        gt_heading_mag = float(abs(gt[-1, 2]))

        gain = float(d["ade_m"] - r["ade_m"])
        d_attn = dict(d.get("attention_summary", {}))
        r_attn = dict(r.get("attention_summary", {}))

        sample = {
            "index": index,
            "id": sid,
            "clip": str(row.get("clip", "")),
            "scenario_type": str(row.get("scenario_type", "")),
            "mission_command": row["command"],
            "speed_mps": float(row["speed_mps"]),
            "acceleration_mps2": float(row["acceleration_mps2"]),
            "heading_rad": float(row["heading_rad"]),
            "scene_description": row["scene_description"],
            "gt_reasoning": row["gt_reasoning"],
            "generated_reasoning": reasoning_text,
            "reasoning_token_f1": f1,
            "reasoning_rouge_l_f1": rouge,
            "reasoning_quality_mean": rq,
            "reasoning_trace_tokens": reasoning_tokens,
            "reasoning_intent": r_intent,
            "trajectory_text": trajectory_text,
            "vlm_trajectory_parse_ok": vlm_xy is not None,
            "vlm_trajectory_tokens": trajectory_tokens,
            "vlm_trajectory": vlm_xy.tolist() if vlm_xy is not None else None,
            "vlm_trajectory_ade_m": vlm_metrics["ade_m"],
            "vlm_trajectory_fde_m": vlm_metrics["fde_m"],
            "decision_text": str(dec_rec.get("decision_text", "")),
            "decision_parse_ok": bool(dec_rec.get("decision_parse_ok", False)),
            "pred_longitudinal": pred_lon,
            "pred_lateral": pred_lat,
            "gt_longitudinal": gt_lon,
            "gt_lateral": gt_lat,
            "decision_longitudinal_correct": decision_lon_correct if pred_lon and gt_lon else None,
            "decision_lateral_correct": decision_lat_correct if pred_lat and gt_lat else None,
            "decision_joint_correct": decision_joint_correct if pred_lon and pred_lat and gt_lon and gt_lat else None,
            "decision_joint_correct_float": bool_float(decision_joint_correct) if pred_lon and pred_lat and gt_lon and gt_lat else float("nan"),
            "reasoning_vs_decision": reasoning_vs_decision,
            "reasoning_vs_gt_decision": reasoning_vs_gt_decision,
            "reasoning_decision_consistency_score": finite_float(reasoning_vs_decision.get("score")),
            "gt_motion": gt_motion,
            "vlm_motion": vlm_motion,
            "direct_ae_motion": direct_motion,
            "reasoning_ae_motion": reasoning_motion,
            "gt_decision_vs_gt_trajectory": gt_dec_vs_gt_traj,
            "pred_decision_vs_vlm_trajectory": pred_dec_vs_vlm,
            "pred_decision_vs_direct_ae_trajectory": pred_dec_vs_direct,
            "pred_decision_vs_reasoning_ae_trajectory": pred_dec_vs_reasoning,
            "reasoning_vs_vlm_trajectory": reasoning_vs_vlm,
            "reasoning_vs_direct_ae_trajectory": reasoning_vs_direct,
            "reasoning_vs_reasoning_ae_trajectory": reasoning_vs_reasoning,
            "decision_vlm_alignment_score": finite_float(pred_dec_vs_vlm.get("signed_score")),
            "decision_direct_ae_alignment_score": finite_float(pred_dec_vs_direct.get("signed_score")),
            "decision_reasoning_ae_alignment_score": finite_float(pred_dec_vs_reasoning.get("signed_score")),
            "reasoning_vlm_alignment_score": finite_float(reasoning_vs_vlm.get("signed_score")),
            "reasoning_direct_ae_alignment_score": finite_float(reasoning_vs_direct.get("signed_score")),
            "reasoning_reasoning_ae_alignment_score": finite_float(reasoning_vs_reasoning.get("signed_score")),
            "direct_attention_diagnostics": d_attn,
            "reasoning_attention_diagnostics": r_attn,
            "reasoning_attention_mass": finite_float(r_attn.get("reasoning_attention_mass")),
            "reasoning_attention_share_within_vlm": finite_float(r_attn.get("reasoning_attention_share_within_vlm")),
            "reasoning_attention_enrichment": finite_float(r_attn.get("reasoning_attention_enrichment")),
            "reasoning_attention_entropy_nats": finite_float(r_attn.get("reasoning_attention_entropy_nats")),
            "reasoning_attention_entropy_normalized": finite_float(r_attn.get("reasoning_attention_entropy_normalized")),
            "reasoning_effective_token_fraction": finite_float(r_attn.get("reasoning_effective_token_fraction")),
            "top1_reasoning_token_share": finite_float(r_attn.get("top1_reasoning_token_share")),
            "top1_reasoning_token_share_within_vlm": finite_float(r_attn.get("top1_reasoning_token_share_within_vlm")),
            "reasoning_vlm_attention_mass": finite_float(r_attn.get("vlm_attention_mass")),
            "direct_vlm_attention_mass": finite_float(d_attn.get("vlm_attention_mass")),
            "reasoning_attended_context_uniform_mean_v_cosine": finite_float(
                r_attn.get("attended_context_uniform_mean_v_cosine")
            ),
            "direct_attended_context_uniform_mean_v_cosine": finite_float(
                d_attn.get("attended_context_uniform_mean_v_cosine")
            ),
            "attention_context_uniform_cosine_delta": (
                finite_float(r_attn.get("attended_context_uniform_mean_v_cosine"))
                - finite_float(d_attn.get("attended_context_uniform_mean_v_cosine"))
            ),
            "vlm_attention_mass_delta_reasoning_minus_direct": (
                finite_float(r_attn.get("vlm_attention_mass"))
                - finite_float(d_attn.get("vlm_attention_mass"))
            ),
            "direct_ae_ade_m": float(d["ade_m"]),
            "direct_ae_fde_m": float(d["fde_m"]),
            "direct_ae_heading_mae_rad": float(d["heading_mae_rad"]),
            "reasoning_ae_ade_m": float(r["ade_m"]),
            "reasoning_ae_fde_m": float(r["fde_m"]),
            "reasoning_ae_heading_mae_rad": float(r["heading_mae_rad"]),
            "reasoning_ae_gain_ade_m": gain,
            "reasoning_ae_delta_ade_m": -gain,
            "reasoning_ae_win": bool(gain > 0.0),
            "direct_ae_trajectory": d["trajectory"],
            "reasoning_ae_trajectory": r["trajectory"],
            "gt_trajectory": row["trajectory"],
            "vlm_vs_direct_ae_mean_xy_m": vlm_vs_direct["mean_xy_m"],
            "vlm_vs_reasoning_ae_mean_xy_m": vlm_vs_reasoning["mean_xy_m"],
            "vlm_vs_direct_ae_final_xy_m": vlm_vs_direct["final_xy_m"],
            "vlm_vs_reasoning_ae_final_xy_m": vlm_vs_reasoning["final_xy_m"],
            "gt_path_length_m": gt_path_len,
            "gt_final_abs_lateral_m": gt_final_lat,
            "gt_final_abs_heading_rad": gt_heading_mag,
        }
        out.append(sample)
        del cache
    return out


# =============================================================================
# SUMMARY / COUNTEREXAMPLES
# =============================================================================

def mean_bool(samples: Sequence[Dict[str, Any]], getter) -> float:
    vals = []
    for s in samples:
        v = getter(s)
        if v is True:
            vals.append(1.0)
        elif v is False:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else float("nan")


def mean_finite(samples: Sequence[Dict[str, Any]], key: str) -> float:
    vals = np.asarray([finite_float(s.get(key)) for s in samples], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def percentile_finite(samples: Sequence[Dict[str, Any]], key: str, q: float) -> float:
    vals = np.asarray([finite_float(s.get(key)) for s in samples], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def build_stratification(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rq_med = percentile_finite(samples, "reasoning_quality_mean", 50)
    va_med = percentile_finite(samples, "vlm_trajectory_ade_m", 50)
    cells = {}
    for rq_name, rq_pred in (
        ("reasoning_low", lambda s: finite_float(s.get("reasoning_quality_mean")) < rq_med),
        ("reasoning_high", lambda s: finite_float(s.get("reasoning_quality_mean")) >= rq_med),
    ):
        for vt_name, vt_pred in (
            ("vlm_traj_good", lambda s: finite_float(s.get("vlm_trajectory_ade_m")) < va_med),
            ("vlm_traj_bad", lambda s: finite_float(s.get("vlm_trajectory_ade_m")) >= va_med),
        ):
            chosen = [s for s in samples if rq_pred(s) and vt_pred(s)
                      and math.isfinite(finite_float(s.get("reasoning_quality_mean")))
                      and math.isfinite(finite_float(s.get("vlm_trajectory_ade_m")))]
            cells[f"{rq_name}__{vt_name}"] = {
                "n": len(chosen),
                "reasoning_quality_mean": mean_finite(chosen, "reasoning_quality_mean"),
                "vlm_trajectory_ade_m": mean_finite(chosen, "vlm_trajectory_ade_m"),
                "direct_ae_ade_m": mean_finite(chosen, "direct_ae_ade_m"),
                "reasoning_ae_ade_m": mean_finite(chosen, "reasoning_ae_ade_m"),
                "reasoning_ae_gain_ade_m": mean_finite(chosen, "reasoning_ae_gain_ade_m"),
                "reasoning_ae_win_rate": mean_bool(chosen, lambda s: s.get("reasoning_ae_win")),
            }
    return {
        "reasoning_quality_median": rq_med,
        "vlm_trajectory_ade_median": va_med,
        "cells": cells,
    }


def build_group_summaries(samples: Sequence[Dict[str, Any]], bootstrap: int, seed: int) -> Dict[str, Any]:
    result = {}
    groups = {
        "decision_joint_correct": lambda s: s.get("decision_joint_correct") is True,
        "decision_joint_wrong": lambda s: s.get("decision_joint_correct") is False,
        "reasoning_decision_consistent": lambda s: finite_float(s.get("reasoning_decision_consistency_score")) >= 0.999,
        "reasoning_decision_inconsistent": lambda s: math.isfinite(finite_float(s.get("reasoning_decision_consistency_score"))) and finite_float(s.get("reasoning_decision_consistency_score")) < 0.999,
        "decision_vlm_traj_consistent": lambda s: finite_float(s.get("decision_vlm_alignment_score")) >= 0.999,
        "decision_vlm_traj_inconsistent": lambda s: math.isfinite(finite_float(s.get("decision_vlm_alignment_score"))) and finite_float(s.get("decision_vlm_alignment_score")) < 0.999,
        "decision_reasoning_ae_consistent": lambda s: finite_float(s.get("decision_reasoning_ae_alignment_score")) >= 0.999,
        "decision_reasoning_ae_inconsistent": lambda s: math.isfinite(finite_float(s.get("decision_reasoning_ae_alignment_score"))) and finite_float(s.get("decision_reasoning_ae_alignment_score")) < 0.999,
    }
    for i, (name, pred) in enumerate(groups.items()):
        chosen = [s for s in samples if pred(s)]
        result[name] = {
            "n": len(chosen),
            "reasoning_ae_gain_ade_m": bootstrap_mean_ci(
                [s["reasoning_ae_gain_ade_m"] for s in chosen], bootstrap, seed + i
            ),
            "reasoning_ae_ade_m": bootstrap_mean_ci(
                [s["reasoning_ae_ade_m"] for s in chosen], bootstrap, seed + 100 + i
            ),
            "vlm_trajectory_ade_m": bootstrap_mean_ci(
                [s["vlm_trajectory_ade_m"] for s in chosen], bootstrap, seed + 200 + i
            ),
            "reasoning_quality_mean": bootstrap_mean_ci(
                [s["reasoning_quality_mean"] for s in chosen], bootstrap, seed + 300 + i
            ),
        }
    return result


def build_counterexamples(samples: Sequence[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    rq25 = percentile_finite(samples, "reasoning_quality_mean", 25)
    rq75 = percentile_finite(samples, "reasoning_quality_mean", 75)
    va25 = percentile_finite(samples, "vlm_trajectory_ade_m", 25)
    va75 = percentile_finite(samples, "vlm_trajectory_ade_m", 75)
    ra25 = percentile_finite(samples, "reasoning_ae_ade_m", 25)
    ra75 = percentile_finite(samples, "reasoning_ae_ade_m", 75)

    compact_keys = [
        "index", "id", "clip", "mission_command",
        "reasoning_quality_mean", "reasoning_token_f1", "reasoning_rouge_l_f1",
        "vlm_trajectory_ade_m", "direct_ae_ade_m", "reasoning_ae_ade_m",
        "reasoning_ae_gain_ade_m", "pred_longitudinal", "pred_lateral",
        "gt_longitudinal", "gt_lateral", "reasoning_intent",
        "reasoning_decision_consistency_score", "decision_vlm_alignment_score",
        "decision_reasoning_ae_alignment_score", "generated_reasoning",
        "gt_reasoning", "trajectory_text",
    ]

    def compact(s):
        return {k: s.get(k) for k in compact_keys}

    categories = {}
    specs = {
        "high_reasoning_but_ae_degraded": (
            lambda s: finite_float(s.get("reasoning_quality_mean")) >= rq75 and finite_float(s.get("reasoning_ae_gain_ade_m")) < -1.0,
            lambda s: finite_float(s.get("reasoning_ae_gain_ade_m")),
        ),
        "low_reasoning_but_ae_improved": (
            lambda s: finite_float(s.get("reasoning_quality_mean")) <= rq25 and finite_float(s.get("reasoning_ae_gain_ade_m")) > 1.0,
            lambda s: -finite_float(s.get("reasoning_ae_gain_ade_m")),
        ),
        "high_reasoning_but_vlm_trajectory_bad": (
            lambda s: finite_float(s.get("reasoning_quality_mean")) >= rq75 and finite_float(s.get("vlm_trajectory_ade_m")) >= va75,
            lambda s: -finite_float(s.get("vlm_trajectory_ade_m")),
        ),
        "low_reasoning_but_vlm_trajectory_good": (
            lambda s: finite_float(s.get("reasoning_quality_mean")) <= rq25 and finite_float(s.get("vlm_trajectory_ade_m")) <= va25,
            lambda s: finite_float(s.get("vlm_trajectory_ade_m")),
        ),
        "vlm_trajectory_good_but_reasoning_ae_bad": (
            lambda s: finite_float(s.get("vlm_trajectory_ade_m")) <= va25 and finite_float(s.get("reasoning_ae_ade_m")) >= ra75,
            lambda s: -finite_float(s.get("reasoning_ae_ade_m")),
        ),
        "vlm_trajectory_bad_but_reasoning_ae_good": (
            lambda s: finite_float(s.get("vlm_trajectory_ade_m")) >= va75 and finite_float(s.get("reasoning_ae_ade_m")) <= ra25,
            lambda s: finite_float(s.get("reasoning_ae_ade_m")),
        ),
        "reasoning_vs_explicit_decision_conflict": (
            lambda s: math.isfinite(finite_float(s.get("reasoning_decision_consistency_score"))) and finite_float(s.get("reasoning_decision_consistency_score")) < 0.5,
            lambda s: finite_float(s.get("reasoning_decision_consistency_score")),
        ),
        "explicit_decision_vs_vlm_trajectory_conflict": (
            lambda s: math.isfinite(finite_float(s.get("decision_vlm_alignment_score"))) and finite_float(s.get("decision_vlm_alignment_score")) < 0.5,
            lambda s: finite_float(s.get("decision_vlm_alignment_score")),
        ),
        "explicit_decision_vs_reasoning_ae_conflict": (
            lambda s: math.isfinite(finite_float(s.get("decision_reasoning_ae_alignment_score"))) and finite_float(s.get("decision_reasoning_ae_alignment_score")) < 0.5,
            lambda s: finite_float(s.get("decision_reasoning_ae_alignment_score")),
        ),
    }
    for name, (pred, sort_key) in specs.items():
        selected = [s for s in samples if pred(s)]
        selected = sorted(selected, key=sort_key)[:topk]
        categories[name] = [compact(s) for s in selected]

    return {
        "thresholds": {
            "reasoning_q25": rq25,
            "reasoning_q75": rq75,
            "vlm_ade_q25": va25,
            "vlm_ade_q75": va75,
            "reasoning_ae_ade_q25": ra25,
            "reasoning_ae_ade_q75": ra75,
        },
        "categories": categories,
    }


def build_summary(samples: Sequence[Dict[str, Any]], bootstrap: int, seed: int) -> Dict[str, Any]:
    gt_calibration = {
        "longitudinal_sign_match": mean_bool(samples, lambda s: s["gt_decision_vs_gt_trajectory"].get("longitudinal_sign_match")),
        "lateral_match": mean_bool(samples, lambda s: s["gt_decision_vs_gt_trajectory"].get("lateral_match")),
        "signed_joint_match": mean_bool(samples, lambda s: s["gt_decision_vs_gt_trajectory"].get("signed_joint_match")),
        "signed_score_mean": float(np.nanmean([
            finite_float(s["gt_decision_vs_gt_trajectory"].get("signed_score")) for s in samples
        ])),
    }

    correlations = {
        # Core question 1
        "reasoning_quality_vs_vlm_trajectory_ade": corr_pair(
            [s["reasoning_quality_mean"] for s in samples],
            [s["vlm_trajectory_ade_m"] for s in samples],
        ),
        "reasoning_quality_vs_vlm_trajectory_fde": corr_pair(
            [s["reasoning_quality_mean"] for s in samples],
            [s["vlm_trajectory_fde_m"] for s in samples],
        ),
        # Core question 2
        "vlm_trajectory_ade_vs_direct_ae_ade": corr_pair(
            [s["vlm_trajectory_ade_m"] for s in samples],
            [s["direct_ae_ade_m"] for s in samples],
        ),
        "vlm_trajectory_ade_vs_reasoning_ae_ade": corr_pair(
            [s["vlm_trajectory_ade_m"] for s in samples],
            [s["reasoning_ae_ade_m"] for s in samples],
        ),
        "vlm_trajectory_ade_vs_reasoning_ae_gain": corr_pair(
            [s["vlm_trajectory_ade_m"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        # Does reasoning semantic quality explain the gain?
        "reasoning_quality_vs_reasoning_ae_gain": corr_pair(
            [s["reasoning_quality_mean"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        "reasoning_quality_vs_reasoning_ae_ade": corr_pair(
            [s["reasoning_quality_mean"] for s in samples],
            [s["reasoning_ae_ade_m"] for s in samples],
        ),
        "reasoning_trace_tokens_vs_reasoning_ae_gain": corr_pair(
            [s["reasoning_trace_tokens"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        # Decision/consistency probes
        "decision_joint_correct_vs_vlm_trajectory_ade": corr_pair(
            [s["decision_joint_correct_float"] for s in samples],
            [s["vlm_trajectory_ade_m"] for s in samples],
        ),
        "decision_joint_correct_vs_reasoning_ae_gain": corr_pair(
            [s["decision_joint_correct_float"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        "reasoning_decision_consistency_vs_reasoning_ae_gain": corr_pair(
            [s["reasoning_decision_consistency_score"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        "decision_vlm_alignment_vs_reasoning_ae_gain": corr_pair(
            [s["decision_vlm_alignment_score"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
        "decision_reasoning_ae_alignment_vs_reasoning_ae_ade": corr_pair(
            [s["decision_reasoning_ae_alignment_score"] for s in samples],
            [s["reasoning_ae_ade_m"] for s in samples],
        ),
        # Common latent/planning similarity
        "vlm_direct_vs_reasoning_ae_distance_vs_reasoning_ae_ade": corr_pair(
            [s["vlm_vs_reasoning_ae_mean_xy_m"] for s in samples],
            [s["reasoning_ae_ade_m"] for s in samples],
        ),
        "direct_ae_ade_vs_reasoning_gain_room_to_improve": corr_pair(
            [s["direct_ae_ade_m"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
        ),
    }

    partial = {
        "reasoning_quality_vs_gain_control_vlm_ade": partial_corr(
            [s["reasoning_quality_mean"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
            [[s["vlm_trajectory_ade_m"] for s in samples]],
        ),
        "reasoning_quality_vs_gain_control_vlm_ade_and_trace_length": partial_corr(
            [s["reasoning_quality_mean"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
            [
                [s["vlm_trajectory_ade_m"] for s in samples],
                [s["reasoning_trace_tokens"] for s in samples],
            ],
        ),
        "vlm_ade_vs_gain_control_reasoning_quality": partial_corr(
            [s["vlm_trajectory_ade_m"] for s in samples],
            [s["reasoning_ae_gain_ade_m"] for s in samples],
            [[s["reasoning_quality_mean"] for s in samples]],
        ),
        "reasoning_quality_vs_reasoning_ae_ade_control_vlm_ade": partial_corr(
            [s["reasoning_quality_mean"] for s in samples],
            [s["reasoning_ae_ade_m"] for s in samples],
            [[s["vlm_trajectory_ade_m"] for s in samples]],
        ),
    }

    # Standardized multivariable association model for gain.
    predictors = [
        "reasoning_quality_mean",
        "vlm_trajectory_ade_m",
        "decision_joint_correct_float",
        "reasoning_decision_consistency_score",
        "decision_vlm_alignment_score",
        "reasoning_trace_tokens",
        "speed_mps",
        "gt_path_length_m",
        "gt_final_abs_lateral_m",
        "gt_final_abs_heading_rad",
    ]
    regression = standardized_regression(samples, "reasoning_ae_gain_ade_m", predictors)

    summary = {
        "samples": len(samples),
        "parse_rates": {
            "vlm_trajectory": mean_bool(samples, lambda s: s.get("vlm_trajectory_parse_ok")),
            "decision": mean_bool(samples, lambda s: s.get("decision_parse_ok")),
        },
        "overall": {
            "reasoning_token_f1_mean": mean_finite(samples, "reasoning_token_f1"),
            "reasoning_rouge_l_f1_mean": mean_finite(samples, "reasoning_rouge_l_f1"),
            "reasoning_quality_mean": mean_finite(samples, "reasoning_quality_mean"),
            "vlm_trajectory_ade_m": mean_finite(samples, "vlm_trajectory_ade_m"),
            "vlm_trajectory_fde_m": mean_finite(samples, "vlm_trajectory_fde_m"),
            "direct_ae_ade_m": mean_finite(samples, "direct_ae_ade_m"),
            "direct_ae_fde_m": mean_finite(samples, "direct_ae_fde_m"),
            "reasoning_ae_ade_m": mean_finite(samples, "reasoning_ae_ade_m"),
            "reasoning_ae_fde_m": mean_finite(samples, "reasoning_ae_fde_m"),
            "reasoning_ae_gain_ade_m": mean_finite(samples, "reasoning_ae_gain_ade_m"),
            "reasoning_ae_win_rate": mean_bool(samples, lambda s: s.get("reasoning_ae_win")),
            "decision_longitudinal_accuracy": mean_bool(samples, lambda s: s.get("decision_longitudinal_correct")),
            "decision_lateral_accuracy": mean_bool(samples, lambda s: s.get("decision_lateral_correct")),
            "decision_joint_accuracy": mean_bool(samples, lambda s: s.get("decision_joint_correct")),
            "reasoning_decision_consistency_mean": mean_finite(samples, "reasoning_decision_consistency_score"),
            "decision_vlm_trajectory_alignment_mean": mean_finite(samples, "decision_vlm_alignment_score"),
            "decision_direct_ae_alignment_mean": mean_finite(samples, "decision_direct_ae_alignment_score"),
            "decision_reasoning_ae_alignment_mean": mean_finite(samples, "decision_reasoning_ae_alignment_score"),
        },
        "gt_decision_trajectory_calibration": gt_calibration,
        "correlations": correlations,
        "partial_correlations": partial,
        "standardized_gain_regression": regression,
        "stratification_reasoning_x_vlm_trajectory": build_stratification(samples),
        "group_summaries": build_group_summaries(samples, bootstrap, seed),
    }
    return summary


# =============================================================================
# REPORT
# =============================================================================

def fmt(x: Any, digits: int = 4) -> str:
    v = finite_float(x)
    return "N/A" if not math.isfinite(v) else f"{v:.{digits}f}"


def pct(x: Any) -> str:
    v = finite_float(x)
    return "N/A" if not math.isfinite(v) else f"{100.0*v:.2f}%"


def corr_line(name: str, obj: Dict[str, Any]) -> str:
    return f"| {name} | {obj.get('n', 0)} | {fmt(obj.get('pearson'))} | {fmt(obj.get('spearman'))} |"


def strength(value: float) -> str:
    if not math.isfinite(value):
        return "unavailable"
    a = abs(value)
    if a < 0.10:
        return "very weak"
    if a < 0.20:
        return "weak"
    if a < 0.35:
        return "moderate"
    return "strong"


def auto_interpretation(summary: Dict[str, Any]) -> List[str]:
    c = summary["correlations"]
    p = summary["partial_correlations"]
    cal = summary["gt_decision_trajectory_calibration"]
    lines = []

    rq_gain = finite_float(c["reasoning_quality_vs_reasoning_ae_gain"].get("spearman"))
    rq_vlm = finite_float(c["reasoning_quality_vs_vlm_trajectory_ade"].get("spearman"))
    vlm_ae = finite_float(c["vlm_trajectory_ade_vs_reasoning_ae_ade"].get("spearman"))
    rq_gain_partial = finite_float(p["reasoning_quality_vs_gain_control_vlm_ade_and_trace_length"].get("spearman"))
    token_gain = finite_float(c["reasoning_trace_tokens_vs_reasoning_ae_gain"].get("spearman"))

    lines.append(
        f"Reasoning lexical quality -> AE gain is {strength(rq_gain)} (Spearman={fmt(rq_gain)}). "
        "A near-zero value argues against a simple 'better text reasoning directly causes better planner' explanation."
    )
    lines.append(
        f"Reasoning quality -> VLM direct trajectory ADE is {strength(rq_vlm)} (Spearman={fmt(rq_vlm)}). "
        "Negative correlation would mean better reasoning accompanies a better VLM planner output."
    )
    lines.append(
        f"VLM direct trajectory ADE -> reasoning-AE ADE is {strength(vlm_ae)} (Spearman={fmt(vlm_ae)}). "
        "A positive correlation suggests a shared scene/planning-difficulty or common VLM-representation factor."
    )
    lines.append(
        f"After controlling VLM trajectory ADE and reasoning trace length, reasoning quality -> AE gain is "
        f"{strength(rq_gain_partial)} (partial Spearman={fmt(rq_gain_partial)})."
    )
    lines.append(
        f"Reasoning trace token count -> AE gain is {strength(token_gain)} (Spearman={fmt(token_gain)}). "
        "If this is stronger than reasoning quality, token/memory quantity is a plausible confound."
    )

    joint_cal = finite_float(cal.get("signed_joint_match"))
    if math.isfinite(joint_cal) and joint_cal < 0.65:
        lines.append(
            f"WARNING: GT decision vs GT trajectory direction calibration is only {pct(joint_cal)}. "
            "Therefore direction-consistency metrics should be treated as auxiliary, not definitive."
        )
    else:
        lines.append(
            f"GT decision vs GT trajectory direction calibration is {pct(joint_cal)}; "
            "prediction-side direction consistency is interpretable as an auxiliary diagnostic."
        )
    return lines


def write_report(
    path: Path,
    summary: Dict[str, Any],
    counterexamples: Dict[str, Any],
    attention_diagnostics: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    o = summary["overall"]
    cal = summary["gt_decision_trajectory_calibration"]
    corr = summary["correlations"]
    partial = summary["partial_correlations"]
    strat = summary["stratification_reasoning_x_vlm_trajectory"]
    reg = summary["standardized_gain_regression"]
    attn_r = attention_diagnostics["reasoning"]["overall"]
    attn_d = attention_diagnostics["direct"]["overall"]
    attn_corr = attention_diagnostics[
        "attention_metric_vs_ae_gain_correlation"
    ]["reasoning_attention_overall"]
    collapse = attention_diagnostics["cross_scene_collapse"]
    collapse_direct = collapse["direct"]["attended_context"]["overall"]
    collapse_reasoning = collapse["reasoning"]["attended_context"]["overall"]
    collapse_reasoning_only = collapse["reasoning"]["reasoning_context"]["overall"]

    lines = [
        "# Reasoning–Decision–Trajectory–Action Expert Causal Diagnostic",
        "",
        "## Experiment",
        "",
        f"- Samples: {summary['samples']}",
        f"- VLM: `{args.vlm}`",
        f"- Dataset: `{args.jsonl}`",
        f"- Direct AE: `{args.direct_ckpt}`",
        f"- Reasoning AE: `{args.reasoning_ckpt}`",
        f"- Trace cache used: `{args._trace_cache_used}`",
        "- VLM decoding: greedy (`do_sample=False`)",
        "- Flow sampling: identical per-sample Gaussian x0 for Direct/Reasoning AE",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Reasoning Token F1 | {fmt(o['reasoning_token_f1_mean'])} |",
        f"| Reasoning ROUGE-L | {fmt(o['reasoning_rouge_l_f1_mean'])} |",
        f"| Reasoning combined quality | {fmt(o['reasoning_quality_mean'])} |",
        f"| VLM direct trajectory ADE | {fmt(o['vlm_trajectory_ade_m'])} m |",
        f"| VLM direct trajectory FDE | {fmt(o['vlm_trajectory_fde_m'])} m |",
        f"| Direct AE ADE | {fmt(o['direct_ae_ade_m'])} m |",
        f"| Reasoning AE ADE | {fmt(o['reasoning_ae_ade_m'])} m |",
        f"| Mean Reasoning AE gain (Direct ADE - Reasoning ADE) | {fmt(o['reasoning_ae_gain_ade_m'])} m |",
        f"| Reasoning AE win rate | {pct(o['reasoning_ae_win_rate'])} |",
        f"| VLM decision longitudinal accuracy | {pct(o['decision_longitudinal_accuracy'])} |",
        f"| VLM decision lateral accuracy | {pct(o['decision_lateral_accuracy'])} |",
        f"| VLM decision joint accuracy | {pct(o['decision_joint_accuracy'])} |",
        f"| Reasoning ↔ explicit Decision consistency | {fmt(o['reasoning_decision_consistency_mean'])} |",
        f"| Decision ↔ VLM trajectory direction alignment | {fmt(o['decision_vlm_trajectory_alignment_mean'])} |",
        f"| Decision ↔ Direct AE direction alignment | {fmt(o['decision_direct_ae_alignment_mean'])} |",
        f"| Decision ↔ Reasoning AE direction alignment | {fmt(o['decision_reasoning_ae_alignment_mean'])} |",
        "",
        "## Attention diagnostics",
        "",
        "Attention is measured inside each RAW-KV AE fusion layer before dropout, separately for every Euler step.",
        "",
        "| Metric | Direct AE | Reasoning AE |",
        "|---|---:|---:|",
        f"| VLM attention mass | {fmt(attn_d.get('vlm_attention_mass'))} | {fmt(attn_r.get('vlm_attention_mass'))} |",
        f"| Reasoning attention mass | N/A | {fmt(attn_r.get('reasoning_attention_mass'))} |",
        f"| Reasoning share within VLM | N/A | {fmt(attn_r.get('reasoning_attention_share_within_vlm'))} |",
        f"| Reasoning attention enrichment | N/A | {fmt(attn_r.get('reasoning_attention_enrichment'))} |",
        f"| Reasoning attention entropy (nats) | N/A | {fmt(attn_r.get('reasoning_attention_entropy_nats'))} |",
        f"| Reasoning effective token fraction | N/A | {fmt(attn_r.get('reasoning_effective_token_fraction'))} |",
        f"| Top-1 reasoning token share | N/A | {fmt(attn_r.get('top1_reasoning_token_share'))} |",
        f"| Attended context ↔ uniform-mean V cosine | {fmt(attn_d.get('attended_context_uniform_mean_v_cosine'))} | {fmt(attn_r.get('attended_context_uniform_mean_v_cosine'))} |",
        "",
        "### Cross-scene cosine collapse",
        "",
        "| Vector | Mean off-diagonal cosine | Centroid norm ratio |",
        "|---|---:|---:|",
        f"| Direct attended context | {fmt(collapse_direct.get('cross_scene_mean_cosine'))} | {fmt(collapse_direct.get('centroid_norm_ratio'))} |",
        f"| Reasoning attended context | {fmt(collapse_reasoning.get('cross_scene_mean_cosine'))} | {fmt(collapse_reasoning.get('centroid_norm_ratio'))} |",
        f"| Reasoning-only context | {fmt(collapse_reasoning_only.get('cross_scene_mean_cosine'))} | {fmt(collapse_reasoning_only.get('centroid_norm_ratio'))} |",
        "",
        "Higher cross-scene cosine means different scenes are mapped toward a more similar attended V-context direction.",
        "",
        "### Attention metric ↔ Reasoning AE gain",
        "",
        "`gain = Direct AE ADE - Reasoning AE ADE`; positive gain means the reasoning-conditioned AE is better.",
        "",
        "| Attention metric | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
        corr_line("Reasoning attention mass vs AE gain", attn_corr["reasoning_attention_mass"]),
        corr_line("Reasoning share within VLM vs AE gain", attn_corr["reasoning_attention_share_within_vlm"]),
        corr_line("Reasoning enrichment vs AE gain", attn_corr["reasoning_attention_enrichment"]),
        corr_line("Reasoning entropy vs AE gain", attn_corr["reasoning_attention_entropy_nats"]),
        corr_line("Reasoning effective token fraction vs AE gain", attn_corr["reasoning_effective_token_fraction"]),
        corr_line("Top-1 reasoning token share vs AE gain", attn_corr["top1_reasoning_token_share"]),
        corr_line("Context↔uniform-V cosine vs AE gain", attn_corr["attended_context_uniform_mean_v_cosine"]),
        "",
        "Full layer-wise and Euler-step-wise attention diagnostics are saved to `attention_diagnostics.json`.",
        "Raw diagnostic vectors are saved to `attention_vectors.pt`.",
        "",
        "## Direction heuristic calibration",
        "",
        "Before interpreting predicted decision/trajectory consistency, the same trajectory-direction heuristic is tested on GT.",
        "",
        "| GT calibration | Match |",
        "|---|---:|",
        f"| Longitudinal sign | {pct(cal['longitudinal_sign_match'])} |",
        f"| Lateral direction | {pct(cal['lateral_match'])} |",
        f"| Joint | {pct(cal['signed_joint_match'])} |",
        f"| Mean axis score | {fmt(cal['signed_score_mean'])} |",
        "",
        "## Core correlations",
        "",
        "Spearman is the primary robust monotonic statistic. For ADE, lower is better; for `reasoning_ae_gain_ade_m`, higher is better.",
        "",
        "| Relationship | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]
    order = [
        ("Reasoning quality vs VLM trajectory ADE", "reasoning_quality_vs_vlm_trajectory_ade"),
        ("Reasoning quality vs VLM trajectory FDE", "reasoning_quality_vs_vlm_trajectory_fde"),
        ("VLM trajectory ADE vs Direct AE ADE", "vlm_trajectory_ade_vs_direct_ae_ade"),
        ("VLM trajectory ADE vs Reasoning AE ADE", "vlm_trajectory_ade_vs_reasoning_ae_ade"),
        ("VLM trajectory ADE vs Reasoning AE gain", "vlm_trajectory_ade_vs_reasoning_ae_gain"),
        ("Reasoning quality vs Reasoning AE gain", "reasoning_quality_vs_reasoning_ae_gain"),
        ("Reasoning quality vs Reasoning AE ADE", "reasoning_quality_vs_reasoning_ae_ade"),
        ("Reasoning trace tokens vs Reasoning AE gain", "reasoning_trace_tokens_vs_reasoning_ae_gain"),
        ("Decision correctness vs VLM trajectory ADE", "decision_joint_correct_vs_vlm_trajectory_ade"),
        ("Decision correctness vs Reasoning AE gain", "decision_joint_correct_vs_reasoning_ae_gain"),
        ("Reasoning↔Decision consistency vs AE gain", "reasoning_decision_consistency_vs_reasoning_ae_gain"),
        ("Decision↔VLM trajectory alignment vs AE gain", "decision_vlm_alignment_vs_reasoning_ae_gain"),
        ("Direct AE ADE vs Reasoning gain (room-to-improve diagnostic)", "direct_ae_ade_vs_reasoning_gain_room_to_improve"),
    ]
    for label, key in order:
        lines.append(corr_line(label, corr[key]))

    lines += [
        "",
        "## Partial correlations",
        "",
        "| Relationship | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for label, key in [
        ("Reasoning quality vs gain | control VLM ADE", "reasoning_quality_vs_gain_control_vlm_ade"),
        ("Reasoning quality vs gain | control VLM ADE + trace length", "reasoning_quality_vs_gain_control_vlm_ade_and_trace_length"),
        ("VLM ADE vs gain | control reasoning quality", "vlm_ade_vs_gain_control_reasoning_quality"),
        ("Reasoning quality vs Reasoning AE ADE | control VLM ADE", "reasoning_quality_vs_reasoning_ae_ade_control_vlm_ade"),
    ]:
        lines.append(corr_line(label, partial[key]))

    lines += [
        "",
        "## 2×2 stratification: Reasoning quality × VLM direct trajectory quality",
        "",
        f"- Reasoning quality median: {fmt(strat['reasoning_quality_median'])}",
        f"- VLM trajectory ADE median: {fmt(strat['vlm_trajectory_ade_median'])} m",
        "",
        "| Cell | n | Reasoning Q | VLM ADE | Direct AE ADE | Reasoning AE ADE | AE gain | AE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cell in strat["cells"].items():
        lines.append(
            f"| {name} | {cell['n']} | {fmt(cell['reasoning_quality_mean'])} | "
            f"{fmt(cell['vlm_trajectory_ade_m'])} | {fmt(cell['direct_ae_ade_m'])} | "
            f"{fmt(cell['reasoning_ae_ade_m'])} | {fmt(cell['reasoning_ae_gain_ade_m'])} | "
            f"{pct(cell['reasoning_ae_win_rate'])} |"
        )

    lines += [
        "",
        "## Standardized multivariable association model for AE gain",
        "",
        f"- n = {reg.get('n', 0)}",
        f"- R² = {fmt(reg.get('r2'))}",
        "- Coefficients are standardized OLS associations, **not causal effects**.",
        "",
        "| Predictor | Standardized coefficient |",
        "|---|---:|",
    ]
    for name, value in reg.get("coefficients", {}).items():
        lines.append(f"| {name} | {fmt(value)} |")

    lines += [
        "",
        "## Automatic diagnostic reading",
        "",
    ]
    for text in auto_interpretation(summary):
        lines.append(f"- {text}")

    lines += [
        "",
        "## Counterexample counts",
        "",
        "These cases are written in detail to `counterexamples.json`.",
        "",
        "| Category | Count saved |",
        "|---|---:|",
    ]
    for name, vals in counterexamples["categories"].items():
        lines.append(f"| {name} | {len(vals)} |")

    lines += [
        "",
        "## Interpretation rule",
        "",
        "This script cannot establish causal identification by correlation alone. The useful diagnostic patterns are:",
        "",
        "1. **Reasoning quality strongly predicts AE gain, including after controls** → evidence consistent with semantic reasoning utility.",
        "2. **Reasoning quality is weak, but VLM direct trajectory quality strongly tracks AE quality** → common VLM scene/planning representation is the stronger explanation.",
        "3. **Reasoning/Decision directional agreement stratifies AE gain** → semantic action-direction alignment matters even if lexical F1/ROUGE does not.",
        "4. **Trace length predicts gain more strongly than semantic metrics** → added KV/token memory quantity is a plausible confound.",
        "5. **High-quality reasoning can still degrade AE and low-quality reasoning can improve it** → reasoning text quality alone is insufficient; inspect the saved counterexamples and explicit Decision/VLM trajectory consistency.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CSV / PLOTS
# =============================================================================

def flatten_for_csv(s: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "index", "id", "clip", "scenario_type", "mission_command", "speed_mps",
        "reasoning_token_f1", "reasoning_rouge_l_f1", "reasoning_quality_mean",
        "reasoning_trace_tokens", "vlm_trajectory_parse_ok", "vlm_trajectory_ade_m",
        "vlm_trajectory_fde_m", "decision_parse_ok", "pred_longitudinal", "pred_lateral",
        "gt_longitudinal", "gt_lateral", "decision_joint_correct",
        "reasoning_decision_consistency_score", "decision_vlm_alignment_score",
        "decision_direct_ae_alignment_score", "decision_reasoning_ae_alignment_score",
        "reasoning_attention_mass", "reasoning_attention_share_within_vlm",
        "reasoning_attention_enrichment", "reasoning_attention_entropy_nats",
        "reasoning_attention_entropy_normalized", "reasoning_effective_token_fraction",
        "top1_reasoning_token_share", "top1_reasoning_token_share_within_vlm",
        "reasoning_vlm_attention_mass", "direct_vlm_attention_mass",
        "reasoning_attended_context_uniform_mean_v_cosine",
        "direct_attended_context_uniform_mean_v_cosine",
        "attention_context_uniform_cosine_delta",
        "vlm_attention_mass_delta_reasoning_minus_direct",
        "direct_ae_ade_m", "direct_ae_fde_m", "reasoning_ae_ade_m", "reasoning_ae_fde_m",
        "reasoning_ae_gain_ade_m", "reasoning_ae_win", "vlm_vs_direct_ae_mean_xy_m",
        "vlm_vs_reasoning_ae_mean_xy_m", "generated_reasoning", "gt_reasoning",
        "decision_text", "trajectory_text",
    ]
    return {k: s.get(k) for k in keys}


def write_csv(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    rows = [flatten_for_csv(s) for s in samples]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def maybe_make_plots(plot_dir: Path, samples: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; plots skipped: {exc}")
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    specs = [
        ("reasoning_quality_mean", "vlm_trajectory_ade_m", "Reasoning quality", "VLM trajectory ADE (m)", "reasoning_quality_vs_vlm_ade.png"),
        ("vlm_trajectory_ade_m", "reasoning_ae_ade_m", "VLM trajectory ADE (m)", "Reasoning AE ADE (m)", "vlm_ade_vs_reasoning_ae_ade.png"),
        ("reasoning_quality_mean", "reasoning_ae_gain_ade_m", "Reasoning quality", "Reasoning AE gain (m)", "reasoning_quality_vs_ae_gain.png"),
        ("reasoning_decision_consistency_score", "reasoning_ae_gain_ade_m", "Reasoning ↔ Decision consistency", "Reasoning AE gain (m)", "reasoning_decision_consistency_vs_gain.png"),
    ]
    for xkey, ykey, xlabel, ylabel, filename in specs:
        x, y = clean_pair([s.get(xkey) for s in samples], [s.get(ykey) for s in samples])
        if len(x) < 3:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.65)
        if np.std(x) > 1e-12:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xx, coef[0] * xx + coef[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Spearman={spearman_corr(x, y):.3f}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = plot_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose Reasoning↔Decision↔VLM trajectory↔Action Expert relationships."
    )
    p.add_argument("--vlm", type=Path, default=VLM_PATH)
    p.add_argument("--jsonl", type=Path, default=TEST_JSONL)
    p.add_argument("--preferred-trace-cache", type=Path, default=PREFERRED_TRACE_CACHE_ROOT)
    p.add_argument("--work-cache-root", type=Path, default=WORK_CACHE_ROOT)
    p.add_argument("--direct-ckpt", type=Path, default=DIRECT_CKPT)
    p.add_argument("--reasoning-ckpt", type=Path, default=REASONING_CKPT)
    p.add_argument("--output", type=Path, default=RESULT_ROOT)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    p.add_argument("--reasoning-max-new-tokens", type=int, default=128)
    p.add_argument("--trajectory-max-new-tokens", type=int, default=128)
    p.add_argument("--decision-max-new-tokens", type=int, default=64)
    p.add_argument("--allow-truncated", action="store_true")
    p.add_argument("--max-prefix-diff", type=float, default=0.05)
    p.add_argument("--flow-noise-seed", type=int, default=FLOW_NOISE_SEED)
    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--counterexample-topk", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Debug only")
    p.add_argument("--force-regenerate-trace-cache", action="store_true")
    p.add_argument("--rebuild-decision-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "vlm", "jsonl", "preferred_trace_cache", "work_cache_root",
        "direct_ckpt", "reasoning_ckpt", "output",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    if not args.vlm.is_dir():
        raise FileNotFoundError(args.vlm)
    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    if not args.direct_ckpt.is_file():
        raise FileNotFoundError(args.direct_ckpt)
    if not args.reasoning_ckpt.is_file():
        raise FileNotFoundError(args.reasoning_ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 selected but GPU does not support BF16")

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {args.output}\nUse --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    args.work_cache_root.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(r) for r in read_jsonl(args.jsonl)]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError("No test rows")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate test IDs")

    # Hard evidence required for the requested semantic analysis.
    missing_reasoning = sum(not bool(r["gt_reasoning"]) for r in rows)
    missing_decision = sum(not bool(r["longitudinal"] and r["lateral"]) for r in rows)
    if missing_reasoning:
        print(f"[WARN] GT reasoning missing for {missing_reasoning}/{len(rows)} samples")
    if missing_decision:
        print(f"[WARN] GT decision missing for {missing_decision}/{len(rows)} samples")

    set_seed(SEED)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = resolve_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    direct_ckpt_obj = load_checkpoint(args.direct_ckpt)
    reasoning_ckpt_obj = load_checkpoint(args.reasoning_ckpt)
    validate_checkpoint_pair(direct_ckpt_obj, reasoning_ckpt_obj)
    del direct_ckpt_obj, reasoning_ckpt_obj

    print("=" * 132)
    print("REASONING ↔ DECISION ↔ VLM TRAJECTORY ↔ ACTION EXPERT DIAGNOSTIC")
    print("=" * 132)
    print("GPU             :", torch.cuda.get_device_name(args.gpu_id))
    print("VLM             :", args.vlm)
    print("Dataset         :", args.jsonl)
    print("Samples         :", len(rows))
    print("Direct AE       :", args.direct_ckpt)
    print("Reasoning AE    :", args.reasoning_ckpt)
    print("Preferred cache :", args.preferred_trace_cache)
    print("Output          :", args.output)
    print("Same Flow x0    : YES, stable sample-ID seed")
    print("Reasoning GT    :", len(rows) - missing_reasoning, "/", len(rows))
    print("Decision GT     :", len(rows) - missing_decision, "/", len(rows))
    print("=" * 132)

    # ------------------------------------------------------------------
    # Stage 1: choose/rebuild exact trace cache + build decision outputs.
    # ------------------------------------------------------------------
    trace_manifest = None
    trace_root = None
    if not args.force_regenerate_trace_cache:
        trace_manifest = validate_trace_cache(args.preferred_trace_cache, rows, args.vlm)
        if trace_manifest is not None:
            trace_root = args.preferred_trace_cache
            print("[TRACE CACHE] Reusing exact prior controlled cache:", trace_root)

    decision_path = args.work_cache_root / "decision_cache.jsonl"
    decision_meta = args.work_cache_root / "decision_cache_meta.json"
    request = decision_cache_request(args.vlm, args.jsonl, args)
    decision_records = None if args.rebuild_decision_cache else load_decision_cache(
        decision_path, decision_meta, rows, request
    )

    need_vlm = trace_manifest is None or decision_records is None
    model = processor = None
    if need_vlm:
        print("[VLM] Loading target model...")
        model, processor = load_target_model(args.vlm, device, dtype, args.attn)

    if trace_manifest is None:
        trace_root = args.work_cache_root / "trace_cache"
        print("[TRACE CACHE] Building fresh trace cache:", trace_root)
        trace_manifest = build_trace_cache(
            trace_root, rows, model, processor, args.vlm, args.jsonl, device, dtype, args
        )

    if decision_records is None:
        print("[DECISION] Building explicit decision cache...")
        decision_records = build_decision_cache(
            decision_path, decision_meta, rows, model, processor, device, dtype,
            args.vlm, args.jsonl, args
        )
    else:
        print("[DECISION] Reusing compatible decision cache:", decision_path)

    args._trace_cache_used = str(trace_root)

    if model is not None:
        del model, processor
        cleanup_cuda()

    # ------------------------------------------------------------------
    # Stage 2: exact same x0, sequential model loading.
    # ------------------------------------------------------------------
    print("\n" + "=" * 132)
    print("STAGE 2 | ACTION EXPERT INFERENCE")
    print("=" * 132)
    direct_ae, direct_attention_recorder = evaluate_action_expert(
        args.direct_ckpt, "direct", rows, trace_manifest, device,
        args.flow_noise_seed, args.solver_steps,
        store_attention_vectors=True,
    )
    reasoning_ae, reasoning_attention_recorder = evaluate_action_expert(
        args.reasoning_ckpt, "reasoning", rows, trace_manifest, device,
        args.flow_noise_seed, args.solver_steps,
        store_attention_vectors=True,
    )

    # ------------------------------------------------------------------
    # Stage 3: per-sample relationship analysis.
    # ------------------------------------------------------------------
    print("\n" + "=" * 132)
    print("STAGE 3 | RELATIONSHIP / CAUSAL-DIAGNOSTIC ANALYSIS")
    print("=" * 132)
    samples = analyze_samples(rows, trace_manifest, decision_records, direct_ae, reasoning_ae)
    summary = build_summary(samples, args.bootstrap, SEED)
    counterexamples = build_counterexamples(samples, args.counterexample_topk)

    print("[ATTENTION] Aggregating layer/Euler diagnostics, collapse, and AE-gain correlations...")
    attention_diagnostics = build_attention_diagnostics(
        samples,
        direct_attention_recorder,
        reasoning_attention_recorder,
    )
    summary["attention_diagnostics_overview"] = {
        "direct_overall": attention_diagnostics["direct"]["overall"],
        "reasoning_overall": attention_diagnostics["reasoning"]["overall"],
        "cross_scene_collapse": {
            "direct_attended_context_overall":
                attention_diagnostics["cross_scene_collapse"]["direct"]["attended_context"]["overall"],
            "reasoning_attended_context_overall":
                attention_diagnostics["cross_scene_collapse"]["reasoning"]["attended_context"]["overall"],
            "reasoning_context_overall":
                attention_diagnostics["cross_scene_collapse"]["reasoning"]["reasoning_context"]["overall"],
        },
        "attention_metric_vs_ae_gain_correlation":
            attention_diagnostics["attention_metric_vs_ae_gain_correlation"]["reasoning_attention_overall"],
    }

    # Runtime/provenance.
    summary["runtime"] = {
        "vlm": str(args.vlm),
        "dataset": str(args.jsonl),
        "trace_cache": str(trace_root),
        "decision_cache": str(decision_path),
        "direct_checkpoint": str(args.direct_ckpt),
        "reasoning_checkpoint": str(args.reasoning_ckpt),
        "flow_noise_seed": int(args.flow_noise_seed),
        "same_noise_per_sample": True,
        "dtype": args.dtype,
        "attention": args.attn,
        "bootstrap_iterations": int(args.bootstrap),
        "limit": args.limit,
    }

    write_jsonl(args.output / "samples.jsonl", samples)
    write_csv(args.output / "samples.csv", samples)
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "counterexamples.json", counterexamples)
    save_json(args.output / "attention_diagnostics.json", attention_diagnostics)

    print("[ATTENTION] Packing attention_vectors.pt ...")
    attention_vector_payload = build_attention_vector_payload(
        direct_attention_recorder,
        reasoning_attention_recorder,
    )
    torch.save(attention_vector_payload, args.output / "attention_vectors.pt")
    del attention_vector_payload
    cleanup_cuda()

    plots = maybe_make_plots(args.output / "plots", samples)
    summary["plots"] = plots
    save_json(args.output / "summary.json", summary)
    write_report(
        args.output / "report.md",
        summary,
        counterexamples,
        attention_diagnostics,
        args,
    )

    print("\n" + "=" * 132)
    print("DONE")
    print("=" * 132)
    print("Samples                    :", len(samples))
    print("VLM trajectory parse rate  :", pct(summary["parse_rates"]["vlm_trajectory"]))
    print("Decision parse rate        :", pct(summary["parse_rates"]["decision"]))
    print("Direct AE ADE              :", fmt(summary["overall"]["direct_ae_ade_m"]), "m")
    print("Reasoning AE ADE           :", fmt(summary["overall"]["reasoning_ae_ade_m"]), "m")
    print("Reasoning AE mean gain     :", fmt(summary["overall"]["reasoning_ae_gain_ade_m"]), "m")
    print("Reasoning quality→gain rho :", fmt(summary["correlations"]["reasoning_quality_vs_reasoning_ae_gain"]["spearman"]))
    print("VLM ADE→Reason AE ADE rho  :", fmt(summary["correlations"]["vlm_trajectory_ade_vs_reasoning_ae_ade"]["spearman"]))
    print("GT decision↔traj calib     :", pct(summary["gt_decision_trajectory_calibration"]["signed_joint_match"]))
    print("Report                     :", args.output / "report.md")
    print("Summary                    :", args.output / "summary.json")
    print("Per-sample                 :", args.output / "samples.jsonl")
    print("Counterexamples            :", args.output / "counterexamples.json")
    print("Attention diagnostics      :", args.output / "attention_diagnostics.json")
    print("Attention vectors          :", args.output / "attention_vectors.pt")
    print("Reasoning attn share       :", fmt(
        attention_diagnostics["reasoning"]["overall"].get(
            "reasoning_attention_share_within_vlm"
        )
    ))
    print("Reasoning attn enrichment  :", fmt(
        attention_diagnostics["reasoning"]["overall"].get(
            "reasoning_attention_enrichment"
        )
    ))
    if plots:
        print("Plots                      :", args.output / "plots")


if __name__ == "__main__":
    main()
