#!/usr/init/env python3
# -*- coding: utf-8 -*-
"""
Integrated Paired Causal Analysis, HD-MAP & Camera Image Trajectory Visualization
[FINAL CORRECTED] Fixed dataset path to part1_fixedsplit/test.jsonl and relocated Token Impacts box.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import pickle
import random
import sys
import textwrap
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# PATHS / CONSTANTS (Corrected for part1_fixedsplit test set)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_PROJECT_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_PROJECT_ROOT / "scripts"
for _path in (SCRIPT_DIR, ACTION_PROJECT_ROOT, ACTION_SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scripts.action_model_flow_dit import TrajectoryNormalizer
except ImportError:
    from action_model_flow_dit import TrajectoryNormalizer

try:
    from train_rawkv_flow_dit_fixedsplit import build_flow_dit_rawkv
except ImportError as exc:
    raise ImportError("train_rawkv_flow_dit_fixedsplit.py를 찾을 수 없습니다.") from exc

# 올바른 파트 1 테스트 데이터셋 경로로 수정
DEFAULT_DATASET = Path("/home/lhh/lab/Dataset/ActionExpert/part1_fixedsplit/test.jsonl")
RAW_DATA_ROOT = Path("/media/HDD/nuReasoning/train")
DEFAULT_CACHE_ROOT = Path("/home/lhh/lab/Dataset/ActionExpert/reasoning_vlm_v2_fixedsplit_rawkv_dit/action_kv_cache")
DEFAULT_MODEL_ROOT = Path("/home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit")
DEFAULT_RESULT_ROOT = Path("/home/lhh/lab/E2E/Result/rawkv_dit_causality_hdmap_visual")

CONDITIONS = ("DIRECT", "ALL_ON", "ALL_MASKED", "BEST_ON")
REASONING_CONDITIONS = ("ALL_ON", "ALL_MASKED")

# =============================================================================
# nuReasoning PICKLE COMPATIBILITY & HELPERS
# =============================================================================

class NuReasoningCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in {"data_schema", "data_schema_v0"}:
            def _dummy_new(cls, *args, **kwargs): return object.__new__(cls)
            cls = type(str(name), (), {"__new__": _dummy_new, "__init__": lambda s, *a, **kw: None})
            cls.__module__ = module
            return cls
        return super().find_class(module, name)

def load_nureasoning_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return NuReasoningCompatUnpickler(f).load()

def extract_ego_pose(ego_state: Any) -> Tuple[float, float, float]:
    candidates = [ego_state, getattr(ego_state, "pose", None), getattr(ego_state, "rear_axle", None), getattr(ego_state, "center", None)]
    for obj in candidates:
        if obj is None: continue
        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj).reshape(-1)
            if arr.size >= 3: return float(arr[0]), float(arr[1]), float(arr[2])
        if isinstance(obj, dict):
            x, y = obj.get("x"), obj.get("y")
            if x is not None and y is not None:
                yaw = obj.get("heading", obj.get("yaw", obj.get("heading_rad", 0.0)))
                return float(x), float(y), float(yaw)
        if hasattr(obj, "x") and hasattr(obj, "y"):
            yaw = getattr(obj, "heading", getattr(obj, "yaw", 0.0))
            return float(obj.x), float(obj.y), float(yaw)
    return 0.0, 0.0, 0.0

@dataclass
class MapGeometry:
    category: str
    name: str
    xy: np.ndarray

def _numeric_xy_array(value: Any) -> Optional[np.ndarray]:
    if value is None: return None
    if torch.is_tensor(value): value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size in (2, 3): arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] in (2, 3, 4) and arr.shape[0] >= 1:
            try: arr = arr.astype(np.float64, copy=False)[:, :2]
            except Exception: return None
            if np.isfinite(arr).all(): return arr
    if isinstance(value, (list, tuple)) and value:
        try:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1 and arr.size in (2, 3): arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] in (2, 3, 4) and arr.shape[0] >= 1:
                arr = arr[:, :2]
                if np.isfinite(arr).all(): return arr
        except Exception: pass
    return None

def extract_map_geometries(map_obj: Any) -> List[MapGeometry]:
    found = []
    seen = set()
    def visit(obj, path="map", depth=0):
        if depth > 8 or obj is None: return
        arr = _numeric_xy_array(obj)
        if arr is not None and len(arr) >= 2:
            digest = hashlib.sha1(np.round(arr, 3).astype(np.float32).tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                found.append(MapGeometry("map", path, arr))
            return
        if hasattr(obj, "coords"):
            try:
                coords_arr = _numeric_xy_array(list(obj.coords))
                if coords_arr is not None and len(coords_arr) >= 2:
                    found.append(MapGeometry("map", path, coords_arr))
                    return
            except Exception: pass
        if isinstance(obj, dict):
            for k, v in obj.items(): visit(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj): visit(v, f"{path}[{i}]", depth + 1)
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if not str(k).startswith("__"): visit(v, f"{path}.{k}", depth + 1)
    visit(map_obj)
    return found

def global_to_ego_xy(xy: np.ndarray, ego_x: float, ego_y: float, ego_heading: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    dx, dy = xy[:, 0] - ego_x, xy[:, 1] - ego_y
    c, s = math.cos(ego_heading), math.sin(ego_heading)
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy])

def resolve_clip_dir(clip_name: str) -> Path:
    for sub in ("part_1", "part_2", "train"):
        p = RAW_DATA_ROOT / sub / clip_name
        if p.is_dir(): return p
    return RAW_DATA_ROOT / clip_name

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)

def make_x0(base_seed: int, sample_id: str, num_steps: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_sample_seed(base_seed, sample_id))
    return torch.randn((1, num_steps, 3), generator=generator, dtype=torch.float32)

@dataclass
class CacheSample:
    sample_id: str
    clip: str
    direct_key: torch.Tensor
    direct_value: torch.Tensor
    reasoning_key: torch.Tensor
    reasoning_value: torch.Tensor
    trajectory: torch.Tensor
    reasoning_text: str
    gt_reasoning_text: str
    reasoning_token_ids: List[int]
    direct_tokens: int
    reasoning_tokens: int

def load_cache_sample(item: Mapping[str, Any], manifest_path: Path) -> CacheSample:
    sample_id = str(item["id"]).strip()
    raw = Path(str(item["cache_file"]))
    cache_file = raw if raw.is_absolute() else (manifest_path.parent / raw).resolve()
    cache = torch.load(cache_file, map_location="cpu", weights_only=False)

    return CacheSample(
        sample_id=sample_id,
        clip=str(cache.get("clip", item.get("clip", ""))),
        direct_key=cache["direct_key"].cpu().contiguous(),
        direct_value=cache["direct_value"].cpu().contiguous(),
        reasoning_key=cache["reasoning_delta_key"].cpu().contiguous(),
        reasoning_value=cache["reasoning_delta_value"].cpu().contiguous(),
        trajectory=torch.as_tensor(cache["trajectory"], dtype=torch.float32).cpu().contiguous(),
        reasoning_text=str(cache.get("reasoning_text", item.get("reasoning_text", ""))).strip(),
        gt_reasoning_text=str(cache.get("gt_reasoning", cache.get("reasoning", item.get("gt_reasoning", "")))).strip(),
        reasoning_token_ids=[int(x) for x in cache.get("reasoning_token_ids", [])],
        direct_tokens=int(cache["direct_key"].shape[1]),
        reasoning_tokens=int(cache["reasoning_delta_key"].shape[1]),
    )

def load_checkpoint(model_root: Path, branch: str) -> Dict[str, Any]:
    path = model_root / "flow" / f"{branch}_flow_dit" / "best.pt"
    return torch.load(path, map_location="cpu", weights_only=False)

def build_model(checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint["action_config"]
    model = build_flow_dit_rawkv(
        hidden_dim=int(cfg["hidden_dim"]),
        num_steps=int(cfg["num_steps"]),
        num_layers=int(cfg["num_layers"]),
        vlm_num_attention_heads=int(checkpoint["vlm_num_attention_heads"]),
        vlm_num_kv_heads=int(checkpoint["vlm_num_kv_heads"]),
        vlm_head_dim=int(checkpoint["vlm_head_dim"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model

def make_memory_mask(direct_len: int, reasoning_len: int, visible: bool, device: torch.device) -> torch.Tensor:
    mask = torch.ones((1, direct_len + reasoning_len), dtype=torch.bool, device=device)
    if not visible: mask[:, direct_len:] = False
    return mask

@torch.inference_mode()
def euler_sample_from_x0(
    model: torch.nn.Module, vlm_key: torch.Tensor, vlm_value: torch.Tensor,
    direct_len: int, reasoning_len: int, normalizer: TrajectoryNormalizer,
    solver_steps: int, x0_cpu: torch.Tensor, condition: str,
    custom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = vlm_key.device
    x = x0_cpu.to(device=device, dtype=torch.float32).clone()
    dt = 1.0 / float(solver_steps)

    for step in range(solver_steps):
        if custom_mask is not None:
            memory_mask = custom_mask
        else:
            visible = True if condition == "ALL_ON" else False
            memory_mask = make_memory_mask(direct_len, reasoning_len, visible, device)
            
        t = torch.full((1, 1, 1), float(step) / float(solver_steps), device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            velocity = model(x_t=x, t=t, vlm_key=vlm_key, vlm_value=vlm_value, memory_mask=memory_mask)
        x = x + dt * velocity.float()

    return normalizer.to(device).denormalize(x)

@torch.inference_mode()
def compute_token_loo_impact(
    model: torch.nn.Module, vlm_key: torch.Tensor, vlm_value: torch.Tensor,
    direct_len: int, reasoning_len: int, normalizer: TrajectoryNormalizer,
    solver_steps: int, x0_cpu: torch.Tensor, gt_traj: np.ndarray, all_on_ade: float
) -> List[float]:
    device = vlm_key.device
    dt = 1.0 / float(solver_steps)
    impacts = []

    for target_idx in range(reasoning_len):
        x = x0_cpu.to(device=device, dtype=torch.float32).clone()
        mask = torch.ones((1, direct_len + reasoning_len), dtype=torch.bool, device=device)
        mask[0, direct_len + target_idx] = False
        
        for step in range(solver_steps):
            t = torch.full((1, 1, 1), float(step) / float(solver_steps), device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                velocity = model(x_t=x, t=t, vlm_key=vlm_key, vlm_value=vlm_value, memory_mask=mask)
            x = x + dt * velocity.float()
            
        pred = normalizer.to(device).denormalize(x)[0].float().cpu().numpy()
        loo_ade = float(np.linalg.norm(pred[:, :2] - gt_traj[:, :2], axis=-1).mean())
        impacts.append(loo_ade - all_on_ade)

    return impacts

# =============================================================================
# REVISED INTEGRATED PLOTTING LAYOUT (Impact box strictly moved to bottom right)
# =============================================================================

def plot_integrated_sample(
    output_path: Path, sample: CacheSample, map_geoms_ego: Sequence[MapGeometry],
    gt: np.ndarray, predictions: Dict[str, np.ndarray], trajectory_metrics: Dict[str, Dict[str, float]],
    token_impacts: List[float], camera_images: List[np.ndarray], map_radius: float, show: bool
) -> None:
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(
        nrows=4, ncols=2, 
        width_ratios=[2.2, 1.1], 
        height_ratios=[1.0, 1.2, 4.5, 1.8], 
        hspace=0.22, wspace=0.15
    )

    # 1. 상단: Generated Reasoning 및 GT Reasoning 텍스트 박스
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    gen_text = textwrap.shorten(sample.reasoning_text if sample.reasoning_text else "N/A", width=160, placeholder="...")
    gt_text = textwrap.shorten(sample.gt_reasoning_text if sample.gt_reasoning_text else "N/A", width=160, placeholder="...")
    
    text_content = (
        f"Generated Reasoning: {gen_text}\n"
        f"GT Reasoning: {gt_text}"
    )
    ax_text.text(0.0, 0.95, text_content, fontsize=10, va="top", ha="left", 
                 bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="0.8"))

    # 2. 두 번째 행: 입력 카메라 이미지 3장 가로 배치
    sub_gs = gs[1, :].subgridspec(1, 3, wspace=0.08)
    for i in range(3):
        ax_sub = fig.add_subplot(sub_gs[0, i])
        ax_sub.axis("off")
        if i < len(camera_images) and camera_images[i] is not None:
            ax_sub.imshow(camera_images[i])
            ax_sub.set_title(f"Cam {i+1}", fontsize=8)
        else:
            ax_sub.text(0.5, 0.5, f"Cam {i+1} N/A", ha="center", va="center", fontsize=8)
            ax_sub.set_facecolor("#eeeeee")

    # 3. 좌측 하단: HD-Map 및 Trajectory 뷰
    ax_map = fig.add_subplot(gs[2:, 0])
    for geom in map_geoms_ego[:5000]:
        ax_map.plot(geom.xy[:, 0], geom.xy[:, 1], color="0.4", lw=0.8, alpha=0.6)
    
    ax_map.plot(gt[:, 0], gt[:, 1], color="tab:green", lw=2.5, marker="o", ms=4, label="Ground Truth")
    
    colors = {"DIRECT": "tab:blue", "ALL_ON": "tab:red", "ALL_MASKED": "gray", "BEST_ON": "tab:orange"}
    for cond, pred in predictions.items():
        m = trajectory_metrics.get(cond, {"ade_m": 0})
        ax_map.plot(pred[:, 0], pred[:, 1], color=colors.get(cond, "black"), lw=2, label=f"{cond} (ADE:{m['ade_m']:.2f}m)")

    ax_map.scatter([0.0], [0.0], marker="*", s=150, color="black", label="Ego", zorder=10)
    ax_map.set_xlim(-map_radius, map_radius)
    ax_map.set_ylim(-map_radius, map_radius)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", fontsize=8)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_xlabel("Ego X [m]")
    ax_map.set_ylabel("Ego Y [m]")

    # 4. 우측 중단: 토큰별 Causal Impact 목록 박스 (위치 확실히 내림)
    ax_tokens = fig.add_subplot(gs[2, 1])
    ax_tokens.axis("off")
    token_lines = ["--- Token Causal Impacts ---"]
    if sample.reasoning_token_ids and token_impacts:
        sorted_tokens = sorted(zip(sample.reasoning_token_ids, token_impacts), key=lambda x: x[1], reverse=True)
        for tid, imp in sorted_tokens[:15]:
            sign = "+" if imp > 0 else ""
            token_lines.append(f"TID {tid}: {sign}{imp:.3f}")
    ax_tokens.text(0.0, 1.0, "\n".join(token_lines), fontsize=9, family="monospace", va="top", ha="left", 
                   bbox=dict(boxstyle="square,pad=0.6", fc="#f9f9f9", ec="0.6"))

    # 5. 우측 하단: 조건별 ADE 메트릭 박스
    ax_metrics = fig.add_subplot(gs[3, 1])
    ax_metrics.axis("off")
    metric_lines = ["--- Condition Metrics (ADE) ---"]
    for cond, m in trajectory_metrics.items():
        metric_lines.append(f"{cond:<12}: {m['ade_m']:.3f}m")
    ax_metrics.text(0.0, 1.0, "\n".join(metric_lines), fontsize=9, family="monospace", va="top", ha="left", 
                    bbox=dict(boxstyle="square,pad=0.6", fc="#f1f1f1", ec="0.6"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

# =============================================================================
# MAIN EVALUATION SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-part2-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--map-radius", type=float, default=50.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    manifest_path = args.cache_root / args.split / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)[:args.limit]
    
    dataset_rows = {row["id"]: row for row in read_jsonl(args.dataset)}

    out_dir = args.result_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "direct": load_checkpoint(args.model_root, "direct"),
        "reasoning": load_checkpoint(args.model_root, "reasoning")
    }

    direct_model = build_model(checkpoints["direct"], device)
    reason_model = build_model(checkpoints["reasoning"], device)
    normalizer = TrajectoryNormalizer.from_dict(checkpoints["direct"]["normalizer"]).to(device)
    solver_steps = 10

    print("==================================================================")
    print("🚀 RUNNING CAUSALITY ANALYSIS, HD-MAP & CAMERA VISUALIZATION")
    print("==================================================================")

    for index, item in enumerate(manifest, 1):
        sample = load_cache_sample(item, manifest_path)
        
        if not sample.gt_reasoning_text and sample.sample_id in dataset_rows:
            raw_row = dataset_rows[sample.sample_id]
            sample.gt_reasoning_text = first_nonempty(raw_row.get("gt_reasoning"), raw_row.get("reasoning"))

        x0 = make_x0(20260841, sample.sample_id, 10)
        gt_np = sample.trajectory.numpy()
        
        # 1. DIRECT 평가
        dk, dv = sample.direct_key.unsqueeze(0).to(device, dtype=torch.bfloat16), sample.direct_value.unsqueeze(0).to(device, dtype=torch.bfloat16)
        pred_dir = euler_sample_from_x0(direct_model, dk, dv, sample.direct_tokens, 0, normalizer, solver_steps, x0, "ALL_ON")
        
        preds, metrics = {}, {}
        preds["DIRECT"] = pred_dir[0].float().cpu().numpy()
        metrics["DIRECT"] = {"ade_m": float(np.linalg.norm(preds["DIRECT"][:, :2] - gt_np[:, :2], axis=-1).mean())}
        del dk, dv

        # 2. Reasoning 조건별 평가
        rk = torch.cat([sample.direct_key, sample.reasoning_key], dim=1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        rv = torch.cat([sample.direct_value, sample.reasoning_value], dim=1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        all_on_ade = 0.0
        for cond in REASONING_CONDITIONS:
            pred = euler_sample_from_x0(reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, normalizer, solver_steps, x0, cond)
            pred_np = pred[0].float().cpu().numpy()
            preds[cond] = pred_np
            ade = float(np.linalg.norm(pred_np[:, :2] - gt_np[:, :2], axis=-1).mean())
            metrics[cond] = {"ade_m": ade}
            if cond == "ALL_ON": all_on_ade = ade

        # 3. 토큰 인과성 분석 및 BEST_ON 오라클 마스킹
        impacts = compute_token_loo_impact(
            reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, 
            normalizer, solver_steps, x0, gt_np, all_on_ade
        )
        
        helpful_indices = [i for i, imp in enumerate(impacts) if imp > 0]
        best_mask = torch.ones((1, sample.direct_tokens + sample.reasoning_tokens), dtype=torch.bool, device=device)
        best_mask[0, sample.direct_tokens:] = False
        for idx in helpful_indices:
            best_mask[0, sample.direct_tokens + idx] = True
            
        pred_best = euler_sample_from_x0(
            reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, 
            normalizer, solver_steps, x0, "BEST_ON", custom_mask=best_mask
        )
        preds["BEST_ON"] = pred_best[0].float().cpu().numpy()
        metrics["BEST_ON"] = {"ade_m": float(np.linalg.norm(preds["BEST_ON"][:, :2] - gt_np[:, :2], axis=-1).mean())}

        # 4. HD-Map 및 카메라 이미지 로드
        map_geoms_ego = []
        camera_images = [None, None, None]
        try:
            clip_dir = resolve_clip_dir(sample.clip)
            metadata_path = clip_dir / "metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                matched_frame = None
                for f in metadata.get("frames", []):
                    if str(f.get("token")) == sample.sample_id:
                        matched_frame = f
                        break
                if not matched_frame and metadata.get("frames"):
                    matched_frame = metadata["frames"][0]

                if matched_frame:
                    ego_pose = extract_ego_pose(load_nureasoning_pickle(clip_dir / matched_frame["ego_state"]))
                    map_obj = load_nureasoning_pickle(clip_dir / metadata.get("map_annotation", "map.pkl"))
                    map_geoms = extract_map_geometries(map_obj)
                    map_geoms_ego = [MapGeometry(g.category, g.name, global_to_ego_xy(g.xy, *ego_pose)) for g in map_geoms]

                    # 카메라 이미지 로드
                    cams = matched_frame.get("cameras", matched_frame.get("cam_paths", []))
                    if isinstance(cams, dict): cams = list(cams.values())
                    for i in range(min(3, len(cams))):
                        img_path = clip_dir / str(cams[i])
                        if img_path.is_file():
                            from matplotlib.image import imread
                            camera_images[i] = imread(img_path)
        except Exception as e:
            print(f"[WARN] HD-map or camera load skipped for {sample.sample_id}: {e}")

        plot_integrated_sample(
            output_path=out_dir / f"{index:04d}_{sample.sample_id}.png",
            sample=sample, map_geoms_ego=map_geoms_ego, gt=gt_np,
            predictions=preds, trajectory_metrics=metrics,
            token_impacts=impacts, camera_images=camera_images, map_radius=args.map_radius, show=args.show
        )
        print(f"[{index:04d}/{len(manifest):04d}] Visualized & Saved: {sample.sample_id}")

    cleanup_cuda()
    print(f"\n✅ All results and visualizations successfully saved to: {out_dir}")

if __name__ == "__main__":
    main()