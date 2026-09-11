#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired causal analysis for RAW-KV Flow/DiT Action Experts.
[UPDATED] Token-level LOO Causality & BEST_ON (Oracle Masking) Evaluation.

This script evaluates:
    DIRECT       : direct prompt RAW K/V
    ALL_ON       : prompt and reasoning RAW K/V
    ALL_MASKED   : reasoning RAW K/V always masked
    EARLY_ON     : reasoning RAW K/V visible in the first half of Euler steps
    LATE_ON      : reasoning RAW K/V visible in the second half of Euler steps
    BEST_ON      : [NEW] Only keeps reasoning tokens that were proven helpful by LOO test.

Outputs:
    - samples_causality.jsonl : Per-sample metrics and LOO impacts
    - token_causality_raw.csv : Raw LOO impact for every token in every sample
    - global_token_stats.csv  : [NEW] Aggregated mean impact and win rate per token_id
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_PROJECT_ROOT = Path("/home/lhh/lab/Action_Expert")
ACTION_SCRIPT_DIR = ACTION_PROJECT_ROOT / "scripts"
for _path in (SCRIPT_DIR, ACTION_PROJECT_ROOT, ACTION_SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from scripts.action_model_flow_dit import (
        TrajectoryNormalizer,
        per_sample_metrics_np,
        trajectory_metrics_np,
    )
except ImportError:
    from action_model_flow_dit import (
        TrajectoryNormalizer,
        per_sample_metrics_np,
        trajectory_metrics_np,
    )

try:
    from train_rawkv_flow_dit_fixedsplit import build_flow_dit_rawkv
except ImportError as exc:
    raise ImportError("train_rawkv_flow_dit_fixedsplit.py를 찾을 수 없습니다.") from exc

DEFAULT_CACHE_ROOT = Path("/home/lhh/lab/Dataset/ActionExpert/reasoning_vlm_v2_fixedsplit_rawkv_dit/action_kv_cache")
DEFAULT_MODEL_ROOT = Path("/home/lhh/lab/models/action_expert/reasoning_vlm_v2_fixedsplit_rawkv_dit")
DEFAULT_RESULT_ROOT = Path("/home/lhh/lab/E2E/Result/rawkv_dit_causality_token_impact")

# 추가된 BEST_ON 조건
CONDITIONS = ("DIRECT", "ALL_ON", "ALL_MASKED", "EARLY_ON", "LATE_ON", "BEST_ON")
REASONING_CONDITIONS = ("ALL_ON", "ALL_MASKED", "EARLY_ON", "LATE_ON")

# =============================================================================
# GENERIC HELPERS
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        reasoning_token_ids=[int(x) for x in cache.get("reasoning_token_ids", [])],
        direct_tokens=int(cache["direct_key"].shape[1]),
        reasoning_tokens=int(cache["reasoning_delta_key"].shape[1]),
    )

def load_checkpoint(model_root: Path, branch: str) -> Dict[str, Any]:
    path = model_root / "flow" / f"{branch}_flow_dit" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint

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

# =============================================================================
# EULER SAMPLING & TOKEN CAUSALITY
# =============================================================================

def make_memory_mask(direct_len: int, reasoning_len: int, visible: bool, device: torch.device) -> torch.Tensor:
    mask = torch.ones((1, direct_len + reasoning_len), dtype=torch.bool, device=device)
    if not visible:
        mask[:, direct_len:] = False
    return mask

@torch.inference_mode()
def euler_sample_from_x0(
    model: torch.nn.Module,
    vlm_key: torch.Tensor,
    vlm_value: torch.Tensor,
    direct_len: int,
    reasoning_len: int,
    normalizer: TrajectoryNormalizer,
    solver_steps: int,
    x0_cpu: torch.Tensor,
    condition: str,
    custom_mask: Optional[torch.Tensor] = None, # 커스텀 마스크 지원 추가
) -> torch.Tensor:
    device = vlm_key.device
    x = x0_cpu.to(device=device, dtype=torch.float32).clone()
    dt = 1.0 / float(solver_steps)
    split_step = solver_steps // 2

    for step in range(solver_steps):
        if custom_mask is not None:
            memory_mask = custom_mask
        else:
            if condition == "ALL_ON": visible = True
            elif condition == "ALL_MASKED": visible = False
            elif condition == "EARLY_ON": visible = (step < split_step)
            elif condition == "LATE_ON": visible = (step >= split_step)
            else: visible = True

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
        mask[0, direct_len + target_idx] = False # 토큰 하나만 가림
        
        for step in range(solver_steps):
            t = torch.full((1, 1, 1), float(step) / float(solver_steps), device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                velocity = model(x_t=x, t=t, vlm_key=vlm_key, vlm_value=vlm_value, memory_mask=mask)
            x = x + dt * velocity.float()
            
        pred = normalizer.to(device).denormalize(x)[0].float().cpu().numpy()
        loo_ade = float(np.linalg.norm(pred[:, :2] - gt_traj[:, :2], axis=-1).mean())
        
        # Impact > 0 : 도움이 되던 단어
        # Impact < 0 : 방해하던 단어
        impact = loo_ade - all_on_ade
        impacts.append(impact)

    return impacts

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_all(manifest, manifest_path, checkpoints, device, seed, solver_steps):
    direct_model = build_model(checkpoints["direct"], device)
    reason_model = build_model(checkpoints["reasoning"], device)
    normalizer = TrajectoryNormalizer.from_dict(checkpoints["direct"]["normalizer"]).to(device)

    predictions = {c: [] for c in CONDITIONS}
    samples_data = []
    token_causality_rows = []
    global_token_stats = defaultdict(list)

    start = time.perf_counter()
    for index, item in enumerate(manifest, 1):
        sample = load_cache_sample(item, manifest_path)
        x0 = make_x0(seed, sample.sample_id, 10)
        gt_np = sample.trajectory.numpy()
        
        # 1. DIRECT Evaluate
        dk, dv = sample.direct_key.unsqueeze(0).to(device, dtype=torch.bfloat16), sample.direct_value.unsqueeze(0).to(device, dtype=torch.bfloat16)
        pred_dir = euler_sample_from_x0(direct_model, dk, dv, sample.direct_tokens, 0, normalizer, solver_steps, x0, "ALL_ON")
        predictions["DIRECT"].append(pred_dir[0].float().cpu().numpy())
        del dk, dv
        
        # 2. Reasoning Baseline Evaluate
        rk, rv = torch.cat([sample.direct_key, sample.reasoning_key], dim=1).unsqueeze(0).to(device, dtype=torch.bfloat16), \
                 torch.cat([sample.direct_value, sample.reasoning_value], dim=1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        all_on_ade = 0.0
        for cond in REASONING_CONDITIONS:
            pred = euler_sample_from_x0(reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, normalizer, solver_steps, x0, cond)
            pred_np = pred[0].float().cpu().numpy()
            predictions[cond].append(pred_np)
            
            if cond == "ALL_ON":
                all_on_ade = float(np.linalg.norm(pred_np[:, :2] - gt_np[:, :2], axis=-1).mean())

        # 3. Token-Level LOO Causality Evaluation
        impacts = compute_token_loo_impact(
            reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, 
            normalizer, solver_steps, x0, gt_np, all_on_ade
        )
        
        # 4. [NEW] Oracle Masking: BEST_ON (Helpful token만 활성화)
        helpful_indices = [i for i, imp in enumerate(impacts) if imp > 0] # Impact가 양수인 것들만 추출
        
        best_mask = torch.ones((1, sample.direct_tokens + sample.reasoning_tokens), dtype=torch.bool, device=device)
        best_mask[0, sample.direct_tokens:] = False # 기본적으로 Reasoning 토큰 전부 마스킹
        for idx in helpful_indices:
            best_mask[0, sample.direct_tokens + idx] = True # 도움이 되는 놈들만 마스킹 해제 (True)
            
        pred_best = euler_sample_from_x0(
            reason_model, rk, rv, sample.direct_tokens, sample.reasoning_tokens, 
            normalizer, solver_steps, x0, "BEST_ON", custom_mask=best_mask
        )
        pred_best_np = pred_best[0].float().cpu().numpy()
        predictions["BEST_ON"].append(pred_best_np)
        best_on_ade = float(np.linalg.norm(pred_best_np[:, :2] - gt_np[:, :2], axis=-1).mean())
        
        # 데이터 저장
        samples_data.append({
            "sample_id": sample.sample_id,
            "reasoning_text": sample.reasoning_text,
            "reasoning_token_ids": sample.reasoning_token_ids,
            "token_causal_impacts": impacts,
            "all_on_ade": all_on_ade,
            "best_on_ade": best_on_ade,
            "best_on_gain": all_on_ade - best_on_ade # Best ON 했을 때 얼마나 더 개선되었는지
        })
        
        for t_idx, (t_id, impact) in enumerate(zip(sample.reasoning_token_ids, impacts)):
            token_causality_rows.append({
                "sample_id": sample.sample_id,
                "token_index": t_idx,
                "token_id": t_id,
                "causal_impact_ade": impact
            })
            global_token_stats[t_id].append(impact)
            
        print(f"[{index:04d}/{len(manifest):04d}] ALL_ON_ADE: {all_on_ade:.4f} | BEST_ON_ADE: {best_on_ade:.4f}", flush=True)

    cleanup_cuda()
    
    # Global Stats 계산
    global_rows = []
    for t_id, imp_list in global_token_stats.items():
        global_rows.append({
            "token_id": t_id,
            "total_occurrences": len(imp_list),
            "mean_causal_impact": float(np.mean(imp_list)),
            "helpful_ratio_percent": float(np.mean([1 if x > 0 else 0 for x in imp_list]) * 100)
        })
    global_rows = sorted(global_rows, key=lambda x: x["mean_causal_impact"], reverse=True)

    return {c: np.stack(v) for c, v in predictions.items()}, samples_data, token_causality_rows, global_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    manifest_path = args.cache_root / args.split / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    
    out_dir = args.result_root / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "direct": load_checkpoint(args.model_root, "direct"),
        "reasoning": load_checkpoint(args.model_root, "reasoning")
    }

    print("==================================================================")
    print("🚀 RUNNING TOKEN-LEVEL CAUSALITY & BEST_ON (ORACLE) EVALUATION")
    print("==================================================================")
    
    preds, samples_data, causality_rows, global_rows = evaluate_all(manifest, manifest_path, checkpoints, device, 20260841, 10)

    # Save outputs
    write_json(out_dir / "samples_causality.jsonl", [json.dumps(s) for s in samples_data])
    
    with (out_dir / "token_causality_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "token_index", "token_id", "causal_impact_ade"])
        writer.writeheader()
        writer.writerows(causality_rows)

    with (out_dir / "global_token_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token_id", "total_occurrences", "mean_causal_impact", "helpful_ratio_percent"])
        writer.writeheader()
        writer.writerows(global_rows)

    # 간단한 결과 요약 출력
    mean_all_on = np.mean([s["all_on_ade"] for s in samples_data])
    mean_best_on = np.mean([s["best_on_ade"] for s in samples_data])
    
    print("\n==================================================================")
    print("🎯 EVALUATION SUMMARY")
    print("==================================================================")
    print(f"Avg ALL_ON ADE    : {mean_all_on:.4f} m")
    print(f"Avg BEST_ON ADE   : {mean_best_on:.4f} m")
    print(f"Oracle ADE Gain   : {mean_all_on - mean_best_on:.4f} m improvement!")
    print(f"✅ Saved global token stats to: {out_dir}/global_token_stats.csv")

if __name__ == "__main__":
    main()