#!/usr/bin/env python3
"""Offline FTP-1 checkpoint smoke test; never connects to or commands a robot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from XPolicyLab.policy.FTP_1.model import Model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _synthetic_observation() -> dict[str, object]:
    tactile = {
        "right_tactile_fingertip": np.zeros((5, 4, 4), dtype=np.float32),
        "right_tactile_palm": np.zeros((1, 15, 16), dtype=np.float32),
        "left_tactile_fingertip": np.zeros((5, 4, 4), dtype=np.float32),
        "left_tactile_palm": np.zeros((1, 15, 16), dtype=np.float32),
    }
    for value in tactile.values():
        value.reshape(-1)[0] = 1.0

    return {
        "instruction": "Build a tower with the blocks.",
        "vision": {
            "cam_head": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "state": {
            "left_ee_pose": np.array([0.3, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "right_ee_pose": np.array([0.3, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "left_ee_joint_state": np.zeros(20, dtype=np.float32),
            "right_ee_joint_state": np.zeros(20, dtype=np.float32),
        },
        "tactile": tactile,
    }


def main() -> None:
    args = _parse_args()
    if os.environ.get("FTP1_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("Unset FTP1_TEST_MODE: this command must load the real checkpoint")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy_dir = Path(__file__).resolve().parent
    with (policy_dir / "deploy.yml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.update(
        checkpoint_path=str(args.checkpoint.resolve()),
        env_cfg_type="tianji_marvin_wuji",
        action_type="ee",
        device=args.device,
        num_inference_steps=args.num_inference_steps,
    )

    start = time.monotonic()
    model = Model(config)
    loaded_s = time.monotonic() - start
    expected_tactile_keys = {
        "right_tactile_fingertip",
        "right_tactile_palm",
        "left_tactile_fingertip",
        "left_tactile_palm",
    }
    if set(model.wrapper.tactile_key_type_map) != expected_tactile_keys:
        raise AssertionError(
            "Checkpoint tactile type map is incomplete: "
            f"got={sorted(model.wrapper.tactile_key_type_map)}, expected={sorted(expected_tactile_keys)}"
        )
    if set(model.wrapper.tactile_key_type_map.values()) != {"matrix"}:
        raise AssertionError(f"Unexpected tactile types: {model.wrapper.tactile_key_type_map}")
    if model.wrapper.domain_norm_stats is None:
        raise AssertionError("Checkpoint domain normalization statistics were not loaded")
    normalization_dir = Path(model.wrapper.ckpt_dir) / "normalization"
    shared_stats_files = list(normalization_dir.glob("share_norm_stats_*.json"))
    if len(shared_stats_files) != 1:
        raise AssertionError(f"Expected one shared normalization JSON, got {shared_stats_files}")
    with shared_stats_files[0].open(encoding="utf-8") as file:
        shared_stats_document = json.load(file)
    shared_params = shared_stats_document.get("params")
    if not isinstance(shared_params, dict):
        raise AssertionError(f"Invalid shared normalization JSON: {shared_stats_files[0]}")
    if shared_params and model.wrapper.shared_norm_stats is None:
        raise AssertionError("Non-empty shared normalization statistics were not loaded")
    tokenizer_dir = Path(model.wrapper.ckpt_dir) / "hpt_tokenizer"
    required_tokenizers = {
        "MoxianTactileGlove460_matrix_4_4.safetensors",
        "MoxianTactileGlove460_matrix_15_16.safetensors",
    }
    missing_tokenizers = [name for name in required_tokenizers if not (tokenizer_dir / name).is_file()]
    if missing_tokenizers:
        raise FileNotFoundError(f"Missing Moxian tokenizer checkpoints: {sorted(missing_tokenizers)}")

    model.update_obs(_synthetic_observation())
    infer_start = time.monotonic()
    actions = model.get_action()
    inference_s = time.monotonic() - infer_start

    expected_keys = {
        "left_ee_pose": 7,
        "right_ee_pose": 7,
        "left_ee_joint_state": 20,
        "right_ee_joint_state": 20,
    }
    expected_action_count = model.execute_horizon or (model.action_horizon - model.action_start_index)
    if len(actions) != expected_action_count:
        raise AssertionError(f"Expected {expected_action_count} executable actions, got {len(actions)}")
    for step, action in enumerate(actions):
        if set(action) != set(expected_keys):
            raise AssertionError(f"Step {step} action keys are {sorted(action)}")
        for key, dimension in expected_keys.items():
            value = np.asarray(action[key])
            if value.shape != (dimension,):
                raise AssertionError(f"Step {step} {key} shape is {value.shape}, expected {(dimension,)}")
            if not np.all(np.isfinite(value)):
                raise AssertionError(f"Step {step} {key} contains NaN or infinity")

    result = {
        "status": "FTP1_REAL_CHECKPOINT_SMOKE_OK",
        "checkpoint": str(model.wrapper.ckpt_dir),
        "device": str(args.device),
        "num_inference_steps": args.num_inference_steps,
        "model_action_horizon": model.action_horizon,
        "action_start_index": model.action_start_index,
        "executed_action_count": len(actions),
        "moxian_tokenizers_present": sorted(required_tokenizers),
        "domain_normalization_loaded": True,
        "shared_normalization_parameter_count": len(shared_params),
        "tactile_contract_keys": sorted(expected_tactile_keys),
        "model_load_seconds": round(loaded_s, 3),
        "inference_seconds": round(inference_s, 3),
        "all_outputs_finite": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
