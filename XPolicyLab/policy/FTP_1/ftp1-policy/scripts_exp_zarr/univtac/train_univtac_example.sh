#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ================================ Core parameters ================================
repo_id="univtac_example"
exp_name="univtac_train"
dataset_config_path="scripts_exp_zarr/univtac/dataset_univtac.json"

checkpoint_base_dir="/path/to/ftp1_cache/checkpoints"
assets_base_dir="/path/to/ftp1_cache/assets"
export OPENPI_DATA_HOME="/path/to/ftp1_cache/openpi"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HYDRA_FULL_ERROR=1

batch_size=64
num_train_steps=20000
log_interval=100
val_interval=2000
val_ratio=0.0
save_interval=10000
keep_period=10000
num_workers=12
val_num_workers=2

action_down_sample_steps=1
norm_type="zscore"
norm_image_tactile_mode="channel_wise"
state_input_mode="adarms"
model_tactile_expert_variant="gemma_small"
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"

pytorch_training_precision="bfloat16"
use_torch_compile="true"
lr_warmup_steps=500
lr_peak_lr="5e-5"
lr_decay_lr="5e-6"

optimizer_b1=0.9
optimizer_b2=0.95
optimizer_eps="1e-8"
optimizer_weight_decay="1e-10"
optimizer_clip_gradient_norm=1.0

pytorch_weight_path="./checkpoints/pretrain_ckpt/ftp1_pretrain_v0426_50kstep"
use_wandb="true"

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c . || echo "1")
if [ "${num_gpus}" -lt 1 ]; then
  num_gpus=1
fi

TRAIN_ARGS=(
  ftp1
  --exp_name=${exp_name}
  --repo_id=${repo_id}
  --data.repo-id=${repo_id}
  --checkpoint_base_dir=${checkpoint_base_dir}
  --assets_base_dir=${assets_base_dir}
  --dataset_config_path=${dataset_config_path}
  --batch_size=${batch_size}
  --action_down_sample_steps=${action_down_sample_steps}
  --num_train_steps=${num_train_steps}
  --log_interval=${log_interval}
  --val_interval=${val_interval}
  --val_ratio=${val_ratio}
  --save_interval=${save_interval}
  --keep_period=${keep_period}
  --num_workers=${num_workers}
  --val_num_workers=${val_num_workers}
  --use_val_dataset
  --create_train_val_split
  --no-check_norm_params_snapshot
  --norm_type=${norm_type}
  --norm_image_tactile_mode=${norm_image_tactile_mode}
  --pytorch_training_precision=${pytorch_training_precision}
  --model.state_input_mode=${state_input_mode}
  --model.tactile_expert_variant=${model_tactile_expert_variant}
  --model.use_tactile_input
  --proprioception_pose_rep=${proprioception_pose_rep}
  --action_pose_rep=${action_pose_rep}
  --proprioception_joint_rep=${proprioception_joint_rep}
  --action_joint_rep=${action_joint_rep}
  --lr_schedule.warmup_steps=${lr_warmup_steps}
  --lr_schedule.peak_lr=${lr_peak_lr}
  --lr_schedule.decay_steps=${num_train_steps}
  --lr_schedule.decay_lr=${lr_decay_lr}
  --optimizer.b1=${optimizer_b1}
  --optimizer.b2=${optimizer_b2}
  --optimizer.eps=${optimizer_eps}
  --optimizer.weight_decay=${optimizer_weight_decay}
  --optimizer.clip_gradient_norm=${optimizer_clip_gradient_norm}
  --lr_decay_till_end
  --ema_decay=0.99
)

if [ "${use_wandb}" = "true" ]; then
  TRAIN_ARGS+=(--wandb_enabled)
else
  TRAIN_ARGS+=(--no-wandb_enabled)
fi

if [ "${use_torch_compile}" = "true" ]; then
  TRAIN_ARGS+=(--use_torch_compile)
else
  TRAIN_ARGS+=(--no-use_torch_compile)
fi

TRAIN_ARGS+=(--no-gradient_checkpointing_enable)

if [ -n "${pytorch_weight_path}" ]; then
  TRAIN_ARGS+=(--pytorch_weight_path=${pytorch_weight_path})
fi

if [ "${num_gpus}" -gt 1 ]; then
  uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
else
  uv run python scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
fi
