# FTP_1

**Upstream:** [FTP-1](https://github.com/michaelyuancb/ftp1-policy)  
**Pinned source:** `89fa681d6c014cce28300946b7526db808e0b1c1`  
**Paper:** [FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation](https://arxiv.org/abs/2606.13102)

Current production recipe is `Spark0_real_bench_v5` + Moxian, trained on the 8-task joint-only Zarr under `data/spark0_real_bench_v5_moxian_joint_only`. The source HDF5 is `/personal/xspark_shared/hand_data/hdf5/spark0_real/bench_v5`.

## Supported interface

- `env_cfg_type: tianji_marvin_wuji`
- `action_type: ee` only
- one RGB observation: `cam_head`, resized to 224×224
- absolute world-coordinate left/right EE poses `[x,y,z,qw,qx,qy,qz]`
- left/right Wuji hand joints, 20 dimensions per hand; negative joint values are valid and must not be clamped
- four Moxian tactile streams: left/right fingertip `(5,4,4)` and palm `(1,15,16)`
- 120-D FTP container with 58 supervised dimensions: two pose9 rot6D blocks and two 20-D hands
- model horizon 33 with `action_start_index=1`, so `execute_horizon` must be 32

Normalization domain and deploy `domain_name` are both `Spark0_real_bench_v5_Moxian_joint_only`.

Training instructions stored in the Zarr (and required at deploy time) are the HDF5 strings:

```text
Build a tower with the blocks.
Wipe the blackboard clean.
Insert the test tubes into the rack.
Pack the shoes into the box.
Place the phone in the designated location.
Stack the bowls on top of one another.
Stand the bottle upright.
Transfer water from one container to the other using the dropper.
```

`model.py` also accepts task-directory names and shorter aliases, then maps them back to these eight strings.

## Installation

```bash
source /mnt/xspark-data/conda_envs/FTP_1/bin/activate
```

The vendored upstream checkout is under `policy/FTP_1/ftp1-policy`. Official pretrained weights live in `pretrain_model/ftp1_pretrain_v0426_50kstep`; `download_pretrained.sh` can refresh that snapshot.

## Data Processing

```bash
/mnt/xspark-data/conda_envs/FTP_1/bin/python \
  XPolicyLab/policy/FTP_1/data_scripts/convert_spark_moxian_dataset.py \
  --workers 4 --joint-only-120d
```

See `data_scripts/README.md` for the 30 Hz row-identity contract and validation.

## Training

```bash
bash XPolicyLab/policy/FTP_1/train.sh \
  Spark0_real_bench_v5 Moxian tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

Defaults are 200000 steps, `val_interval=10000`, `save_interval=10000`, batch 8 per GPU. Override with `FTP1_NUM_TRAIN_STEPS`, `FTP1_VAL_INTERVAL`, and `FTP1_SAVE_INTERVAL`.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> tianji_marvin_wuji ee 42 \
  <policy_gpu_id> <env_gpu_id> <ftp_policy_env> <eval_env_conda_env>
```

`deploy.yml` must stay aligned with this recipe: the v5 domain name, local `assets/openpi`, 8-task instructions, `action_start_index=1`, and `execute_horizon=32`.

## Known limitations

- Moxian matrix tokenizers are new sensors and are not in the official 50k pretrained snapshot; the backbone still loads, and those two tokenizers start from random initialization unless a later fine-tune checkpoint saved them.
- Moxian pressure uses the training convention `max(raw - session_baseline, 0)`.
- Joint-control evaluation is intentionally rejected.
