# Spark 墨现触觉数据 → FTP-1

本目录保存全量转换和验收脚本。转换只处理数据，不修改 FTP-1 网络结构，也不会启动训练。

## 脚本

- `parse_data_spark_moxian.py`：单个 HDF5 episode 转换器。
- `convert_spark_moxian_dataset.py`：可断点续跑的全量并行转换器。
- `validate_spark_moxian_dataset.py`：全量格式和 QC 验收。
- `dataset_spark0_real_bench_v5_moxian_joint_only.json`：指向 bench_v5 joint-only 输出目录的 FTP-1 数据配置。

## 数据表示

生产输入是已经逐行对齐的 30 Hz Spark/HumanTouch HDF5。转换器唯一读取的触觉
源是 `tactile/taxel_pressure_3D/{left,right}_ee`，其 shape 为 `(T,320,4)`；只取
最后一列 pressure，并逐值保留为 `float32`：

- 指尖：`(T, 5, 4, 4)`
- 手掌：`(T, 1, 15, 16)`

两组都使用 FTP-1 已有的 `type="matrix"` 和 `MatrixCNNEncoder`。数据中的传感器
标签保留为 `MoxianTactileGlove460`；这里的 `460` 只是硬件标签，不代表读取或
输出了 460 维张量，实际压力输入始终是上述 320 点。

标准化文件已经按 `palm_then_thumb_index_middle_ring_pinky` 排列为掌区 240 点和
五个指尖各 16 点。因此生产分支只做 reshape：前 240 点变为 `1×15×16`，后
80 点变为 `5×4×4`，点序不变。

FTP-1 转换器不做 baseline、裁剪、插值或空间重排。HDF5 内的
`finger_pressure_60hz`、`region_pressure_vector_60hz` 和
`region_pressure_vector` 都禁止读取，也不读取任何 `(T,460)` 数据。

## 时间处理

批处理默认读取已经逐行对齐的标准化数据根目录：

```text
/personal/xspark_shared/hand_data/hdf5/spark0_real/bench_v5
```

标准化分支严格执行一行输入对应一行输出：

1. 校验 `additional_info/frequency == 30`，并要求源 `timestamps` 严格为 30 Hz；
2. 三路 RGB 在同一行统一调用 `XPolicyLab.utils.process_data.decode_image_bit`，不做 RGB/BGR 通道转换，只缩放到 224；
3. arm/hand state 保持同一行和值，只转为 `float32`；
4. joint-only 生产配方不读取、不写出两侧手腕 EE pose，也不把 EE-pose validity 纳入行验收；
5. 30 Hz 的 320 点触觉保持同一行，只拆成 palm 和 fingertips；
6. 不做最近邻抽帧、插值、clock fit、ZOH、stride sampling 或二次 baseline。

`vision/cam_head/extrinsics` 仍会作为固定相机标定接受一致性检查和 provenance 记录，但 joint-only Zarr
不写 `camera_ego_pose`、`left_wrist_pose`、`right_wrist_pose` 三个数组。FTP-1 固定维度保持 120；上游
loader 自动把右/左 wrist pose、head 和 reserved 共 66 维置零且 mask 为 0，只监督双臂和双手共 54 维。

若请求的 `--target-fps` 不是 30，转换器会直接失败，不会暗中重采样。标准化
validity mask 只在转换阶段用于 fail-closed QC；任一行无效就令该源 episode转换
失败，不会丢行、切段、补值或插值。源 `timestamps` 和全部 `qc_*` 均不写入
训练 Zarr；episode 边界由 `meta/episode_ends` 保存。生产 CLI 和批处理都会拒绝
raw/legacy processed 输入。

## 全量转换

```bash
/mnt/xspark-data/conda_envs/FTP_1/bin/python \
  /personal/xiangpc/0814_Xpolicylab_bench/FTP-1/XPolicyLab/policy/FTP_1/data_scripts/convert_spark_moxian_dataset.py \
  --workers 4 --joint-only-120d
```

每个 episode 先写入 `.staging`，验证通过后才原子重命名为正式 `.zarr`。已存在且验证通过的输出会跳过，因此命令可以安全重跑。

源 HDF5 出现瞬态 `EIO` 或对象存储 `NoSuchKey` 时，单 episode 默认额外重试 2 次，退避为 2 秒、4 秒，并在每次重试前删除该次未完成的 staging。可用 `--source-io-retries` 和 `--source-io-retry-delay-seconds` 调整。重试耗尽仍会记录为 `failed` 并令批处理返回非零；普通缺字段、源文件不存在、格式或 QC 错误不会重试，也不会静默跳过。

批处理同时保存：

- `source_manifest.json`：本次发现的源文件、大小、mtime 与转换参数；
- `conversion_log.jsonl`：逐源文件追加式结果和完整错误堆栈；
- `conversion_summary.json`：全量结束后的成功、跳过和失败数量；
- `metadata/*.json`：每个源文件的逐行映射、固定相机外参、触觉切片和 QC；
- `.conversion.lock`：防止两个批处理进程写入同一输出根目录。

## 全量验收

```bash
/mnt/xspark-data/conda_envs/FTP_1/bin/python \
  /personal/xiangpc/0814_Xpolicylab_bench/FTP-1/XPolicyLab/policy/FTP_1/data_scripts/validate_spark_moxian_dataset.py \
  --workers 8 --joint-only-120d
```

正式数据位于：

```text
/personal/xiangpc/0814_Xpolicylab_bench/FTP-1/data/spark0_real_bench_v5_moxian_joint_only
```

该目录直接包含每个 episode 的 `.zarr`；FTP-1 数据配置中的 `datasets[].path` 应指向这个父目录。
