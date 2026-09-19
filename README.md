# FTP-1_Xpolicylab

XPolicyLab integration for FTP-1 on the Spark0 real-robot `bench_v5` dataset.

Source provenance:

- XPolicyLab: `3dddc0f0305b78e0d28bdc53e14fbdcf59b5cd08`
- FTP-1 upstream: `89fa681d6c014cce28300946b7526db808e0b1c1`
- Local FTP-1 adapter and training fixes are vendored under `XPolicyLab/policy/FTP_1`.

Large artifacts are intentionally not tracked. On H20 they live at:

- raw HDF5: `/vepfs-cnbje63de6fae220/xiangpc/data/bench_v5`
- converted Zarr: `data/spark0_real_bench_v5_moxian_joint_only`
- pretrained weights: `pretrain_model/ftp1_pretrain_v0426_50kstep`

See `XPolicyLab/policy/FTP_1/README.md` for conversion, training, and evaluation details.
