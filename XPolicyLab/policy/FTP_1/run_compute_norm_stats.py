#!/usr/bin/env python3
"""Run FTP-1 normalization with its derived training config finalized.

The pinned upstream normalization entry point parses ``FTP1TrainConfig`` but does
not call ``finalize_config()``.  As a result, command-line values such as the
dataset config path and action down-sampling step do not reach the nested data
factory used by normalization.  Keep that compatibility fix in the XPolicyLab
adapter and delegate the actual computation to the unmodified upstream script.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import openpi.training.config as _config


POLICY_DIR = Path(__file__).resolve().parent


def _upstream_norm_script() -> Path:
    upstream_root = Path(
        os.environ.get("FTP1_UPSTREAM_ROOT", str(POLICY_DIR / "ftp1-policy"))
    ).expanduser()
    script = upstream_root.resolve() / "scripts" / "zarr_compute_norm_stats.py"
    if not script.is_file():
        raise FileNotFoundError(f"FTP-1 normalization script not found: {script}")
    return script


def _load_upstream_norm_module(script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_ftp1_upstream_zarr_compute_norm_stats", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FTP-1 normalization script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finalize_ftp1_config(config: _config.TrainConfig) -> _config.FTP1TrainConfig:
    """Propagate FTP-1 CLI fields into the nested data/model configuration."""
    if not isinstance(config, _config.FTP1TrainConfig):
        raise TypeError(
            "The FTP_1 normalization adapter requires an FTP1TrainConfig; "
            f"received {type(config).__name__}"
        )
    config.finalize_config()
    return config


def main() -> None:
    config = finalize_ftp1_config(_config.cli())
    upstream_norm = _load_upstream_norm_module(_upstream_norm_script())
    upstream_norm.main(config)


if __name__ == "__main__":
    main()
