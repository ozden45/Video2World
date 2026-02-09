"""
Configuration loading utilities.

Responsible for:
- reading yaml
- merging multiple configs
- converting to dataclasses
- saving final config
"""

from pathlib import Path
import yaml
import os
from dataclasses import asdict
from v2w.config.types import Config


#=================================
# Helper functions
#=================================

def _read_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge dict b into a."""
    out = dict(a)

    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v

    return out


#=================================
# Loaders
#=================================

def load_config(path: str | Path) -> Config:
    """Load single yaml file."""
    
    # Check that path exists
    if isinstance(path, str) and not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    elif isinstance(path, Path) and not path.exists():
        raise FileNotFoundError(f"The path {path} is not found")
        
    raw = _read_yaml(path)
    return Config.from_dict(raw)


def load_many_config(paths: list[str | Path]) -> Config:
    """Load and merge multiple yaml files."""
    merged = {}

    for path in paths:
        # Check that paths exist
        if isinstance(path, str) and not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} is not found")
        elif isinstance(path, Path) and not path.exists():
            raise FileNotFoundError(f"The path {path} is not found")
        
        merged = _deep_merge(merged, _read_yaml(p))

    return Config.from_dict(merged)


def save_config(cfg: Config, path: str | Path):
    """Save config to yaml (for experiment reproducibility)."""
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f)

