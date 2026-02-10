from pathlib import Path
import os


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge dict b into a."""
    out = dict(a)

    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v

    return out


def if_path_exists(path: str | Path) -> bool:
    if isinstance(path, str) and not os.path.exists(path):
        return False
    elif isinstance(path, Path) and not path.exists():
        return False
    
    return True