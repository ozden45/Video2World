import yaml
from dataclasses import dataclass, fields, is_dataclass
from typing import Type, TypeVar, List
from pathlib import Path

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """
    Base class for all config dataclasses.
    Provides:
      - load from yaml
      - dict → dataclass conversion
    """

    @classmethod
    def load(cls: Type[T], path: str | Path) -> T:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def load_many(cls, paths: List[str] | List[Path]):
        merged = {}
    
        for p in paths:
            with open(Path(p)) as f:
                data = yaml.safe_load(f)
                merged.update(data)   # simple top-level merge
    
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        return _convert(cls, d)


def _convert(cls, d):
    if not is_dataclass(cls):
        return d

    kwargs = {}

    for f in fields(cls):
        value = d[f.name]
        kwargs[f.name] = _convert(f.type, value)

    return cls(**kwargs)
