"""Load project config and resolve paths relative to the project root."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolve_paths(config)
    return config


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    for key, value in paths.items():
        if value is None:
            continue
        p = Path(value)
        paths[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return config


if __name__ == "__main__":
    cfg = load_config()
    print(cfg["paths"])
