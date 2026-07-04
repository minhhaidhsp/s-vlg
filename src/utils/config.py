"""Load project config and resolve paths relative to the project root.

Two entry points:
  - load_config(path): a single YAML file, unchanged behavior (used directly
    for configs/config.yaml, or any standalone file).
  - load_version_config(version): configs/config.yaml as a base, deep-merged
    with configs/config_{version}.yaml on top (version="v1" -> S-VLG,
    version="v2" -> SU-MedVQA) — see PROJECT_STATE.md for the two-version split.
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_raw(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config = _load_yaml_raw(config_path)
    resolve_paths(config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` on top of `base` (override wins on conflicts,
    nested dicts are merged key-by-key rather than replaced wholesale)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_version_config(
    version: str, base_config_path: str | Path = "configs/config.yaml"
) -> dict[str, Any]:
    """Load the shared base config, then deep-merge the version-specific
    override file on top: configs/config.yaml + configs/config_{version}.yaml.

    Args:
        version: "v1" (S-VLG) or "v2" (SU-MedVQA).
        base_config_path: path to the shared base config.

    Paths are resolved to absolute AFTER merging, so overrides introducing
    new relative paths are handled correctly too.
    """
    base = _load_yaml_raw(base_config_path)
    override_path = Path(base_config_path).parent / f"config_{version}.yaml"
    override_full_path = override_path if override_path.is_absolute() else PROJECT_ROOT / override_path
    override = _load_yaml_raw(override_path) if override_full_path.exists() else {}

    merged = _deep_merge(base, override)
    resolve_paths(merged)
    return merged


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
