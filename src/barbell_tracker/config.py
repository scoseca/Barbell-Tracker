from copy import deepcopy
from pathlib import Path


def load_pipeline_config(config_path):
    """Load a YAML pipeline configuration file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load configuration files. "
            "Install it with `pip install PyYAML` or from your conda environment."
        ) from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def apply_config_overrides(config, overrides):
    """Return a copy of config with selected nested values replaced."""
    updated = deepcopy(config)
    for path, value in overrides.items():
        if value is None:
            continue
        _set_nested(updated, path, value)
    return updated


def _set_nested(config, path, value):
    current = config
    keys = path.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value
