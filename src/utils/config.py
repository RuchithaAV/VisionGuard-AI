"""
VisionGuard AI - Centralized YAML Configuration Loader
"""

from pathlib import Path
from typing import Any, Dict
import yaml
DEFAULT_CONFIG_PATH = Path("configs/model.yaml")

def load_config(config_path: str | Path =  DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads and validates the VisionGuard AI YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return cfg