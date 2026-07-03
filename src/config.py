from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("output_dir", "output")
    cfg.setdefault("roi_polygon", [])
    cfg.setdefault("scene_roi", {})
    cfg.setdefault("preprocess", {})
    cfg.setdefault("pose", {})
    cfg.setdefault("optical_flow", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("sequence", {})
    cfg.setdefault("lstm", {})
    cfg.setdefault("postprocess", {})
    cfg.setdefault("rules", {})
    cfg.setdefault("preview", {})

    return cfg
