from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class FlowStats:
    mean_mag: float = 0.0
    max_mag: float = 0.0
    mean_dx: float = 0.0
    mean_dy: float = 0.0
    downward_ratio: float = 0.0


class OpticalFlowExtractor:
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", True))
        self.params = dict(
            pyr_scale=float(cfg.get("pyr_scale", 0.5)),
            levels=int(cfg.get("levels", 3)),
            winsize=int(cfg.get("winsize", 15)),
            iterations=int(cfg.get("iterations", 3)),
            poly_n=int(cfg.get("poly_n", 5)),
            poly_sigma=float(cfg.get("poly_sigma", 1.2)),
            flags=0,
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.last_flow: Optional[np.ndarray] = None

    def update(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.last_flow = None
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.params)
        self.prev_gray = gray
        self.last_flow = flow
        return flow

    def stats_for_bbox(self, bbox: Sequence[float]) -> FlowStats:
        if self.last_flow is None:
            return FlowStats()
        h, w = self.last_flow.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return FlowStats()
        region = self.last_flow[y1:y2, x1:x2]
        dx = region[..., 0]
        dy = region[..., 1]
        mag = np.sqrt(dx * dx + dy * dy)
        if mag.size == 0:
            return FlowStats()
        return FlowStats(
            mean_mag=float(np.mean(mag)),
            max_mag=float(np.max(mag)),
            mean_dx=float(np.mean(dx)),
            mean_dy=float(np.mean(dy)),
            downward_ratio=float(np.mean(dy > 0.5)),
        )
