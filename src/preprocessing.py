from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .roi import ROI


@dataclass
class FrameTransform:
    offset_x: int = 0
    offset_y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def point_to_original(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale_x + self.offset_x, y / self.scale_y + self.offset_y

    def bbox_to_original(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        ox1, oy1 = self.point_to_original(x1, y1)
        ox2, oy2 = self.point_to_original(x2, y2)
        return ox1, oy1, ox2, oy2


def enhance_frame(frame: np.ndarray, clahe: bool = True, denoise: bool = False) -> np.ndarray:
    out = frame.copy()
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return out


def prepare_frame(frame: np.ndarray, roi: ROI, cfg: Dict[str, Any]) -> tuple[np.ndarray, FrameTransform]:
    crop_to_roi = bool(cfg.get("crop_to_roi", True))
    resize_width = cfg.get("resize_width", None)

    if crop_to_roi:
        work, (ox, oy) = roi.crop(frame)
    else:
        work, (ox, oy) = frame.copy(), (0, 0)

    original_h, original_w = work.shape[:2]
    sx = sy = 1.0

    if resize_width:
        resize_width = int(resize_width)
        if original_w > 0 and original_w != resize_width:
            scale = resize_width / float(original_w)
            new_h = max(1, int(round(original_h * scale)))
            work = cv2.resize(work, (resize_width, new_h), interpolation=cv2.INTER_LINEAR)
            sx = sy = scale

    work = enhance_frame(
        work,
        clahe=bool(cfg.get("clahe", True)),
        denoise=bool(cfg.get("denoise", False)),
    )
    return work, FrameTransform(offset_x=ox, offset_y=oy, scale_x=sx, scale_y=sy)
