from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass
class ROI:
    polygon: List[Point]

    @classmethod
    def from_config(cls, points: Sequence[Sequence[int]]) -> "ROI":
        polygon = [(int(p[0]), int(p[1])) for p in points]
        return cls(polygon=polygon)

    @property
    def is_valid(self) -> bool:
        return len(self.polygon) >= 3

    def as_np(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)

    def bounding_rect(self) -> tuple[int, int, int, int]:
        if not self.is_valid:
            raise ValueError("ROI polygon must contain at least 3 points")
        x, y, w, h = cv2.boundingRect(self.as_np())
        return int(x), int(y), int(w), int(h)

    def contains_point(self, x: float, y: float) -> bool:
        if not self.is_valid:
            return True
        return cv2.pointPolygonTest(self.as_np(), (float(x), float(y)), False) >= 0

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        if self.is_valid:
            cv2.fillPoly(m, [self.as_np()], 255)
        else:
            m[:, :] = 255
        return m

    def crop(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if not self.is_valid:
            return frame.copy(), (0, 0)
        x, y, w, h = self.bounding_rect()
        crop = frame[y : y + h, x : x + w].copy()
        return crop, (x, y)

    def draw(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        if self.is_valid:
            cv2.polylines(frame, [self.as_np()], isClosed=True, color=color, thickness=2)
        return frame


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def bbox_bottom_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y2))
