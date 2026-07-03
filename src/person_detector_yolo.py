from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class PersonBox:
    bbox: tuple[float, float, float, float]
    score: float


class YoloPersonDetector:
    """YOLO person detector used before MoveNet.

    It finds person bounding boxes on the original frame. Then MoveNet can be run
    on enlarged person crops instead of the entire CCTV frame.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", False))
        self.model_path = str(cfg.get("model_path", "yolo11n.pt"))
        self.conf = float(cfg.get("confidence", 0.25))
        self.iou = float(cfg.get("iou", 0.50))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = str(cfg.get("device", "cpu"))
        self.max_persons = int(cfg.get("max_persons", 6))
        self.person_class_id = int(cfg.get("person_class_id", 0))
        self.min_bbox_height_px = float(cfg.get("min_bbox_height_px", 20))
        self.min_bbox_area_ratio = float(cfg.get("min_bbox_area_ratio", 0.0003))
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Не удалось импортировать ultralytics. Установи: pip install ultralytics") from exc
        self._model = YOLO(self.model_path)

    def infer(self, frame_bgr: np.ndarray) -> List[PersonBox]:
        if not self.enabled:
            return []
        self.load()
        h, w = frame_bgr.shape[:2]
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.person_class_id],
            verbose=False,
        )
        boxes: list[PersonBox] = []
        if not results:
            return boxes
        r = results[0]
        if r.boxes is None:
            return boxes
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = [float(v) for v in b]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area_ratio = (bw * bh) / max(1.0, float(w * h))
            if bh < self.min_bbox_height_px:
                continue
            if area_ratio < self.min_bbox_area_ratio:
                continue
            boxes.append(PersonBox(bbox=(x1, y1, x2, y2), score=float(c)))
        boxes.sort(key=lambda x: x.score, reverse=True)
        return boxes[: self.max_persons]


def expand_box(
    bbox: tuple[float, float, float, float],
    frame_shape,
    padding_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    xx1 = int(max(0, np.floor(x1 - pad_x)))
    yy1 = int(max(0, np.floor(y1 - pad_y)))
    xx2 = int(min(w, np.ceil(x2 + pad_x)))
    yy2 = int(min(h, np.ceil(y2 + pad_y)))
    if xx2 <= xx1:
        xx2 = min(w, xx1 + 1)
    if yy2 <= yy1:
        yy2 = min(h, yy1 + 1)
    return xx1, yy1, xx2, yy2
