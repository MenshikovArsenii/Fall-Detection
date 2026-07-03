from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class SceneBoundaryResult:
    polygon: list[tuple[int, int]]
    class_name: str = "scene_roi"
    confidence: float = 0.0
    source: str = "unknown"
    mask: Optional[np.ndarray] = None

    @property
    def is_valid(self) -> bool:
        return len(self.polygon) >= 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_polygon": [[int(x), int(y)] for x, y in self.polygon],
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "source": self.source,
        }


class SceneBoundaryDetector:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.method = str(self.cfg.get("method", "auto")).lower()
        self.model_path = Path(str(self.cfg.get("model_path", "models/stair_escalator_yoloseg.pt")))
        self.min_confidence = float(self.cfg.get("min_confidence", 0.25))
        self.target_classes = set(self.cfg.get("classes", ["stairs", "staircase", "escalator"]))
        self.fallback_to_lines = bool(self.cfg.get("fallback_to_lines", True))
        self.padding_px = int(self.cfg.get("padding_px", 12))
        self._model = None

        if self.enabled and self.method in {"auto", "yolo_seg"} and self.model_path.exists():
            self._model = self._load_yolo_model(self.model_path)

    def _load_yolo_model(self, model_path: Path):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Для режима yolo_seg нужна библиотека ultralytics. "
                "Установи её командой: pip install ultralytics"
            ) from exc
        return YOLO(str(model_path))

    def detect(self, frame: np.ndarray) -> Optional[SceneBoundaryResult]:
        if not self.enabled:
            return None

        if self._model is not None:
            result = self._detect_yolo_seg(frame)
            if result and result.is_valid:
                return result

        if self.method in {"auto", "lines"} and self.fallback_to_lines:
            result = self._detect_by_lines(frame)
            if result and result.is_valid:
                return result

        return None

    def _detect_yolo_seg(self, frame: np.ndarray) -> Optional[SceneBoundaryResult]:
        h, w = frame.shape[:2]
        results = self._model.predict(frame, conf=self.min_confidence, verbose=False)
        if not results:
            return None

        r = results[0]
        if r.boxes is None or r.masks is None:
            return None

        names = r.names or {}
        boxes = r.boxes
        masks = r.masks.data.cpu().numpy() if hasattr(r.masks.data, "cpu") else np.asarray(r.masks.data)

        candidates: list[dict[str, Any]] = []
        min_area_ratio = float(self.cfg.get("min_mask_area_ratio", 0.0005))

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
            class_name = str(names.get(cls_id, cls_id))
            if class_name not in self.target_classes:
                continue
            conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
            if conf < self.min_confidence:
                continue

            mask = masks[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            area = float(np.count_nonzero(mask_u8))
            if area < min_area_ratio * float(w * h):
                continue
            candidates.append(
                {
                    "class_name": class_name,
                    "confidence": conf,
                    "mask": mask_u8,
                    "area": area,
                    "score": conf * max(area, 1.0),
                }
            )

        if not candidates:
            return None

        use_all_masks = bool(self.cfg.get("use_all_masks", True))
        max_masks = int(self.cfg.get("max_masks", 5))

        if use_all_masks:
            candidates = sorted(candidates, key=lambda d: float(d["score"]), reverse=True)[:max_masks]
            combined = np.zeros((h, w), dtype=np.uint8)
            for c in candidates:
                combined = cv2.bitwise_or(combined, c["mask"])

            polygon = self._mask_to_polygon_union(combined, padding_px=self.padding_px)
            if len(polygon) < 3:
                return None
            best_conf = max(float(c["confidence"]) for c in candidates)
            return SceneBoundaryResult(
                polygon=polygon,
                class_name=str(candidates[0]["class_name"]),
                confidence=best_conf,
                source=f"yolo_seg_multi:{len(candidates)}",
                mask=combined,
            )

        best = max(candidates, key=lambda d: float(d["score"]))
        polygon = self._mask_to_polygon(best["mask"], padding_px=self.padding_px)
        if len(polygon) < 3:
            return None
        return SceneBoundaryResult(
            polygon=polygon,
            class_name=str(best["class_name"]),
            confidence=float(best["confidence"]),
            source="yolo_seg",
            mask=best["mask"],
        )


    def _mask_to_polygon_union(self, mask: np.ndarray, padding_px: int = 0) -> list[tuple[int, int]]:
        h, w = mask.shape[:2]
        work = mask.copy()
        if padding_px > 0:
            k = max(1, int(padding_px) * 2 + 1)
            kernel = np.ones((k, k), dtype=np.uint8)
            work = cv2.dilate(work, kernel, iterations=1)

        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 100]
        if not contours:
            return []

        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        epsilon = 0.025 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) < 3 or len(approx) > 12:
            x, y, bw, bh = cv2.boundingRect(all_pts)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]

        return self._order_polygon(points)

    def _mask_to_polygon(self, mask: np.ndarray, padding_px: int = 0) -> list[tuple[int, int]]:
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            return []

        if padding_px > 0:
            x, y, bw, bh = cv2.boundingRect(contour)
            x1 = max(0, x - padding_px)
            y1 = max(0, y - padding_px)
            x2 = min(w - 1, x + bw + padding_px)
            y2 = min(h - 1, y + bh + padding_px)
            crop = mask[y1:y2, x1:x2]
            kernel = np.ones((padding_px * 2 + 1, padding_px * 2 + 1), dtype=np.uint8)
            crop = cv2.dilate(crop, kernel, iterations=1)
            expanded = np.zeros_like(mask)
            expanded[y1:y2, x1:x2] = crop
            contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3 or len(approx) > 10:
            hull = cv2.convexHull(contour)
            epsilon = 0.03 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        return self._order_polygon(points)

    def _detect_by_lines(self, frame: np.ndarray) -> Optional[SceneBoundaryResult]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(self.cfg.get("hough_threshold", 70)),
            minLineLength=int(self.cfg.get("min_line_length", max(50, min(w, h) * 0.15))),
            maxLineGap=int(self.cfg.get("max_line_gap", 25)),
        )
        if lines is None:
            return None

        pts: list[tuple[int, int]] = []
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < max(40, min(w, h) * 0.12):
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 8 or angle > 172:
                continue
            pts.extend([(x1, y1), (x2, y2)])

        if len(pts) < 6:
            return None

        arr = np.array(pts, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(arr)
        if bw * bh < 0.04 * w * h:
            return None

        pad = int(self.cfg.get("lines_padding_px", 30))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw + pad)
        y2 = min(h - 1, y + bh + pad)

        hull = cv2.convexHull(arr)
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(polygon) < 3 or len(polygon) > 8:
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        return SceneBoundaryResult(
            polygon=self._order_polygon(polygon),
            class_name="stairs_or_escalator_candidate",
            confidence=0.35,
            source="hough_lines_fallback",
            mask=None,
        )

    def _order_polygon(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(points) < 3:
            return points
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = pts[order].astype(int)
        return [(int(x), int(y)) for x, y in ordered]


def draw_scene_boundary(frame: np.ndarray, result: Optional[SceneBoundaryResult]) -> np.ndarray:
    out = frame.copy()
    if result is None or not result.is_valid:
        return out
    polygon = np.array(result.polygon, dtype=np.int32)
    overlay = out.copy()
    if result.mask is not None and result.mask.shape[:2] == out.shape[:2]:
        m = result.mask > 0
        overlay[m] = (0, 255, 255)
        contours, _ = cv2.findContours(result.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    else:
        cv2.fillPoly(overlay, [polygon], (0, 255, 255))
        cv2.polylines(out, [polygon], True, (0, 255, 255), 2)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    cv2.polylines(out, [polygon], True, (0, 180, 255), 1)
    label = f"{result.class_name} {result.confidence:.2f} [{result.source}]"
    x, y, _, _ = cv2.boundingRect(polygon)
    cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out
