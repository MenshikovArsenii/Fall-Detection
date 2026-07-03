from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pose_movenet import PoseDetection


@dataclass
class Track:
    track_id: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    missing: int = 0
    history_centers: list[tuple[float, float]] = field(default_factory=list)
    baseline_height: Optional[float] = None
    consecutive_fall: int = 0
    low_state_count: int = 0
    cooldown: int = 0

    def update(self, center: tuple[float, float], bbox: tuple[float, float, float, float]) -> None:
        self.history_centers.append(self.center)
        if len(self.history_centers) > 50:
            self.history_centers = self.history_centers[-50:]
        self.center = center
        self.bbox = bbox
        self.missing = 0
        h = max(1.0, bbox[3] - bbox[1])
        if self.baseline_height is None:
            self.baseline_height = h
        else:
            if h > self.baseline_height:
                self.baseline_height = 0.8 * self.baseline_height + 0.2 * h
            else:
                self.baseline_height = 0.98 * self.baseline_height + 0.02 * h

    @property
    def prev_center(self) -> Optional[tuple[float, float]]:
        return self.history_centers[-1] if self.history_centers else None


class SimpleTracker:
    def __init__(self, max_distance_px: float = 120.0, max_missing_frames: int = 12):
        self.max_distance_px = float(max_distance_px)
        self.max_missing_frames = int(max_missing_frames)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    @staticmethod
    def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def update(self, detections: List[PoseDetection]) -> Dict[int, PoseDetection]:
        assigned: Dict[int, PoseDetection] = {}
        unused_track_ids = set(self.tracks.keys())

        for det in detections:
            center = self.bbox_center(det.bbox)
            best_id = None
            best_dist = float("inf")
            for tid in list(unused_track_ids):
                tr = self.tracks[tid]
                dist = float(np.hypot(center[0] - tr.center[0], center[1] - tr.center[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist <= self.max_distance_px:
                tr = self.tracks[best_id]
                tr.update(center, det.bbox)
                assigned[best_id] = det
                unused_track_ids.remove(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(track_id=tid, center=center, bbox=det.bbox)
                self.tracks[tid].update(center, det.bbox)
                assigned[tid] = det

        for tid in list(unused_track_ids):
            self.tracks[tid].missing += 1
            if self.tracks[tid].missing > self.max_missing_frames:
                del self.tracks[tid]

        return assigned
