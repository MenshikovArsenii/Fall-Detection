from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

from .optical_flow import FlowStats


@dataclass
class FeatureRow:
    frame: int
    timestamp_sec: float
    track_id: int
    in_roi: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_w: float
    bbox_h: float
    bbox_aspect: float
    center_x: float
    center_y: float
    center_speed: float
    avg_keypoint_score: float
    torso_angle_deg: float
    shoulder_hip_height: float
    flow_mean_mag: float
    flow_max_mag: float
    flow_mean_dx: float
    flow_mean_dy: float
    flow_downward_ratio: float
    fall_score: float = 0.0
    raw_label: str = "not_fall"
    label: str = "not_fall"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_angle_deg(dx: float, dy: float) -> float:
    return float(abs(np.degrees(np.arctan2(dy, dx))))


def pose_features(
    frame_idx: int,
    timestamp_sec: float,
    track_id: int,
    keypoints: np.ndarray,
    bbox: Sequence[float],
    in_roi: bool,
    flow: FlowStats,
    prev_center: Optional[tuple[float, float]] = None,
) -> FeatureRow:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if prev_center is None:
        center_speed = 0.0
    else:
        center_speed = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))

    avg_score = float(np.mean(keypoints[:, 2])) if keypoints.size else 0.0

    valid_torso = keypoints[[5, 6, 11, 12], 2] > 0.05
    torso_angle = 90.0
    shoulder_hip_height = 0.0
    if np.sum(valid_torso) >= 3:
        shoulders = keypoints[[5, 6], :2]
        hips = keypoints[[11, 12], :2]
        shoulder_c = shoulders.mean(axis=0)
        hip_c = hips.mean(axis=0)
        dx = hip_c[0] - shoulder_c[0]
        dy = hip_c[1] - shoulder_c[1]
        torso_angle = _safe_angle_deg(dx, dy)
        shoulder_hip_height = float(abs(dy))

    return FeatureRow(
        frame=int(frame_idx),
        timestamp_sec=float(timestamp_sec),
        track_id=int(track_id),
        in_roi=1 if in_roi else 0,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        bbox_w=w,
        bbox_h=h,
        bbox_aspect=float(w / h),
        center_x=cx,
        center_y=cy,
        center_speed=center_speed,
        avg_keypoint_score=avg_score,
        torso_angle_deg=torso_angle,
        shoulder_hip_height=shoulder_hip_height,
        flow_mean_mag=flow.mean_mag,
        flow_max_mag=flow.max_mag,
        flow_mean_dx=flow.mean_dx,
        flow_mean_dy=flow.mean_dy,
        flow_downward_ratio=flow.downward_ratio,
    )


def feature_vector(row: FeatureRow) -> np.ndarray:
    return np.array(
        [
            row.bbox_aspect,
            row.bbox_w,
            row.bbox_h,
            row.center_x,
            row.center_y,
            row.center_speed,
            row.avg_keypoint_score,
            row.torso_angle_deg,
            row.shoulder_hip_height,
            row.flow_mean_mag,
            row.flow_max_mag,
            row.flow_mean_dx,
            row.flow_mean_dy,
            row.flow_downward_ratio,
        ],
        dtype=np.float32,
    )
