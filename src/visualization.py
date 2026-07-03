from __future__ import annotations

import cv2
import numpy as np

from .features import FeatureRow
from .pose_movenet import draw_keypoints
from .roi import ROI
try:
    from .scene_boundary import draw_scene_boundary
except Exception: 
    draw_scene_boundary = None


def draw_result(
    frame: np.ndarray,
    roi: ROI,
    rows: list[FeatureRow],
    detections_by_track: dict[int, object],
    min_keypoint_score: float = 0.15,
    draw_roi: bool = True,
    draw_kps: bool = True,
    scene_result=None,
) -> np.ndarray:
    out = frame.copy()

    if draw_roi:
        if scene_result is not None and draw_scene_boundary is not None:
            out = draw_scene_boundary(out, scene_result)
        else:
            roi.draw(out)

    for row in rows:
        x1, y1, x2, y2 = map(int, [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2])
        if row.label == "fall":
            color = (0, 0, 255)
        elif row.label == "ignore":
            color = (128, 128, 128)
        elif row.raw_label == "fall_candidate":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"id={row.track_id} {row.label} score={row.fall_score:.2f}"
        cv2.putText(out, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        det = detections_by_track.get(row.track_id)
        if draw_kps and det is not None:
            draw_keypoints(out, det.keypoints, min_keypoint_score, color=color)

    return out
