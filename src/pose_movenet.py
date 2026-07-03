from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class PoseDetection:
    keypoints: np.ndarray
    bbox: tuple[float, float, float, float]
    score: float

    def map_to_original(self, transform) -> "PoseDetection":
        kps = self.keypoints.copy()
        for i in range(kps.shape[0]):
            x, y = kps[i, 0], kps[i, 1]
            ox, oy = transform.point_to_original(float(x), float(y))
            kps[i, 0], kps[i, 1] = ox, oy
        bbox = transform.bbox_to_original(self.bbox)
        return PoseDetection(keypoints=kps, bbox=bbox, score=self.score)


class MoveNetPoseEstimator:
    def __init__(self, cfg: Dict[str, Any]):
        self.model_url = cfg.get("model_url", "https://tfhub.dev/google/movenet/multipose/lightning/1")
        self.input_size = int(cfg.get("input_size", 256))
        self.min_pose_score = float(cfg.get("min_pose_score", 0.12))
        self.min_keypoint_score = float(cfg.get("min_keypoint_score", 0.15))
        self.max_persons = int(cfg.get("max_persons", 6))
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as exc:
            raise RuntimeError(
                "Не удалось импортировать tensorflow/tensorflow_hub. "
                "Установите зависимости: pip install -r requirements.txt"
            ) from exc
        self.tf = tf
        self._model = hub.load(self.model_url)
        self._infer = self._model.signatures["serving_default"]

    def infer(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        self.load()
        tf = self.tf

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = tf.cast(tf.expand_dims(resized, axis=0), dtype=tf.int32)
        outputs = self._infer(input_tensor)
        raw = outputs["output_0"].numpy()[0]

        detections: list[PoseDetection] = []
        for person in raw[: self.max_persons]:
            if person.shape[0] < 56:
                continue
            pose_score = float(person[55])
            if pose_score < self.min_pose_score:
                continue

            kps = person[:51].reshape((17, 3))
            xy = np.zeros((17, 3), dtype=np.float32)
            xy[:, 0] = kps[:, 1] * w
            xy[:, 1] = kps[:, 0] * h
            xy[:, 2] = kps[:, 2]

            valid = xy[:, 2] >= self.min_keypoint_score
            if np.sum(valid) < 4:
                continue

            ymin, xmin, ymax, xmax = person[51], person[52], person[53], person[54]
            bbox = (float(xmin * w), float(ymin * h), float(xmax * w), float(ymax * h))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                xs = xy[valid, 0]
                ys = xy[valid, 1]
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

            detections.append(PoseDetection(keypoints=xy, bbox=bbox, score=pose_score))

        return detections


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    min_score: float = 0.15,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    for a, b in SKELETON_EDGES:
        if keypoints[a, 2] >= min_score and keypoints[b, 2] >= min_score:
            p1 = (int(round(keypoints[a, 0])), int(round(keypoints[a, 1])))
            p2 = (int(round(keypoints[b, 0])), int(round(keypoints[b, 1])))
            cv2.line(frame, p1, p2, color, 2)
    for x, y, s in keypoints:
        if s >= min_score:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, -1)
    return frame
