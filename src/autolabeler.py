from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .classifier import SequenceClassifier
from .config import load_config
from .features import pose_features
from .optical_flow import OpticalFlowExtractor
from .pose_movenet import MoveNetPoseEstimator, PoseDetection
from .person_detector_yolo import YoloPersonDetector, expand_box
from .preprocessing import prepare_frame
from .scene_boundary import SceneBoundaryDetector, draw_scene_boundary
from .roi import ROI, bbox_bottom_center, bbox_center
from .tracking import SimpleTracker
from .visualization import draw_result

def keypoints_to_lstm_vector(keypoints: np.ndarray, frame_width: int, frame_height: int, order: str = "yxs") -> np.ndarray:
    """Return 51 MoveNet-style features: 17 keypoints * (coord1, coord2, score).

    Many MoveNet LSTM models are trained on normalized keypoints in the order
    y, x, confidence. The order can be changed in config if your model was
    trained differently.
    """
    if keypoints is None or keypoints.size == 0:
        return np.zeros((51,), dtype=np.float32)
    kps = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
    x = np.clip(kps[:, 0] / max(1.0, float(frame_width)), 0.0, 1.0)
    y = np.clip(kps[:, 1] / max(1.0, float(frame_height)), 0.0, 1.0)
    score = np.clip(kps[:, 2], 0.0, 1.0)
    if order == "xys":
        arr = np.stack([x, y, score], axis=1)
    else:
        arr = np.stack([y, x, score], axis=1)
    return arr.reshape(-1).astype(np.float32)



class FallAutoLabeler:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self.video_path = Path(str(self.cfg.get("video_path", "")))
        self.output_dir = Path(str(self.cfg.get("output_dir", "output")))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.roi = ROI.from_config(self.cfg.get("roi_polygon", []))
        self.scene_roi_cfg = self.cfg.get("scene_roi", {})
        self.scene_roi_result = None
        self.scene_detector = None
        self.dynamic_scene_roi = bool(self.scene_roi_cfg.get("per_frame", False))
        self.scene_update_every = max(1, int(self.scene_roi_cfg.get("update_every_n_frames", 1)))
        self.scene_smooth_alpha = float(self.scene_roi_cfg.get("smooth_alpha", 0.65))
        if bool(self.scene_roi_cfg.get("enabled", False)):
            self.scene_detector = SceneBoundaryDetector(self.scene_roi_cfg)
        self.pose = MoveNetPoseEstimator(self.cfg.get("pose", {}))
        self.person_detector_cfg = self.cfg.get("person_detector", {})
        self.person_detector = YoloPersonDetector(self.person_detector_cfg) if bool(self.person_detector_cfg.get("enabled", False)) else None
        self.flow = OpticalFlowExtractor(self.cfg.get("optical_flow", {}))
        tr_cfg = self.cfg.get("tracking", {})
        self.tracker = SimpleTracker(
            max_distance_px=float(tr_cfg.get("max_distance_px", 120)),
            max_missing_frames=int(tr_cfg.get("max_missing_frames", 12)),
        )
        self.classifier = SequenceClassifier(self.cfg)

        zone_cfg = self.cfg.get("zone_membership", {})
        self.roi_memory_frames = int(zone_cfg.get("roi_memory_frames", 8))
        self.bottom_points = int(zone_cfg.get("bottom_points", 5))
        self._roi_memory: dict[int, int] = {}

    def _point_in_active_roi(self, x: float, y: float) -> bool:
        if self.scene_roi_result is not None and getattr(self.scene_roi_result, "mask", None) is not None:
            mask = self.scene_roi_result.mask
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                return bool(mask[yi, xi] > 0)
            return False
        if not self.roi.is_valid:
            return True
        return self.roi.contains_point(float(x), float(y))

    def _bbox_bottom_in_roi(self, bbox) -> bool:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if not self.roi.is_valid and self.scene_roi_result is None:
            return True

        n = max(1, self.bottom_points)
        if n == 1:
            xs = [(x1 + x2) / 2.0]
        else:
            xs = np.linspace(x1, x2, n)

        return any(self._point_in_active_roi(float(x), y2) for x in xs)

    def _track_in_roi_with_memory(self, track_id: int, bbox) -> bool:
        bottom_in_roi = self._bbox_bottom_in_roi(bbox)
        current_memory = int(self._roi_memory.get(track_id, 0))

        if bottom_in_roi:
            self._roi_memory[track_id] = self.roi_memory_frames
            return True

        if current_memory > 0:
            self._roi_memory[track_id] = current_memory - 1
            return True

        self._roi_memory[track_id] = 0
        return False

    def _filter_detections_in_roi(self, detections: List[PoseDetection]) -> List[PoseDetection]:
        return detections

    def _detect_poses_with_yolo_person(self, frame: np.ndarray) -> List[PoseDetection]:
        """YOLO-person -> crop человека -> MoveNet.

        Это помогает на CCTV: человек может быть маленьким на полном кадре,
        а после YOLO-crop MoveNet получает увеличенный фрагмент с человеком.
        """
        if self.person_detector is None:
            return []

        boxes = self.person_detector.infer(frame)
        if not boxes:
            return []

        padding = float(self.person_detector_cfg.get("crop_padding_ratio", 0.30))
        min_pose_score = float(self.cfg.get("pose", {}).get("min_pose_score", 0.12))
        detections: list[PoseDetection] = []

        for pb in boxes:
            x1, y1, x2, y2 = expand_box(pb.bbox, frame.shape, padding_ratio=padding)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_dets = self.pose.infer(crop)
            if not crop_dets:
                continue

            # Берём лучший скелет внутри YOLO-bbox.
            best = max(crop_dets, key=lambda d: d.score)
            if best.score < min_pose_score:
                continue

            kps = best.keypoints.copy()
            kps[:, 0] += float(x1)
            kps[:, 1] += float(y1)

            # Для стабильности используем bbox от YOLO, а не от MoveNet.
            detections.append(PoseDetection(keypoints=kps, bbox=pb.bbox, score=max(float(best.score), float(pb.score))))

        return detections


    def _smooth_polygon(self, old_polygon, new_polygon):
        """Сглаживает ROI между кадрами, чтобы граница не дрожала на preview."""
        if not old_polygon or not new_polygon or len(old_polygon) != len(new_polygon):
            return new_polygon
        a = float(np.clip(self.scene_smooth_alpha, 0.0, 1.0))
        out = []
        for (ox, oy), (nx, ny) in zip(old_polygon, new_polygon):
            x = int(round(a * float(ox) + (1.0 - a) * float(nx)))
            y = int(round(a * float(oy) + (1.0 - a) * float(ny)))
            out.append((x, y))
        return out

    def _update_scene_roi_for_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Обновляет границы лестницы/эскалатора.

        Если scene_roi.per_frame=true, YOLO-seg запускается регулярно.
        Это нужно для видео с меняющимся ракурсом или движущейся камерой.
        Если модель на кадре ничего не нашла, используется последняя нормальная ROI.
        """
        if self.scene_detector is None:
            return
        if (frame_idx - 1) % self.scene_update_every != 0:
            return

        result = self.scene_detector.detect(frame)
        if result is None or not result.is_valid:
            return

        if self.scene_roi_result is not None:
            result.polygon = self._smooth_polygon(self.scene_roi_result.polygon, result.polygon)

        self.scene_roi_result = result
        self.roi = ROI.from_config(result.polygon)

    def _bbox_roi_intersection_ratio(self, bbox) -> float:
        """Доля bbox, которая пересекается с ROI. Нужна для фильтра ложных скелетов."""
        if not self.roi.is_valid and self.scene_roi_result is None:
            return 1.0
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return 0.0
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)

        if self.scene_roi_result is not None and getattr(self.scene_roi_result, "mask", None) is not None:
            full_mask = self.scene_roi_result.mask
            yy2 = min(y2, full_mask.shape[0])
            xx2 = min(x2, full_mask.shape[1])
            if yy2 <= y1 or xx2 <= x1:
                return 0.0
            crop = full_mask[y1:yy2, x1:xx2]
            return float(np.count_nonzero(crop)) / float(max(1, crop.size))

        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        roi_poly = np.array([(int(px - x1), int(py - y1)) for px, py in self.roi.polygon], dtype=np.int32)
        if len(roi_poly) < 3:
            return 1.0
        cv2.fillPoly(mask, [roi_poly], 1)
        return float(mask.sum()) / float(mask.size)

    def _is_good_pose_detection(self, det: PoseDetection, frame_shape) -> bool:
        """Удаляет лишние скелеты MoveNet: слабые, слишком маленькие и явно вне зоны."""
        fcfg = self.cfg.get("pose_filter", {})
        if not bool(fcfg.get("enabled", True)):
            return True

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in det.bbox]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / max(1.0, float(w * h))

        kps = np.asarray(det.keypoints, dtype=np.float32).reshape(17, 3)
        min_kp_score = float(self.cfg.get("pose", {}).get("min_keypoint_score", 0.15))
        valid_scores = kps[:, 2] >= min_kp_score
        valid_count = int(valid_scores.sum())
        avg_score = float(kps[:, 2].mean())

        if valid_count < int(fcfg.get("min_valid_keypoints", 6)):
            return False
        if avg_score < float(fcfg.get("min_avg_keypoint_score", 0.12)):
            return False
        if bh < float(fcfg.get("min_bbox_height_px", 25)):
            return False
        if area_ratio < float(fcfg.get("min_bbox_area_ratio", 0.0008)):
            return False
        if area_ratio > float(fcfg.get("max_bbox_area_ratio", 0.60)):
            return False

        if bool(fcfg.get("keep_only_roi_candidates", True)) and self.roi.is_valid:
            bottom_ok = self._bbox_bottom_in_roi(det.bbox)
            inter_ok = self._bbox_roi_intersection_ratio(det.bbox) >= float(fcfg.get("min_roi_intersection_ratio", 0.01))
            if not (bottom_ok or inter_ok):
                return False

        return True

    def run(self) -> dict:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if self.scene_detector is not None and not self.dynamic_scene_roi:
            roi_frame_index = int(self.scene_roi_cfg.get("frame_index", 0))
            if roi_frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, roi_frame_index)
            ok_roi, roi_frame = cap.read()
            if ok_roi:
                result = self.scene_detector.detect(roi_frame)
                if result is not None and result.is_valid:
                    self.scene_roi_result = result
                    self.roi = ROI.from_config(result.polygon)
                    roi_yaml = self.output_dir / f"{self.video_path.stem}_auto_scene_roi.yaml"
                    with roi_yaml.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(result.to_dict(), f, allow_unicode=True, sort_keys=False)
                    roi_preview = self.output_dir / f"{self.video_path.stem}_auto_scene_roi.jpg"
                    cv2.imwrite(str(roi_preview), draw_scene_boundary(roi_frame, result))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        preview_cfg = self.cfg.get("preview", {})
        save_preview = bool(preview_cfg.get("save_video", True))
        preview_path = self.output_dir / f"{self.video_path.stem}_preview.mp4"
        writer = None
        if save_preview:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_fps = preview_cfg.get("fps") or fps
            writer = cv2.VideoWriter(str(preview_path), fourcc, float(writer_fps), (width, height))

        rows: list[dict] = []
        frame_idx = 0

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Auto labeling")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = frame_idx / fps if fps else 0.0

            if self.dynamic_scene_roi:
                self._update_scene_roi_for_frame(frame, frame_idx)

            self.flow.update(frame)

            if self.person_detector is not None:
                detections = self._detect_poses_with_yolo_person(frame)
                if not detections:
                    work_frame, transform = prepare_frame(frame, self.roi, self.cfg.get("preprocess", {}))
                    detections = [d.map_to_original(transform) for d in self.pose.infer(work_frame)]
            else:
                work_frame, transform = prepare_frame(frame, self.roi, self.cfg.get("preprocess", {}))
                detections = [d.map_to_original(transform) for d in self.pose.infer(work_frame)]

            detections = [d for d in detections if self._is_good_pose_detection(d, frame.shape)]
            assigned = self.tracker.update(detections)

            frame_rows = []
            for track_id, det in assigned.items():
                track = self.tracker.tracks[track_id]
                in_roi = self._track_in_roi_with_memory(track_id, det.bbox)
                flow_stats = self.flow.stats_for_bbox(det.bbox)
                row = pose_features(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp,
                    track_id=track_id,
                    keypoints=det.keypoints,
                    bbox=det.bbox,
                    in_roi=in_roi,
                    flow=flow_stats,
                    prev_center=track.prev_center,
                )
                lstm_vec = keypoints_to_lstm_vector(
                    det.keypoints,
                    frame_width=width,
                    frame_height=height,
                    order=str(self.cfg.get("lstm", {}).get("keypoint_order", "yxs")),
                )
                row = self.classifier.decide(row, track, lstm_vector=lstm_vec)
                rows.append(row.to_dict())
                frame_rows.append(row)

            if writer is not None:
                vis = draw_result(
                    frame,
                    self.roi,
                    frame_rows,
                    assigned,
                    min_keypoint_score=float(self.cfg.get("pose", {}).get("min_keypoint_score", 0.15)),
                    draw_roi=bool(preview_cfg.get("draw_roi", True)),
                    draw_kps=bool(preview_cfg.get("draw_keypoints", True)),
                    scene_result=self.scene_roi_result,
                )
                writer.write(vis)

            pbar.update(1)

        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

        out_csv = self.output_dir / f"{self.video_path.stem}_annotations.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

        if self.scene_roi_result is not None:
            roi_yaml = self.output_dir / f"{self.video_path.stem}_auto_scene_roi_last.yaml"
            with roi_yaml.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.scene_roi_result.to_dict(), f, allow_unicode=True, sort_keys=False)

        return {
            "video": str(self.video_path),
            "annotations_csv": str(out_csv),
            "preview_video": str(preview_path) if save_preview else None,
            "frames_processed": frame_idx,
            "rows": len(rows),
        }
