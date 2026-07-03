from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .features import FeatureRow, feature_vector
from .tracking import Track


class SequenceClassifier:

    def __init__(self, cfg: Dict[str, Any]):
        self.sequence_len = int(cfg.get("sequence", {}).get("length", 20))
        self.min_frames = int(cfg.get("sequence", {}).get("min_frames_for_decision", 8))
        self.lstm_cfg = cfg.get("lstm", {})
        self.rules = cfg.get("rules", {})
        self.post = cfg.get("postprocess", {})
        self.history: Dict[int, Deque[FeatureRow]] = defaultdict(lambda: deque(maxlen=self.sequence_len))
        self.model = None
        self.model_loaded = False
        self.model_input_steps = None
        self.model_input_features = None
        self._warned_lstm_shape = False
        self.lstm_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=64))
        self._try_load_lstm()

    def _try_load_lstm(self) -> None:
        enabled = bool(self.lstm_cfg.get("enabled", True))
        model_path = Path(str(self.lstm_cfg.get("model_path", "models/lstm_fall.keras")))
        if not enabled or not model_path.exists():
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            shape = self.model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            if shape is not None and len(shape) >= 3:
                self.model_input_steps = int(shape[1]) if shape[1] is not None else None
                self.model_input_features = int(shape[2]) if shape[2] is not None else None
            self.model_loaded = True
            print(f"[INFO] LSTM загружена: input_shape={self.model.input_shape}, output_shape={self.model.output_shape}")
        except Exception as exc:
            print(f"[WARN] LSTM не загружена, используется fallback-классификатор: {exc}")
            self.model = None
            self.model_loaded = False

    def _lstm_score(self, rows: List[FeatureRow], lstm_rows: Optional[List[np.ndarray]] = None) -> Optional[float]:
        if not self.model_loaded:
            return None

        expected_steps = self.model_input_steps or self.sequence_len
        expected_features = self.model_input_features

        if lstm_rows and expected_features is not None and len(lstm_rows[-1]) == expected_features:
            seq = np.stack(lstm_rows, axis=0)
        else:
            seq = np.stack([feature_vector(r) for r in rows], axis=0)

        if expected_features is not None and seq.shape[1] != expected_features:
            if not self._warned_lstm_shape:
                print(
                    f"[WARN] LSTM ждёт {expected_features} признаков, "
                    f"а подготовлено {seq.shape[1]}. Используется fallback-классификатор."
                )
                self._warned_lstm_shape = True
            return None

        if len(seq) < expected_steps:
            pad = np.repeat(seq[:1], expected_steps - len(seq), axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        else:
            seq = seq[-expected_steps:]

        x = seq[np.newaxis, :, :].astype(np.float32)
        pred = np.asarray(self.model.predict(x, verbose=0))

        if pred.ndim == 2 and pred.shape[1] >= 2:
            fall_idx = int(self.lstm_cfg.get("fall_class_index", 1))
            fall_idx = max(0, min(fall_idx, pred.shape[1] - 1))
            score = float(pred[0, fall_idx])
        else:
            score = float(np.ravel(pred)[0])
        return score

    def _rule_score(self, rows: List[FeatureRow], track: Track) -> tuple[float, str]:
        if len(rows) < self.min_frames:
            return 0.0, "мало кадров для решения"

        recent = rows[-min(len(rows), self.sequence_len) :]
        last = recent[-1]

        low_height_ratio = float(self.rules.get("low_height_ratio", 0.70))
        horizontal_aspect = float(self.rules.get("horizontal_aspect_ratio", 1.10))
        speed_thr = float(self.rules.get("center_speed_threshold", 12.0))
        flow_thr = float(self.rules.get("flow_mag_threshold", 2.5))
        min_kp = float(self.rules.get("min_avg_keypoint_score", 0.18))

        baseline_h = track.baseline_height or last.bbox_h
        current_ratio = last.bbox_h / max(1.0, baseline_h)

        max_speed = max(r.center_speed for r in recent)
        max_flow = max(r.flow_mean_mag for r in recent)
        max_down_flow = max(r.flow_mean_dy for r in recent)
        aspect_now = last.bbox_aspect
        torso_now = last.torso_angle_deg
        avg_kp = last.avg_keypoint_score

        confidence_factor = min(1.0, max(0.35, avg_kp / max(0.01, min_kp)))

        sudden_motion = max_speed >= speed_thr or max_flow >= flow_thr
        low_state = current_ratio <= low_height_ratio or last.center_speed < 2 and aspect_now >= horizontal_aspect
        horizontal_state = aspect_now >= horizontal_aspect or torso_now <= 45.0
        downward_motion = max_down_flow > 0.8 or any(r.flow_downward_ratio > 0.35 for r in recent)

        score = 0.0
        reasons = []
        if sudden_motion:
            score += 0.30
            reasons.append("резкое движение")
        if low_state:
            score += 0.30
            reasons.append("низкое положение")
        if horizontal_state:
            score += 0.25
            reasons.append("горизонтальная/наклонная поза")
        if downward_motion:
            score += 0.15
            reasons.append("движение вниз")

        score *= confidence_factor
        return float(min(1.0, score)), ", ".join(reasons) if reasons else "признаки падения не выражены"

    def decide(self, row: FeatureRow, track: Track, lstm_vector: Optional[np.ndarray] = None) -> FeatureRow:
        if not row.in_roi:
            row.raw_label = "ignore"
            row.label = "ignore"
            row.fall_score = 0.0
            row.reason = "вне ROI"
            return row

        self.history[row.track_id].append(row)
        rows = list(self.history[row.track_id])
        if lstm_vector is not None:
            self.lstm_history[row.track_id].append(np.asarray(lstm_vector, dtype=np.float32))
        lstm_rows = list(self.lstm_history[row.track_id])

        lstm_score = self._lstm_score(rows, lstm_rows)
        if lstm_score is None:
            score, reason = self._rule_score(rows, track)
        else:
            rule_score, rule_reason = self._rule_score(rows, track)
            score = 0.75 * lstm_score + 0.25 * rule_score
            reason = f"LSTM={lstm_score:.2f}; {rule_reason}"

        threshold = float(self.lstm_cfg.get("threshold", 0.65))
        candidate = score >= threshold

        baseline_h = track.baseline_height or row.bbox_h
        low_ratio = row.bbox_h / max(1.0, baseline_h)
        low_state = (low_ratio <= float(self.rules.get("low_height_ratio", 0.70))) or (
            row.bbox_aspect >= float(self.rules.get("horizontal_aspect_ratio", 1.10))
        )
        if low_state:
            track.low_state_count += 1
        else:
            track.low_state_count = max(0, track.low_state_count - 1)

        if candidate:
            track.consecutive_fall += 1
        else:
            track.consecutive_fall = 0

        required_low = int(self.post.get("require_low_state_frames", 5))
        required_fall = int(self.post.get("require_consecutive_fall_frames", 4))

        if candidate and track.consecutive_fall >= required_fall and track.low_state_count >= required_low:
            row.raw_label = "fall"
            row.label = "fall"
            row.reason = reason + "; подтверждено несколькими кадрами"
        elif candidate:
            row.raw_label = "fall_candidate"
            row.label = "not_fall"
            row.reason = reason + "; недостаточно подтверждения"
        else:
            row.raw_label = "not_fall"
            row.label = "not_fall"
            row.reason = reason

        row.fall_score = score
        return row
