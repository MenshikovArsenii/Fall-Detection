from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.scene_boundary import SceneBoundaryDetector, draw_scene_boundary


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматически предложить границы лестницы/эскалатора")
    parser.add_argument("--video", required=True, help="Путь к видео")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    parser.add_argument("--out", default="roi_auto.yaml", help="Куда сохранить ROI")
    parser.add_argument("--preview", default="roi_auto_preview.jpg", help="Картинка с визуализацией ROI")
    parser.add_argument("--frame", type=int, default=None, help="Номер кадра для поиска ROI")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scene_cfg = cfg.get("scene_roi", {})
    scene_cfg["enabled"] = True
    if args.frame is not None:
        scene_cfg["frame_index"] = args.frame

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    frame_index = int(scene_cfg.get("frame_index", 0))
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Не удалось прочитать кадр из видео")

    detector = SceneBoundaryDetector(scene_cfg)
    result = detector.detect(frame)
    if result is None or not result.is_valid:
        raise RuntimeError(
            "Границы лестницы/эскалатора не найдены. "
            "Проверь model_path или используй tools/select_roi.py для ручного ROI."
        )

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(result.to_dict(), f, allow_unicode=True, sort_keys=False)

    cv2.imwrite(args.preview, draw_scene_boundary(frame, result))
    print(f"ROI сохранён: {args.out}")
    print(f"Preview сохранён: {args.preview}")
    print(f"Источник: {result.source}, класс: {result.class_name}, confidence={result.confidence:.3f}")


if __name__ == "__main__":
    main()
