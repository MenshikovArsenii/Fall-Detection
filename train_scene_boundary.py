from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение YOLO-seg для границ лестницы/эскалатора")
    parser.add_argument("--data", default="data/stair_escalator_seg.yaml", help="YOLO dataset yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Базовая YOLO-seg модель")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0 для GPU, cpu для CPU")
    parser.add_argument("--project", default="runs/scene_boundary")
    parser.add_argument("--name", default="stair_escalator_yoloseg")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Установи ultralytics: pip install ultralytics") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        task="segment",
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("Обучение завершено.")
    print(f"Лучшие веса обычно лежат здесь: {best}")
    print("Скопируй файл в models/stair_escalator_yoloseg.pt")


if __name__ == "__main__":
    main()
