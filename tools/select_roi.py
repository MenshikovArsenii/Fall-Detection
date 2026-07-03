from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


points: list[tuple[int, int]] = []
window_name = "Select ROI: left click - point, r - reset, s - save, q - quit"


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def draw(frame):
    out = frame.copy()
    for p in points:
        cv2.circle(out, p, 5, (0, 255, 255), -1)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(out, points[i], points[i + 1], (0, 255, 255), 2)
    if len(points) >= 3:
        cv2.line(out, points[-1], points[0], (0, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Выбор ROI по первому кадру видео")
    parser.add_argument("--video", required=True, help="Путь к видео")
    parser.add_argument("--out", default="roi.yaml", help="Куда сохранить roi_polygon")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Не удалось открыть первый кадр: {args.video}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, draw(frame))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("s"):
            if len(points) < 3:
                print("Нужно выбрать минимум 3 точки")
                continue
            data = {"roi_polygon": [[int(x), int(y)] for x, y in points]}
            with Path(args.out).open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            print(f"ROI сохранен: {args.out}")
            print(data)
            break
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
