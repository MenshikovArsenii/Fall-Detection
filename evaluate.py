from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.metrics import evaluate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка автоматической разметки по ручной разметке")
    parser.add_argument("--pred", required=True, help="CSV с автоматической разметкой")
    parser.add_argument("--gt", required=True, help="CSV с ручной разметкой")
    args = parser.parse_args()

    result = evaluate_csv(args.pred, args.gt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
