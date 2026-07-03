from __future__ import annotations

import argparse
import json

from src.autolabeler import FallAutoLabeler


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматическая разметка падений в зоне лестницы/эскалатора")
    parser.add_argument("--config", default="config.yaml", help="Путь к YAML-конфигу")
    args = parser.parse_args()

    labeler = FallAutoLabeler(args.config)
    result = labeler.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
