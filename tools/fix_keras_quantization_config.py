from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove Keras quantization_config keys from a .keras file.")
    parser.add_argument("--src", required=True, help="Input .keras file")
    parser.add_argument("--dst", required=True, help="Output fixed .keras file")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "config.json":
                cfg = json.loads(data.decode("utf-8"))
                remove_quantization_config(cfg)
                data = json.dumps(cfg, ensure_ascii=False).encode("utf-8")
            zout.writestr(info, data)

    print(f"Saved fixed model: {dst}")


if __name__ == "__main__":
    main()
