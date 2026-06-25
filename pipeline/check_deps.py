#!/usr/bin/env python3
"""Check if all dependencies are installed. Output JSON to stdout."""
import json
import shutil
import sys


def main():
    missing = []
    for mod in ["rembg", "PIL", "numpy", "scipy"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    coreml_available = False
    if "rembg" not in missing:
        try:
            import onnxruntime as ort
            coreml_available = "CoreMLExecutionProvider" in ort.get_available_providers()
        except Exception:
            pass

    ffmpeg_ok = shutil.which("ffmpeg") is not None

    json.dump({
        "ok": len(missing) == 0 and ffmpeg_ok,
        "missing": missing,
        "ffmpeg": ffmpeg_ok,
        "coreml": coreml_available,
    }, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
