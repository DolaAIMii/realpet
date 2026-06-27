#!/usr/bin/env python3
"""CLI entry point for Swift app integration. Outputs JSON Lines to stdout."""
import argparse
import json
import os
import shutil
import signal
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def emit(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(130))

    parser = argparse.ArgumentParser(description="Live Photo Pet processor")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=24, help="Frame extraction FPS")
    parser.add_argument("--size", type=int, default=None,
                        help="Max dimension for final frames (auto if not set)")
    parser.add_argument("--use-coreml", action="store_true",
                        help="Use CoreML acceleration for segmentation")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove intermediate files after processing")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        emit({"type": "error", "message": f"Video not found: {args.video}"})
        sys.exit(1)

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        emit({"type": "error", "message": "ffmpeg not found. Install with: brew install ffmpeg"})
        sys.exit(1)

    # Check dependencies
    try:
        import torch  # noqa: F401
        import PIL  # noqa: F401
        import numpy  # noqa: F401
        import cv2  # noqa: F401
        from transformers import AutoModelForImageSegmentation  # noqa: F401
    except ImportError as e:
        emit({"type": "error", "message": f"Missing dependency: {e}"})
        sys.exit(1)

    target_size = (args.size, args.size) if args.size else None

    try:
        from pipeline.ai_animate import process_video_to_frames

        last_emit = [0.0]

        def progress_cb(phase, current, total, detail):
            now = time.time()
            # Throttle progress emits to max 10/sec
            if now - last_emit[0] < 0.1 and phase != "phase":
                return
            last_emit[0] = now
            if phase == "phase":
                emit({"type": "phase", "name": detail, "total": 0})
            elif phase == "fix":
                emit({
                    "type": "progress",
                    "phase": "fix",
                    "current": current,
                    "total": total,
                    "detail": detail,
                })
            else:
                emit({
                    "type": "progress",
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "detail": detail,
                })

        t0 = time.time()
        final_paths = process_video_to_frames(
            args.video, args.output_dir,
            fps=args.fps, target_size=target_size,
            use_coreml=args.use_coreml,
            progress_callback=progress_cb,
        )

        if not final_paths:
            emit({"type": "error", "message": "No frames generated"})
            sys.exit(1)

        # Cleanup intermediate files
        if args.cleanup:
            for d in ["extracted", "segmented"]:
                dpath = os.path.join(args.output_dir, d)
                if os.path.isdir(dpath):
                    shutil.rmtree(dpath)

        final_dir = os.path.join(args.output_dir, "final")
        emit({
            "type": "complete",
            "frames_dir": os.path.abspath(final_dir),
            "frame_count": len(final_paths),
            "fps": args.fps,
            "elapsed": round(time.time() - t0, 1),
        })

    except KeyboardInterrupt:
        emit({"type": "error", "message": "cancelled"})
        sys.exit(130)
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
