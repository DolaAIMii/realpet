#!/usr/bin/env python3
"""Launch a pet display window from a frames directory."""
import argparse
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Display a pet from processed frames")
    parser.add_argument("--frames-dir", required=True, help="Directory with frame files (JPEG pairs or PNG)")
    parser.add_argument("--fps", type=int, default=24, help="Playback FPS")
    parser.add_argument("--window-size", type=int, nargs=2, default=None,
                        help="Max window size (width height)")
    args = parser.parse_args()

    # Try JPEG pairs first (frame_NNNN.jpg + frame_NNNN_a.jpg), fall back to PNG
    all_jpgs = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))
    jpg_frames = [f for f in all_jpgs if not f.endswith("_a.jpg")]
    if jpg_frames:
        frames = jpg_frames
    else:
        frames = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
    if not frames:
        print(f"No frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    from pipeline.display import display_frames
    window_size = tuple(args.window_size) if args.window_size else None
    display_frames(frames, fps=args.fps, window_size=window_size)


if __name__ == "__main__":
    main()
