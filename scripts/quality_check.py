#!/usr/bin/env python3
"""Standalone QC gate — runs check_quality and emits a single JSON result.

Usage:
    python scripts/quality_check.py --video VIDEO_PATH --output-dir DIR

Outputs one JSON line:
    {"type":"qc", "passed":true}
    {"type":"qc", "passed":false, "reason":"no_pet", "message":"未检测到宠物…"}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.track_then_matte import check_quality  # noqa: E402  # requires sys.path setup above


def extract_frames(video_path, num_frames=6):
    """Extract evenly-spaced frames using a single ffmpeg concat pass.

    Writes a concat script that seeks to each timestamp, then runs one ffmpeg
    process to decode all segments and extract one frame each.  6 frames needed
    for blur detection (requires >=4) and multi-frame pet scanning (late-entry).
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(probe.stdout.strip())
    except Exception:
        return []

    tmpdir = tempfile.mkdtemp(prefix="qc_")
    step = duration / (num_frames + 1)

    # Build concat demuxer script — one segment per desired frame
    script_path = os.path.join(tmpdir, "concat.txt")
    with open(script_path, "w") as f:
        for i in range(num_frames):
            t = step * (i + 1)
            f.write(f"file '{video_path}'\n")
            f.write(f"inpoint {t:.3f}\n")
            f.write("duration 0.04\n")  # ~1 frame at 30fps

    out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", script_path,
             "-vsync", "vfr", "-frames:v", str(num_frames),
             "-q:v", "2", out_pattern],
            capture_output=True, timeout=15
        )
    except Exception:
        pass
    frames = sorted(
        os.path.join(tmpdir, f)
        for f in os.listdir(tmpdir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    return frames


def run_qc(video, output_dir, emit=None):
    """Core QC logic — callable from daemon or CLI.

    Args:
        video: path to video file
        output_dir: output directory (unused, kept for API consistency)
        emit: optional callback for streaming JSON lines (unused for QC,
              kept for interface symmetry with run_detect/run_pipeline)

    Returns:
        dict with "type":"qc", "passed":bool, and optionally "reason"/"message"
    """
    frames = extract_frames(video)
    qa = check_quality(video, frames)

    result = qa.to_dict()
    output = {"type": "qc", "passed": result["passed"]}
    if not result["passed"]:
        first = result["issues"][0] if result["issues"] else {}
        output["reason"] = first.get("code", "unknown")
        output["message"] = first.get("message", "素材不合格")

    # Cleanup temp frames
    if frames:
        shutil.rmtree(os.path.dirname(frames[0]), ignore_errors=True)

    return output


def main():
    parser = argparse.ArgumentParser(description="Quality gate for realpet videos")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", required=True, help="Output directory (unused, kept for API consistency)")
    args = parser.parse_args()

    output = run_qc(args.video, args.output_dir)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
