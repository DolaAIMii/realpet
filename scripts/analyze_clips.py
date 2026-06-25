#!/usr/bin/env python3
"""Clip candidate analysis for the segment-selection UI.

For videos longer than the processing budget, the pipeline used to silently
auto-pick the single best-scoring segment — which often missed the moment the
user actually wanted. This script surfaces the choice instead: it returns the
top recommended segments (with a thumbnail each) so the frontend can let the
user pick where the pet clip starts.

Short videos (<= max-seconds) need no selection — the whole clip is used.

Usage:
  python scripts/analyze_clips.py --video V --output-dir D [--max-seconds 15]

Output (one JSON line):
  {"type": "clips", "needs_selection": true, "duration": 42.0, "max_seconds": 15,
   "candidates": [
     {"index": 0, "start": 3.0, "end": 18.0, "duration": 15.0, "score": 0.82,
      "recommended": true, "thumb": "/.../clip_0.jpg", "label": "0:03–0:18"},
     ...]}
"""
import argparse
import json
import os
import subprocess
import sys

import cv2


def emit(msg):
    print(json.dumps(msg), flush=True)


def _video_duration(video_path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True)
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return 0.0


def _thumb(video_path, t, out_path):
    """Grab one JPEG at time t (seconds). Returns path or ''."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(max(0.0, t)), "-i", video_path,
         "-vframes", "1", "-vf", "scale=640:-1", "-q:v", "3", out_path],
        capture_output=True)
    return out_path if os.path.exists(out_path) else ""


def _video_aspect(video_path, out_dir):
    """Probe video aspect ratio (width/height) from first frame. Default 16/9."""
    probe = _thumb(video_path, 0.0, os.path.join(out_dir, "_probe.jpg"))
    if probe:
        im = cv2.imread(probe)
        if im is not None:
            h, w = im.shape[:2]
            if h > 0:
                return round(w / h, 4)
    return round(16.0 / 9.0, 4)


def _fmt(t):
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def _filmstrip(video_path, duration, out_dir, n=10):
    """Evenly-spaced thumbnails across the whole video, for the start scrubber.

    Lets the user roam to ANY start (e.g. the front of the clip) instead of
    being limited to the clustered recommendations.
    """
    strip = []
    step = duration / n
    for i in range(n):
        t = i * step
        p = _thumb(video_path, t, os.path.join(out_dir, f"strip_{i}.jpg"))
        if p:
            strip.append({"t": round(t, 2), "thumb": p, "label": _fmt(t)})
    return strip


def main():
    ap = argparse.ArgumentParser(description="Clip candidate analysis")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    duration = _video_duration(args.video)
    max_s = args.max_seconds
    aspect = _video_aspect(args.video, args.output_dir)

    # Short enough → no selection needed, whole clip is used.
    if duration <= 0 or duration <= max_s:
        emit({"type": "clips", "needs_selection": False,
              "duration": duration, "max_seconds": max_s, "aspect": aspect,
              "candidates": [{
                  "index": 0, "start": 0.0,
                  "end": duration, "duration": duration, "score": 1.0,
                  "recommended": True, "thumb": "",
                  "label": f"0:00–{_fmt(duration)}"}]})
        return

    # Long video → rank segments and surface the top candidates.
    candidates = []
    try:
        from pipeline.smart_clip import analyze_video
        result = analyze_video(args.video, target_duration=max_s,
                               min_duration=min(6.0, max_s),
                               max_duration=max_s)
        cands = result.all_candidates or [result.recommended_clip]
        best_start = result.recommended_clip.start_time
        for i, c in enumerate(cands[:3]):
            dur = min(c.duration, max_s)
            thumb = _thumb(args.video, c.start_time + dur / 2,
                           os.path.join(args.output_dir, f"clip_{i}.jpg"))
            candidates.append({
                "index": i,
                "start": round(c.start_time, 2),
                "end": round(c.start_time + dur, 2),
                "duration": round(dur, 2),
                "score": round(float(c.avg_score), 3),
                "recommended": abs(c.start_time - best_start) < 0.01,
                "thumb": thumb,
                "label": f"{_fmt(c.start_time)}–{_fmt(c.start_time + dur)}",
            })
    except Exception as e:  # analysis must never block import
        emit({"type": "warning", "message": f"clip analysis failed: {e}"})

    if not candidates:
        # Fallback: offer evenly-spaced segments so the user can still choose.
        n = min(3, max(1, int(duration // max_s)))
        for i in range(n):
            start = i * max_s
            thumb = _thumb(args.video, start + max_s / 2,
                           os.path.join(args.output_dir, f"clip_{i}.jpg"))
            candidates.append({
                "index": i, "start": round(start, 2),
                "end": round(min(start + max_s, duration), 2),
                "duration": round(min(max_s, duration - start), 2),
                "score": 0.0, "recommended": i == 0, "thumb": thumb,
                "label": f"{_fmt(start)}–{_fmt(min(start + max_s, duration))}",
            })

    emit({"type": "clips", "needs_selection": True,
          "duration": round(duration, 2), "max_seconds": max_s, "aspect": aspect,
          "candidates": candidates,
          "filmstrip": _filmstrip(args.video, duration, args.output_dir)})


if __name__ == "__main__":
    main()
