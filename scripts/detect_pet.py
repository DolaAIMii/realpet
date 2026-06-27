#!/usr/bin/env python3
"""Quick pet detection for anchor confirmation.

Extracts the first frame, runs torchvision Faster R-CNN, and outputs:
- Annotated frame with bbox overlay
- Detection coordinates

Usage:
  python scripts/detect_pet.py --video /path/to/video.mp4 --output-dir /tmp/output

Output JSON lines:
  {"type": "detected", "cx": 645, "cy": 438, "name": "cat", "conf": 0.92,
   "bbox": [336, 273, 955, 602], "frame": "/tmp/output/first_frame.jpg",
   "annotated": "/tmp/output/first_frame_annotated.jpg"}
  {"type": "no_pet", "frame": "/tmp/output/first_frame.jpg"}
"""
import argparse
import json
import os
import subprocess

import cv2


def emit(msg):
    print(json.dumps(msg), flush=True)


def run_detect(video, output_dir, start=0, emit_fn=None):
    """Core detect logic — callable from daemon or CLI.

    Args:
        video: path to video file
        output_dir: output directory for frames
        start: time offset in seconds for the anchor frame
        emit_fn: optional callback; defaults to stdout print

    Returns:
        dict with detection result (type=detected or type=no_pet)
    """
    _emit = emit_fn or emit

    os.makedirs(output_dir, exist_ok=True)

    # Extract the anchor frame (first frame of the selected segment)
    first_frame_path = os.path.join(output_dir, "first_frame.jpg")
    cmd = ["ffmpeg", "-y"]
    if start and start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video, "-vframes", "1", "-q:v", "2", first_frame_path]
    subprocess.run(cmd, capture_output=True)

    if not os.path.exists(first_frame_path):
        return {"type": "error", "message": "Failed to extract first frame"}

    frame = cv2.imread(first_frame_path)
    h, w = frame.shape[:2]

    # Pet detection (torchvision Faster R-CNN).
    try:
        from pipeline.pet_detector import _run_detection, QC_MIN_SIZE
        detections_raw = _run_detection(frame, min_size=QC_MIN_SIZE)
        detections = []
        for d in detections_raw:
            detections.append({
                "name": d["name"],
                "conf": d["score"],
                "bbox": d["bbox"],
            })
    except Exception as e:
        return {"type": "error", "message": f"pet detection failed: {e}"}

    pets = detections

    # Annotate frame
    annotated = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d['name']} {d['conf']:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if pets:
        cv2.putText(annotated, "Click to confirm or change target", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(annotated, "No pet detected - click to select", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    annotated_path = os.path.join(output_dir, "first_frame_annotated.jpg")
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if pets:
        best = max(pets, key=lambda d: d["conf"])
        cx = int((best["bbox"][0] + best["bbox"][2]) / 2)
        cy = int((best["bbox"][1] + best["bbox"][3]) / 2)
        return {
            "type": "detected",
            "cx": cx, "cy": cy,
            "name": best["name"],
            "conf": round(best["conf"], 2),
            "bbox": [int(v) for v in best["bbox"]],
            "frame": first_frame_path,
            "annotated": annotated_path,
            "all_pets": [{"name": d["name"], "conf": round(d["conf"], 2),
                          "bbox": [int(v) for v in d["bbox"]]}
                         for d in pets],
        }
    else:
        return {
            "type": "no_pet",
            "frame": first_frame_path,
            "annotated": annotated_path,
            "all_detections": [{"name": d["name"], "conf": round(d["conf"], 2)}
                               for d in detections],
        }


def main():
    parser = argparse.ArgumentParser(description="Quick pet detection")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=float, default=0.0,
                        help="Detect the anchor on the frame at this time (s). "
                             "Use the chosen clip's start so the anchor matches "
                             "the segment that will actually be processed.")
    args = parser.parse_args()

    result = run_detect(args.video, args.output_dir, args.start)
    emit(result)


if __name__ == "__main__":
    main()
