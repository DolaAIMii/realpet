"""Parse Live Photo files (.heic + .mov pairs), extract frames."""
import os
import subprocess
import json
import shutil
from pathlib import Path


def find_pairs(input_dir):
    """Find .heic/.mov file pairs by matching filename stems."""
    input_path = Path(input_dir)
    heic_files = sorted(input_path.glob("*.heic")) + sorted(input_path.glob("*.HEIC"))
    mov_files = sorted(input_path.glob("*.mov")) + sorted(input_path.glob("*.MOV"))

    mov_stems = {f.stem.lower(): f for f in mov_files}
    pairs = []
    orphans = []

    for heic in heic_files:
        stem = heic.stem.lower()
        if stem in mov_stems:
            pairs.append((heic, mov_stems[stem]))
        else:
            orphans.append(heic)

    return pairs, orphans


def get_video_info(mov_path):
    """Get video duration, fps, resolution via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(mov_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    # Parse fps from r_frame_rate (e.g. "30000/1001")
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den
    width = int(stream["width"])
    height = int(stream["height"])
    duration = float(stream.get("duration", 0))
    return {"fps": fps, "width": width, "height": height, "duration": duration}


def extract_frames(mov_path, output_dir, target_fps=10):
    """Extract frames from MOV at target FPS using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mov_path),
        "-vf", f"fps={target_fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.png")
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    frames = sorted(Path(output_dir).glob("frame_*.png"))
    return [str(f) for f in frames]


def convert_heic(heic_path, output_path):
    """Convert HEIC to PNG using heif-convert or sips."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try heif-convert first
    heif_convert = shutil.which("heif-convert")
    if heif_convert:
        cmd = [heif_convert, str(heic_path), output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path

    # Fallback to macOS sips
    cmd = ["sips", "-s", "format", "png", str(heic_path), "--out", output_path]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def parse_livephoto(input_dir, output_dir, target_fps=10):
    """Parse all Live Photo pairs in input_dir.

    Returns:
        list of dicts: [{"key_photo": path, "frames": [paths], "info": {...}}, ...]
    """
    pairs, orphans = find_pairs(input_dir)
    if orphans:
        for o in orphans:
            print(f"  [warn] No matching .mov for {o.name}")

    results = []
    for i, (heic, mov) in enumerate(pairs):
        pair_dir = os.path.join(output_dir, f"pair_{i:03d}")
        frames_dir = os.path.join(pair_dir, "frames")
        key_photo_path = os.path.join(pair_dir, "key_photo.png")

        print(f"  Parsing {heic.name} + {mov.name} ...")
        info = get_video_info(mov)
        print(f"    Video: {info['width']}x{info['height']}, {info['fps']:.1f}fps, {info['duration']:.1f}s")

        frames = extract_frames(mov, frames_dir, target_fps)
        print(f"    Extracted {len(frames)} frames at {target_fps}fps")

        convert_heic(heic, key_photo_path)
        print(f"    Key photo: {key_photo_path}")

        results.append({
            "key_photo": key_photo_path,
            "frames": frames,
            "info": info,
            "pair_dir": pair_dir,
        })

    return results
