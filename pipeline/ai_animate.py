"""AI animation generation: prepare images, call API, extract and process frames."""
import os
import subprocess
import time
from PIL import Image


def composite_for_api(rgba_path, output_path, bg_color=(255, 255, 255)):
    """Composite RGBA image onto solid background for API input."""
    img = Image.open(rgba_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (*bg_color, 255))
    composite = Image.alpha_composite(bg, img)
    rgb = composite.convert("RGB")
    rgb.save(output_path, quality=95)
    return output_path


def extract_frames(video_path, output_dir, fps=24, progress_callback=None):
    """Extract frames from video using ffmpeg.

    Extracts as JPEG for speed (12x faster I/O than PNG). rembg can read
    JPEG directly so no conversion needed.

    Args:
        video_path: path to input video
        output_dir: directory to write frame JPEGs
        fps: target frame rate
        progress_callback: optional callback(phase, current, total, detail)

    Returns:
        list of extracted frame paths (sorted)
    """
    import glob as g

    os.makedirs(output_dir, exist_ok=True)

    # Clear old frames
    for old in g.glob(os.path.join(output_dir, "frame_*.*")):
        os.remove(old)

    # Get total frame count for progress tracking
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path
    ]
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        total_frames = int(probe_result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        total_frames = 0

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    seen = set()
    while proc.poll() is None:
        current = sorted(g.glob(os.path.join(output_dir, "frame_*.jpg")))
        new_count = len(current)
        if new_count > len(seen) and progress_callback:
            seen_count = total_frames if total_frames > 0 else new_count
            progress_callback("extract", new_count, seen_count, f"frame_{new_count:04d}.jpg")
            seen = set(current)
        time.sleep(0.05)
    proc.wait()

    if proc.returncode != 0:
        return []

    frames = sorted(g.glob(os.path.join(output_dir, "frame_*.jpg")))
    if progress_callback:
        progress_callback("extract", len(frames), len(frames), "done")
    return frames


def segment_frames_wrapper(frame_paths, output_dir, use_coreml=False,
                           progress_callback=None, max_dim=None):
    """Segment extracted frames with rembg.

    Args:
        frame_paths: list of input RGB frame paths
        output_dir: directory for RGBA output
        use_coreml: use CoreML execution provider
        progress_callback: optional callback(phase, current, total, detail)
        max_dim: max dimension for segmentation (downscale then upscale alpha)

    Returns:
        list of segmented RGBA frame paths
    """
    from pipeline.segment import segment_frames
    frames_dir = os.path.dirname(frame_paths[0]) if frame_paths else output_dir
    return segment_frames(frames_dir, output_dir, use_coreml=use_coreml,
                          progress_callback=progress_callback, max_dim=max_dim,
                          upscale_result=False)


def _crop_bounds_from_alpha(frame_paths, margin=20):
    """Find union bounding box from alpha channels only (memory efficient)."""
    import numpy as np
    min_r, min_c, max_r, max_c = None, None, None, None
    w, h = 0, 0
    for p in frame_paths:
        a = np.array(Image.open(p).convert("L"))  # grayscale, 1/4 memory of RGBA
        w, h = a.shape[1], a.shape[0]
        rows = np.any(a > 0, axis=1)
        cols = np.any(a > 0, axis=0)
        if not rows.any() or not cols.any():
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        min_r = r0 if min_r is None else min(min_r, r0)
        min_c = c0 if min_c is None else min(min_c, c0)
        max_r = r1 if max_r is None else max(max_r, r1)
        max_c = c1 if max_c is None else max(max_c, c1)
    if min_r is None:
        return None
    return (
        max(0, min_c - margin),
        max(0, min_r - margin),
        min(w - 1, max_c + margin),
        min(h - 1, max_r + margin),
    )


def normalize_frames(frame_paths, target_size=None, progress_callback=None):
    """Normalize frames: crop to foreground, resize, ensure consistent dimensions."""
    if not frame_paths:
        return []

    # Pass 1: find crop bounds by reading alpha only (low memory)
    crop_box = _crop_bounds_from_alpha(frame_paths, margin=30)

    # Pass 2: load + crop + resize in one step (never hold all full-res RGBA)
    if crop_box is None:
        # No foreground found, load as-is
        imgs = [Image.open(p).convert("RGBA") for p in frame_paths]
    else:
        imgs = [Image.open(p).convert("RGBA").crop(crop_box) for p in frame_paths]

    # Determine output size — keep content aspect ratio
    w, h = imgs[0].size
    if target_size is None:
        max_dim = max(w, h)
        if max_dim > 400:
            scale = 400.0 / max_dim
            out_size = (int(w * scale), int(h * scale))
        else:
            out_size = (w, h)
    else:
        max_w, max_h = target_size
        scale = min(max_w / w, max_h / h)
        out_size = (int(w * scale), int(h * scale))

    resized = []
    for i, img in enumerate(imgs):
        resized.append(img.resize(out_size, Image.LANCZOS))
        if progress_callback:
            progress_callback("normalize", i + 1, len(imgs), f"frame_{i:04d}.png")

    return resized


def process_video_to_frames(video_path, output_dir, fps=24, target_size=None,
                            use_coreml=False, progress_callback=None,
                            seg_max_dim=768):
    """Full pipeline: video -> extracted -> segmented -> normalized frames.

    Args:
        video_path: path to video file
        output_dir: base output directory
        fps: extraction frame rate
        target_size: optional (max_w, max_h) for final frames
        use_coreml: use CoreML for segmentation acceleration
        progress_callback: optional callback(phase, current, total, detail)
        seg_max_dim: max dimension for segmentation (default 768 for speed)

    Returns:
        list of final frame paths ready for display
    """

    extract_dir = os.path.join(output_dir, "extracted")
    segment_dir = os.path.join(output_dir, "segmented")

    # Step 1: Extract frames
    if progress_callback:
        progress_callback("phase", 0, 0, "extract")
    raw_frames = extract_frames(video_path, extract_dir, fps=fps,
                                progress_callback=progress_callback)
    if not raw_frames:
        return []

    # Step 2: Segment each frame (with downscale optimization)
    if progress_callback:
        progress_callback("phase", 0, 0, "segment")
    seg_frames = segment_frames_wrapper(raw_frames, segment_dir,
                                        use_coreml=use_coreml,
                                        progress_callback=progress_callback,
                                        max_dim=seg_max_dim)

    # Step 3: Normalize
    if progress_callback:
        progress_callback("phase", 0, 0, "normalize")
    normalized = normalize_frames(seg_frames, target_size=target_size,
                                  progress_callback=progress_callback)

    # Save as JPEG pairs (RGB + alpha) — 14x faster I/O than PNG
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_paths = []
    for i, img in enumerate(normalized):
        stem = f"frame_{i:04d}"
        rgb_path = os.path.join(final_dir, f"{stem}.jpg")
        alpha_path = os.path.join(final_dir, f"{stem}_a.jpg")
        img.convert("RGB").save(rgb_path, quality=92)
        img.getchannel("A").save(alpha_path, quality=92)
        final_paths.append(rgb_path)

    return final_paths
