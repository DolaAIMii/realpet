"""Foreground segmentation using BiRefNet with alpha post-processing.

Uses BiRefNet (BiRefNet-matting) on the best available torch device
(MPS on Apple Silicon, else CPU) for high-quality segmentation.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Lazy-loaded model cache
_birefnet_model = None
_birefnet_transform = None


def _get_birefnet():
    """Lazy-load BiRefNet on the best available device."""
    global _birefnet_model, _birefnet_transform
    if _birefnet_model is not None:
        return _birefnet_model, _birefnet_transform

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet-matting", trust_remote_code=True
    )
    _birefnet_model.to(device)
    _birefnet_model.eval()

    _birefnet_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Warm up
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    with torch.no_grad():
        _birefnet_model(dummy)

    return _birefnet_model, _birefnet_transform


def _birefnet_inference(img_bgr):
    """Run BiRefNet on a single BGR image, return alpha as uint8 array."""
    import torch
    import cv2

    model, transform = _get_birefnet()
    device = next(model.parameters()).device
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inp = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = torch.sigmoid(pred)

    alpha = torch.nn.functional.interpolate(
        pred, size=(h, w), mode="bilinear", align_corners=False
    )
    return (alpha.squeeze().cpu().numpy() * 255).astype(np.uint8)


def detect_temporal_anomaly(alphas, idx, h, w):
    """Detect anomalous frames using neighbor consistency.

    A frame is anomalous if it has foreground in regions where ALL 4
    immediate neighbors (±1, ±2) are empty. This is conservative —
    it only catches artifacts that appear for ≤2 consecutive frames.

    Args:
        alphas: list of uint8 alpha arrays
        idx: index of frame to check
        h, w: frame dimensions

    Returns:
        tuple: (is_anomaly, novel_mask) or (False, None)
    """

    if len(alphas) < 5 or idx < 2 or idx >= len(alphas) - 2:
        return False, None

    # Build neighbor consensus from 4 immediate neighbors
    neighbors = []
    for offset in [-2, -1, 1, 2]:
        nidx = idx + offset
        if 0 <= nidx < len(alphas):
            neighbors.append(alphas[nidx])

    neighbor_max = neighbors[0].copy()
    for n in neighbors[1:]:
        neighbor_max = np.maximum(neighbor_max, n)

    # Stable empty: pixels where ALL neighbors have low alpha
    stable_empty = neighbor_max < 30

    # Current foreground
    curr_fg = alphas[idx] > 128

    # Novel in empty: foreground that appeared in previously-empty regions
    novel_in_empty = curr_fg & stable_empty

    novel_area = novel_in_empty.sum()
    total_fg = curr_fg.sum()

    if total_fg == 0 or novel_area == 0:
        return False, None

    novel_ratio = novel_area / total_fg

    if novel_ratio > 0.02:
        return True, novel_in_empty

    return False, None


def repair_temporal_anomaly(alphas, idx, novel_mask, h, w):
    """Repair an anomalous frame by removing novel artifacts.

    Replaces foreground in novel regions with the neighbor consensus
    (which is ~0), with soft blending at boundaries.

    Args:
        alphas: list of uint8 alpha arrays
        idx: index of bad frame
        novel_mask: boolean mask of novel regions
        h, w: frame dimensions

    Returns:
        repaired alpha array
    """
    import cv2

    if novel_mask is None or not novel_mask.any():
        return alphas[idx]

    # Build neighbor consensus
    neighbors = []
    for offset in [-2, -1, 1, 2]:
        nidx = idx + offset
        if 0 <= nidx < len(alphas):
            neighbors.append(alphas[nidx].astype(float))

    neighbor_avg = np.mean(neighbors, axis=0)

    # Create soft mask for removal
    mask = novel_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)

    # Blend: novel regions fade to neighbor average
    factor = mask.astype(float) / 255
    repaired = (alphas[idx].astype(float) * (1 - factor) +
                neighbor_avg * factor).astype(np.uint8)

    return repaired


def _postprocess_alpha(alpha, alpha_threshold=30, edge_blur=1.0, fill_holes=True):
    """Clean up segmentation alpha channel. Uses OpenCV for fast morphology.

    IMPORTANT: fill_holes only fills INTERIOR holes — it must NOT binarize
    edge pixels. BiRefNet's soft alpha gradient (whisker translucency, fur
    edges) must be preserved.
    """
    import cv2

    a = alpha.copy()

    # 1. Threshold — kill near-zero noise
    a[a < alpha_threshold] = 0

    # 2. Fill small interior holes via close operation
    if fill_holes:
        mask = (a > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Only fill pixels that the close operation ADDED (were holes before)
        # These are interior regions that were 0 but should be foreground
        hole_mask = (closed > 0) & (a == 0)
        # For these holes, use the local median alpha of surrounding pixels
        # to maintain soft edges rather than forcing 255
        if hole_mask.any():
            # Dilate the original foreground to get neighbor context
            dilated = cv2.dilate(a, kernel, iterations=2)
            a[hole_mask] = dilated[hole_mask]

    # 3. Edge blur — smooth jagged edges
    if edge_blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), edge_blur)

    return a


def segment_frames(frames_dir, output_dir, postprocess=True, use_coreml=False,
                   progress_callback=None, max_dim=None, upscale_result=True):
    """Segment all frames using BiRefNet MPS with temporal consistency repair.

    Pipeline per frame:
    1. BiRefNet MPS inference (~1.5s)
    2. Alpha post-processing (preserves soft edges)
    3. Temporal anomaly detection (scene-agnostic, no hardcoded coordinates)
    4. If anomalous: optical flow warp from nearest good frame

    Args:
        frames_dir: directory containing frame PNGs/JPGs
        output_dir: directory to write segmented RGBA PNGs
        postprocess: apply alpha post-processing
        use_coreml: unused (kept for API compat)
        progress_callback: optional callback(phase, current, total, detail)
        max_dim: unused (BiRefNet handles resize internally)
        upscale_result: unused (kept for API compat)

    Returns:
        list of output file paths
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(
        list(Path(frames_dir).glob("*.png")) + list(Path(frames_dir).glob("*.jpg"))
    )
    # Deduplicate
    seen_stems = set()
    deduped = []
    for f in frame_files:
        if f.stem not in seen_stems:
            seen_stems.add(f.stem)
            deduped.append(f)
    frame_files = deduped
    total = len(frame_files)
    results = []

    # Pre-load all frames
    all_frames_bgr = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        all_frames_bgr.append(img)
    h, w = all_frames_bgr[0].shape[:2]

    # Phase 1: BiRefNet segmentation
    alphas = []
    for i, frame_path in enumerate(frame_files):
        out_name = frame_path.stem + ".png"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            existing = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if existing is not None and len(existing.shape) == 3 and existing.shape[2] == 4:
                alphas.append(existing[:, :, 3])
                results.append(out_path)
                if progress_callback:
                    progress_callback("segment", i + 1, total, f"{frame_path.name} (cached)")
                continue

        alpha = _birefnet_inference(all_frames_bgr[i])

        if postprocess:
            alpha = _postprocess_alpha(alpha)

        alphas.append(alpha)
        results.append(out_path)
        if progress_callback:
            progress_callback("segment", i + 1, total, frame_path.name)

    # Phase 2: Temporal anomaly detection + repair
    bad_count = 0
    for i in range(total):
        is_anomaly, novel_mask = detect_temporal_anomaly(alphas, i, h, w)
        if is_anomaly:
            bad_count += 1
            if progress_callback:
                progress_callback("fix", bad_count, total,
                                  f"{frame_files[i].name} → temporal repair")

            alphas[i] = repair_temporal_anomaly(alphas, i, novel_mask, h, w)

    # Phase 3: Save all results
    for i in range(total):
        out_path = results[i]
        rgba = cv2.cvtColor(all_frames_bgr[i], cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alphas[i]
        cv2.imwrite(out_path, rgba)

    if bad_count > 0 and progress_callback:
        progress_callback("segment", total, total,
                          f"done ({bad_count} frames repaired via optical flow)")

    return results


def segment_image(input_path, output_path, postprocess=True, use_coreml=False):
    """Remove background from a single image, output RGBA PNG."""
    import cv2

    img_bgr = cv2.imread(input_path)
    alpha = _birefnet_inference(img_bgr)

    if postprocess:
        alpha = _postprocess_alpha(alpha)

    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = alpha

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgba)
    return output_path


def segment_pairs(pairs, postprocess=True, use_coreml=False, progress_callback=None):
    """Segment all parsed Live Photo pairs."""
    import cv2

    for pair in pairs:
        pair_dir = pair["pair_dir"]
        seg_dir = os.path.join(pair_dir, "segmented")
        os.makedirs(seg_dir, exist_ok=True)

        frame_list = pair.get("frames", [])
        segmented = []

        for i, frame_path in enumerate(frame_list):
            out_path = os.path.join(seg_dir, os.path.basename(frame_path))
            if os.path.exists(out_path):
                segmented.append(out_path)
                if progress_callback:
                    progress_callback("segment", i + 1, len(frame_list),
                                      os.path.basename(frame_path))
                continue

            img_bgr = cv2.imread(frame_path)
            alpha = _birefnet_inference(img_bgr)
            if postprocess:
                alpha = _postprocess_alpha(alpha)

            rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = alpha
            cv2.imwrite(out_path, rgba)
            segmented.append(out_path)
            if progress_callback:
                progress_callback("segment", i + 1, len(frame_list),
                                  os.path.basename(frame_path))

        pair["segmented_frames"] = segmented

        key_out = os.path.join(seg_dir, "key_photo.png")
        if not os.path.exists(key_out):
            segment_image(pair["key_photo"], key_out, postprocess=postprocess)
        pair["segmented_key"] = key_out

    return pairs
