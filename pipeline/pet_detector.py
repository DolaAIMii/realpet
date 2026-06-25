"""Unified pet detector using torchvision Faster R-CNN (BSD-3-Clause).

Replaces all Ultralytics YOLOv8 (AGPL-3.0) usage with a license-friendly
detector. The detector ONLY provides the first-frame bbox for SAM2 seeding
and app confirmation — it does NOT affect segmentation quality (SAM2 Apache
+ BiRefNet MIT handle that).

Model: fasterrcnn_resnet50_fpn_v2 (COCO pre-trained).
Threshold: 0.35 (tuned to match yolov8n conf=0.3 recall on pet classes).

COCO pet class IDs (there is NO rabbit/hamster in COCO — yolov8n never
detected those either, so removing them is not a regression):
    bird=16, cat=17, dog=18, horse=19, sheep=20, cow=21, bear=23
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# COCO class indices for pet categories
PET_COCO_IDS = {16, 17, 18, 19, 20, 21, 23}  # bird, cat, dog, horse, sheep, cow, bear

# Human-readable names for output (matched to COCO index order)
_COCO_NAMES = {
    16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 23: "bear",
}

# Detection confidence threshold. 0.5 is a sane floor for Faster R-CNN on pet
# classes; real pets score 0.98–1.00. We do NOT rely on a high threshold to
# kill ghost boxes anymore — class-agnostic NMS (below) handles overlapping
# duplicate labels (cat@1.00 + dog@0.83 on the SAME animal), which a threshold
# alone can't, since the ghost label can itself score high (0.83 seen in test).
DETECTION_THRESHOLD = 0.5

# Class-agnostic NMS IoU. Boxes overlapping more than this are treated as the
# same physical animal; only the top-scoring one survives. 0.5 is the standard
# detection default and cleanly merges the cat/dog double-label case.
NMS_IOU_THRESHOLD = 0.5

# Singleton model + device cache
_model = None
_device = None


def _load_model():
    """Lazy-load the Faster R-CNN model (singleton)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch

    # Force CPU — Faster R-CNN has MPS issues (Metal command buffer errors,
    # hangs, or silent wrong results on Apple Silicon). CPU is fast enough
    # for single-image inference (~200ms vs ~50ms on MPS, but MPS doesn't
    # work reliably here).
    _device = torch.device("cpu")

    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    _model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    _model.to(_device).eval()

    return _model, _device


def _run_detection(bgr: np.ndarray, min_size: Optional[int] = None) -> List[dict]:
    """Run Faster R-CNN on a BGR image. Returns list of {bbox, score, name}.

    Args:
        bgr: BGR image (H,W,3 uint8).
        min_size: temporarily override the model's internal resize shortest-side
            (default 800). LOWER = faster but coarser. The model rescales every
            input to min_size internally, so an EXTERNAL downscale has no effect —
            this knob is the real speed lever. Used by the QC gate (presence
            check). Leave None for full-resolution accuracy (SAM2 seed box).
    """
    import torch

    model, device = _load_model()

    # BGR → RGB → CHW tensor [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(device)

    _saved = None
    if min_size is not None:
        _saved = (model.transform.min_size, model.transform.max_size)
        model.transform.min_size = (min_size,)
        # Keep the standard ~1.66 aspect cap so wide frames aren't blown up.
        model.transform.max_size = int(round(min_size * 1000.0 / 600.0))

    try:
        with torch.no_grad():
            preds = model([tensor])
    finally:
        if _saved is not None:
            model.transform.min_size, model.transform.max_size = _saved

    from torchvision.ops import nms

    pred = preds[0]
    boxes = pred["boxes"]            # (N, 4) xyxy tensor
    scores = pred["scores"]          # (N,) tensor
    labels = pred["labels"]          # (N,) COCO class index tensor

    # Keep only pet classes above threshold.
    keep = []
    for i in range(len(scores)):
        if float(scores[i]) >= DETECTION_THRESHOLD and int(labels[i]) in PET_COCO_IDS:
            keep.append(i)

    results = []
    if keep:
        idx = torch.tensor(keep, dtype=torch.long)
        kb, ks, kl = boxes[idx], scores[idx], labels[idx]
        # CLASS-AGNOSTIC NMS: the same animal often gets two overlapping boxes
        # under different labels (e.g. a cat scored cat@1.00 AND dog@0.83).
        # Plain torchvision keeps both because it only does per-class NMS. We
        # suppress across classes so one physical animal yields one box — the
        # highest-scoring label wins.
        # torchvision's transform already rescales boxes back to the ORIGINAL
        # input resolution, so the min_size override doesn't affect coordinates.
        survivors = nms(kb, ks, iou_threshold=NMS_IOU_THRESHOLD).tolist()
        for j in survivors:
            results.append({
                "bbox": kb[j].cpu().tolist(),  # [x1, y1, x2, y2] in original pixels
                "score": float(ks[j]),
                "name": _COCO_NAMES.get(int(kl[j]), "unknown"),
                "class_id": int(kl[j]),
            })

    return results


# Internal resize shortest-side for the fast QC presence check. 320 (vs default
# 800) cuts 1080p inference ~1.75s → ~0.9s while still detecting a normal pet.
QC_MIN_SIZE = 320


def has_pet_fast(bgr_or_path) -> bool:
    """Fast presence check for the QC gate — low-res inference, no precise box.

    Returns True if any pet class is detected above threshold. Faster than
    detect_pet_bbox (runs at min_size=320 instead of 800). Use ONLY for go/no-go
    quality gating, NOT for SAM2 seeding (which needs full-res boxes).
    """
    if isinstance(bgr_or_path, (str, os.PathLike)):
        bgr = cv2.imread(str(bgr_or_path))
        if bgr is None:
            return False
    else:
        bgr = bgr_or_path

    return len(_run_detection(bgr, min_size=QC_MIN_SIZE)) > 0


def detect_pet_bbox(bgr_or_path) -> Optional[List[int]]:
    """Detect the highest-confidence pet and return its bbox [x1,y1,x2,y2].

    Args:
        bgr_or_path: BGR numpy array (H,W,3 uint8) OR file path string.

    Returns:
        [x1, y1, x2, y2] in pixel coordinates at original frame resolution,
        or None if no pet detected above threshold.
    """
    if isinstance(bgr_or_path, (str, os.PathLike)):
        bgr = cv2.imread(str(bgr_or_path))
        if bgr is None:
            return None
    else:
        bgr = bgr_or_path

    detections = _run_detection(bgr)
    if not detections:
        return None

    # Pick highest-confidence pet
    best = max(detections, key=lambda d: d["score"])
    return [int(v) for v in best["bbox"]]


def detect_pet_centroid(bgr_or_path) -> Optional[Tuple[int, int, str, float]]:
    """Detect the highest-confidence pet and return (cx, cy, name, conf).

    Args:
        bgr_or_path: BGR numpy array OR file path string.

    Returns:
        (cx, cy, class_name, confidence) or None if no pet detected.
    """
    if isinstance(bgr_or_path, (str, os.PathLike)):
        bgr = cv2.imread(str(bgr_or_path))
        if bgr is None:
            return None
    else:
        bgr = bgr_or_path

    detections = _run_detection(bgr)
    if not detections:
        return None

    best = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = best["bbox"]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy, best["name"], best["score"])
