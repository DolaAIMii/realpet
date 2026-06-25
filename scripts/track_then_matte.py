#!/usr/bin/env python3
"""Track-then-Matte: SAM2 tracking + BiRefNet refinement.

Two-pass pipeline with quality check:
  Step 0: Quality check (<1s, pure CV)
  Pass 1 (SAM2): Track pet → logit soft alpha preview
  Pass 2 (BiRefNet): Refine edges → final alpha

Usage:
  python scripts/track_then_matte.py --video /path/to/video.mp4 --output-dir /tmp/output
  python scripts/track_then_matte.py --video /path/to/video.mp4 --output-dir /tmp/output --click 244,498

Requires Python 3.10+ venv (set REALPET_VENV or default to .venv in project root).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

# Ensure project root is in sys.path for pipeline imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _weights_dir():
    """Return the weights directory, configurable via REALPET_WEIGHTS_DIR env var.

    Default: <project_root>/weights/
    """
    return os.environ.get("REALPET_WEIGHTS_DIR",
                          os.path.join(_project_root, "weights"))


def _sam2_checkpoint():
    """Return the SAM2 checkpoint path."""
    return os.path.join(_weights_dir(), "sam2", "sam2.1_hiera_tiny.pt")


# === Quality Check ===

class QualityResult:
    """Quality check result."""
    def __init__(self):
        self.passed = True
        self.issues = []  # list of (code, message)

    def fail(self, code, message):
        self.passed = False
        self.issues.append({"code": code, "message": message})

    def to_dict(self):
        return {"passed": self.passed, "issues": self.issues}


def _any_frame_has_pet(frames, sample=6):
    """Sample up to `sample` evenly-spaced frames; pet exists if ANY hits.

    Uses the FAST downscaled presence check (has_pet_fast, ≤512px) and exits
    on the first hit — the QC gate only needs go/no-go, not a precise box.
    Typical cost: <1s if a pet is in an early frame, ≤~5s worst case (no pet,
    all `sample` frames scanned).

    Raises ImportError if the detector module can't be loaded (service error).
    Returns False only if every sampled frame returned no pet (truly no pet).
    """
    if not frames:
        return False
    from pipeline.pet_detector import has_pet_fast
    step = max(1, len(frames) // sample)
    for fp in frames[::step]:
        if has_pet_fast(fp):
            return True
    return False


def check_quality(video_path, frames, min_brightness=20, min_blur=5,
                  min_resolution=480, min_duration=2, max_duration=60,
                  min_pet_conf=0.3):
    """Run quality checks on video and frames.

    Checks (deliberately MINIMAL — fast gate, low false-reject risk):
    1. Brightness: average luminance > threshold
    2. Blur: Laplacian variance > threshold (bottom 10% frames). NOTE: this
       also covers "动作幅度太大" and "剧烈抖动" — both produce motion blur, so
       there is NO separate motion/shake check (it would误杀追拍/慢动作好素材).
    3. Overexposure: high-clipping ratio < threshold
    4. Resolution: min(width, height) >= threshold
    5. Duration: within [min, max] seconds
    6. Pet detection: fallback (any frame has pet)

    Deliberately NOT checked here (decided 2026-06-22, see QC_QUALITY_GATE_PLAN):
    - Noise: Gaussian high-pass can't separate noise from high-freq fur/whiskers/
      grass →误杀核心猫素材. Dropped.
    - Camera shake: phase-correlation can't cheaply distinguish手抖 from追拍平移,
      and a sharp-but-shaky clip still segments fine. Dropped.
    - "边缘太杂难分割": cannot be cheaply predicted pre-segmentation → handled by
      the POST gate (red_ratio > MAX_RED_RATIO) after the pipeline runs.

    Args:
        video_path: path to video file
        frames: list of extracted frame paths
        min_brightness: minimum average brightness (0-255)
        min_blur: minimum Laplacian variance for blur detection
        min_resolution: minimum width or height in pixels
        min_duration: minimum video duration in seconds
        max_duration: maximum video duration in seconds
        min_pet_conf: minimum pet detection confidence

    Returns:
        QualityResult
    """
    result = QualityResult()

    # 1. Check resolution
    if frames:
        h, w = cv2.imread(frames[0]).shape[:2]
        if min(w, h) < min_resolution:
            result.fail("low_resolution",
                        f"视频分辨率过低（{w}x{h}），请使用高清视频")

    # 2. Check duration
    video_duration = None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=5
        )
        video_duration = float(probe.stdout.strip())
        if video_duration < min_duration:
            result.fail("too_short",
                        f"视频时长过短（{video_duration:.1f}秒），请录制至少{min_duration}秒")
        elif video_duration > max_duration:
            result.fail("too_long",
                        f"视频时长过长（{video_duration:.1f}秒），请控制在{max_duration}秒以内")
    except Exception:
        pass  # ffprobe failure is not fatal

    if not frames:
        result.fail("no_frames", "未能提取视频帧")
        return result

    # 3. Check brightness / blur / overexposure (all reuse the same grayscale,
    # one read per frame — keeps the gate fast, <0.1s total for these three).
    brightness_scores = []
    blur_scores = []
    overexp_scores = []
    for fp in frames[:20]:  # sample first 20 frames for speed
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_scores.append(float(gray.mean()))
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        # Overexposure: ratio of pixels at/above 250 (blown-out highlights)
        overexp_scores.append(float((gray >= 250).mean()))

    if brightness_scores:
        avg_brightness = np.mean(brightness_scores)
        if avg_brightness < min_brightness:
            result.fail("too_dark",
                        f"画面太暗（平均亮度{avg_brightness:.0f}），请在光线充足的环境拍摄")

    # 4. Check blur (bottom 10%)
    if blur_scores and len(blur_scores) >= 4:
        sorted_blur = sorted(blur_scores)
        p10_idx = max(0, int(len(sorted_blur) * 0.1))
        p10_blur = sorted_blur[p10_idx]
        if p10_blur < min_blur:
            result.fail("too_blurry",
                        f"画面模糊（清晰度{p10_blur:.1f}），请保持手机稳定")

    # 5. Check overexposure (high-clipping ratio)
    if overexp_scores:
        avg_overexp = np.mean(overexp_scores)
        if avg_overexp > MAX_OVEREXP_RATIO:
            result.fail("too_bright",
                        f"画面过曝（{avg_overexp*100:.0f}%高光溢出），请降低曝光或避免强光直射")

    # 6. Check pet detection (multi-frame scan — any hit = pass, fallback gate)
    try:
        if not _any_frame_has_pet(frames):
            result.fail("no_pet",
                        "未检测到宠物，请确认视频中有清晰的宠物画面")
    except ImportError as e:
        result.fail("detect_error",
                    f"检测服务异常：{e}，请重试")
    except Exception as e:
        result.fail("detect_error",
                    f"检测服务异常：{e}，请重试")

    return result


# === Pipeline ===

def emit(msg):
    """Output JSON line for CLI integration."""
    print(json.dumps(msg), flush=True)


# Adaptive per-clip auto colour grade (color investigation 2026-06-18). Measured
# ONCE over sampled frames and applied identically to every frame — per-frame
# stats would flicker as the pet moves. Corrects the whole scene toward standard
# targets so the cut-out pet looks well-exposed / neutral / lively regardless of
# how the clip was shot. Conservative clamps make a well-shot clip a near no-op.
# Runs at SAVE on the RGB only — alpha is computed on the untouched frame first,
# so the matte/edges are unaffected.
#
# Matte itself is pixel-faithful (verified: extracted == segmented RGB byte-equal,
# interior alpha fully opaque, sRGB passthrough). For SDR sources this grade is
# optional polish; it also standardises the (now correctly tone-mapped) HDR ones.
AUTO_GRADE = True

# Calibration targets (8-bit luma / HSV-S). "Good" clip ≈ black p1 2–10,
# white p99 235–250, median 105–130, dynamic range ≥180, mean saturation 55–90.
_GRADE_BLACK, _GRADE_WHITE = 6.0, 246.0    # target black / white points
_GRADE_LEVELS_STRENGTH = 0.75              # partial pull toward those points
_GRADE_MED_TARGET = 118.0                  # target midtone (median luma)
_GRADE_SAT_TARGET = 62.0                   # lift dull clips up to here, no further

# === QC gate thresholds (calibrated 2026-06-22 on 6 good assets) ===
# Deliberately minimal — only overexposure was added to the existing
# dark/blur/resolution/duration checks. Noise & shake were dropped (误杀风险高).
MAX_OVEREXP_RATIO = 0.10   # 高光溢出占比上限 (优质 <0.001，10% 才拦大面积死白)
MAX_RED_RATIO     = 0.20   # 后置门：red 坏帧占比上限 (分割达不到)

# Adaptive unsharp mask (clarity). Amount scales inversely with the clip's
# measured sharpness (soft clips get more, already-crisp ones less) so we never
# over-crunch. Threshold skips flat areas so we don't amplify noise. Tuned on
# IMG_0847: amount 0.8 ≈ sweet spot, 1.4 starts to halo.
_SHARPEN_SIGMA, _SHARPEN_THRESH = 1.2, 3.0
_SHARPEN_MIN, _SHARPEN_MAX = 0.4, 0.9      # amount clamp
_SHARPEN_REF = 75.0                        # Laplacian-var pivot for the amount curve


def _luma_bgr(b):
    return 0.114 * b[..., 0] + 0.587 * b[..., 1] + 0.299 * b[..., 2]


def measure_grade_stats(frame_paths, logits=None, sample=8):
    """Robust luma / saturation / white-balance stats over sampled frames.

    White balance is estimated two ways, in priority order (see compute_grade):
      1. white_ref — "white-patch" on the bright, near-neutral pixels INSIDE the
         pet (e.g. white fur). This is the reliable neutral reference; whole-scene
         gray-world over-corrects when the scene has a dominant cast (a warm wood
         floor pushed the neutral white fur strongly blue — the bug this fixes).
      2. neutral — gray-world over near-neutral mid-tones (fallback for pets with
         no bright neutral patch, e.g. an all-black or ginger pet).
    `logits` is the SAM2 logit dict (frame_idx → logit) used to mask the pet.
    """
    n = len(frame_paths)
    if n == 0:
        return None
    idx = (list(range(n)) if n <= sample
           else [int(round(i * (n - 1) / (sample - 1))) for i in range(sample)])
    Ys, Ss, neutral, sharps, white_refs = [], [], [], [], []
    for i in idx:
        bgr = cv2.imread(frame_paths[i])
        if bgr is None:
            continue
        b = bgr.astype(np.float32)
        h, w = b.shape[:2]
        Y = _luma_bgr(b)
        Ys.append(Y.ravel())
        mx = b.max(2); mn = b.min(2)
        S = np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0)
        Ss.append((S * 255).ravel())
        sharps.append(float(cv2.Laplacian(Y, cv2.CV_32F).var()))
        # Gray-world fallback reference: near-neutral mid-tones.
        m = (S < 0.18) & (Y > 30) & (Y < 220)
        if m.sum() > 50:
            neutral.append(b[m].reshape(-1, 3).mean(0))
        # White-patch reference: brightest near-neutral pixels INSIDE the pet.
        lg = logits.get(i) if logits is not None else None
        if lg is not None:
            if lg.shape != (h, w):
                lg = cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR)
            pet = (1.0 / (1.0 + np.exp(-lg * 2.0))) > 0.5
            if pet.sum() > 500:
                thr = np.percentile(Y[pet], 85)
                wm = pet & (Y >= thr) & (S < 0.22)
                if wm.sum() > 300:
                    white_refs.append(b[wm].reshape(-1, 3).mean(0))
    if not Ys:
        return None
    Y = np.concatenate(Ys); S = np.concatenate(Ss)
    return {"p1": float(np.percentile(Y, 1)), "p50": float(np.percentile(Y, 50)),
            "p99": float(np.percentile(Y, 99)), "sat": float(S.mean()),
            "sharp": float(np.mean(sharps)) if sharps else 100.0,
            "neutral": np.mean(neutral, 0) if neutral else None,
            "white_ref": np.median(white_refs, 0) if white_refs else None}


def compute_grade(st):
    """Clip stats → clamped grade params (wb gains, levels affine, gamma, sat)."""
    if st is None:
        return None
    p = {"wb": np.array([1.0, 1.0, 1.0], np.float32),
         "a": 1.0, "b": 0.0, "gamma": 1.0, "sat": 1.0}
    # 1) White balance, green-anchored. Prefer the pet white-patch (white fur);
    # fall back to gray-world with a TIGHTER clamp (gray-world over-corrects a
    # cast scene — that's what tinted the white fur blue).
    if st.get("white_ref") is not None:
        bgr = st["white_ref"]; g = bgr[1]
        p["wb"] = np.clip(np.array([g / max(bgr[0], 1e-3), 1.0,
                                    g / max(bgr[2], 1e-3)], np.float32), 0.85, 1.15)
    elif st["neutral"] is not None:
        bgr = st["neutral"]; g = bgr[1]
        p["wb"] = np.clip(np.array([g / max(bgr[0], 1e-3), 1.0,
                                    g / max(bgr[2], 1e-3)], np.float32), 0.92, 1.10)
    # 2) Auto black/white point: hue-preserving affine, partial pull to targets.
    p1, p99 = st["p1"], st["p99"]
    nb = p1 + (_GRADE_BLACK - p1) * _GRADE_LEVELS_STRENGTH
    nw = p99 + (_GRADE_WHITE - p99) * _GRADE_LEVELS_STRENGTH
    a = (nw - nb) / max(p99 - p1, 1.0)
    p["a"] = float(np.clip(a, 0.95, 1.8)); p["b"] = float(nb - a * p1)
    # 3) Midtone gamma toward target (estimated after the levels stretch).
    med = float(np.clip(st["p50"] * p["a"] + p["b"], 1, 254))
    p["gamma"] = float(np.clip(
        np.log(med / 255.0) / np.log(_GRADE_MED_TARGET / 255.0), 0.85, 1.2))
    # 4) Adaptive saturation: lift dull clips toward the band, gently pull back
    # over-saturated ones, leave in-band clips untouched.
    s0 = st["sat"]
    if s0 < _GRADE_SAT_TARGET:
        p["sat"] = float(np.clip(_GRADE_SAT_TARGET / max(s0, 1.0), 1.0, 1.30))
    elif s0 > 150.0:
        p["sat"] = float(np.clip(150.0 / s0, 0.85, 1.0))
    # 5) Adaptive sharpen amount: soft clips get more, crisp ones less.
    p["sharpen"] = float(np.clip(_SHARPEN_REF / max(st.get("sharp", 100.0), 1.0),
                                 _SHARPEN_MIN, _SHARPEN_MAX))
    return p


def apply_grade(bgr, p):
    """Apply grade params to a BGR frame → graded BGR (uint8)."""
    x = bgr.astype(np.float32) * p["wb"][None, None, :]
    x = np.clip(x * p["a"] + p["b"], 0, 255)
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** (1.0 / p["gamma"])
    if abs(p["sat"] - 1.0) > 1e-3:
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * p["sat"], 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # Unsharp mask (clarity), threshold-gated so flat areas / noise aren't lifted.
    amt = p.get("sharpen", 0.0)
    if amt > 1e-3:
        blur = cv2.GaussianBlur(x, (0, 0), _SHARPEN_SIGMA)
        diff = x - blur
        diff *= (np.abs(diff) >= _SHARPEN_THRESH)
        x = x + amt * diff
    return np.clip(x, 0, 255).astype(np.uint8)


def load_sam2(device="mps"):
    """Load SAM2 video predictor."""
    from sam2.build_sam import build_sam2_video_predictor
    checkpoint = _sam2_checkpoint()
    cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    return build_sam2_video_predictor(cfg, checkpoint, device=device)


BIREFNET_CHECKPOINT = "ZhengPeng7/BiRefNet-matting"  # was "ZhengPeng7/BiRefNet"
# Edge refine spike (2026-06-19, tech/BIREFNET_EDGE_REFINE_SPIKE.md §4) proved:
#   no-sigmoid = hard clip(logits, 0, 1) → "scissor-cut" fur, loses wisps
#   sigmoid alone = green halo on chroma key (soft band 5-7× wider)
#   sigmoid + FB_blur_fusion = natural fur + no halo → clear winner across 4 cases
# "Head built-in alpha mapping" was wrong — matting logits are [-20,+14] and
# MUST sigmoid to get real soft matte. The previous no-sig "good" result was
# just a lucky coincidence on IMG_0847 neck that hid the issue.
USE_SIGMOID = True

# === Edge refine: FB_blur_fusion foreground estimation ===
# Germer et al. fast multi-level foreground estimation (also used by BiRefNet's
# HF handler.py::refine_foreground). Takes a soft sigmoid alpha + RGB and
# unmixes the background color from soft edge pixels. ~0 cost (pure cv2.blur).
#
# r (blur radius) is scaled to the PET's bbox width, NOT the full frame — the
# edge-mixing scale belongs to the subject, not the canvas. (The first cut scaled
# r to full-frame width, so a small pet in a 4K frame got an over-large r and a
# frame-filling pet in a small clip got a tiny one.) 0.18 × a ~510px pet ≈ 92,
# matching the A/B-validated sweet spot (r≈90 beat r≈35; see spike result §3.3).
FG_ESTIMATE = True
FG_RADIUS_RATIO = 0.18   # r = max(FG_RADIUS_MIN, int(pet_bbox_w * FG_RADIUS_RATIO))
FG_RADIUS_MIN = 15



def load_birefnet(device="mps", resolution=1024):
    """Load BiRefNet with fp16 at specified resolution."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_CHECKPOINT, trust_remote_code=True, dtype=torch.float32
    )
    model = model.half().to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Warm up
    with torch.no_grad():
        _ = model(torch.randn(1, 3, resolution, resolution).to(device).half())

    return model, transform


def _get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


# === HDR (HLG / PQ, BT.2020) → SDR extraction ===
#
# iPhone "HDR video" is HLG (transfer arib-std-b67) or Dolby-Vision PQ
# (smpte2084), 10-bit, BT.2020 primaries. The old extraction did a naive
# YUV→RGB with NO tone-mapping, so the HLG curve and wide gamut were dumped
# straight into an SDR JPEG → washed-out / grey picture AND wrong hues (BT.2020
# reds/greens read as if sRGB, e.g. a cat's eye colour shifts). Confirmed on
# IMG_0847 (2026-06-18): naive sat=32 vs tone-mapped sat=59.
#
# We can't rely on `zscale`/`libplacebo` (this Homebrew ffmpeg ships without
# libzimg/libplacebo, and end-user ffmpeg is unknown). So ffmpeg only gives us
# 16-bit BT.2020 RGB (gamma-domain) and we do the colour science in numpy:
#   HLG/PQ inverse curve → BT.2020→BT.709 gamut → Hable tone-map → sRGB OETF.

# BT.2020 → BT.709 linear-RGB gamut matrix.
_M_2020_TO_709 = np.array([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
])
HDR_EXPOSURE = 1.6  # scene-linear gain before tone-map (tuned on IMG_0847)


def _probe_color(video_path):
    """Return (color_transfer, color_primaries) lowercased, or ('', '')."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer,color_primaries",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=5).stdout.split()
        transfer = out[0].lower() if len(out) > 0 else ""
        primaries = out[1].lower() if len(out) > 1 else ""
        return transfer, primaries
    except Exception:
        return "", ""


def _is_hdr(transfer, primaries):
    """HLG (arib-std-b67) or PQ (smpte2084), or any BT.2020-primaries source."""
    return (transfer in ("arib-std-b67", "smpte2084")
            or primaries in ("bt2020", "bt2020-10", "bt2020-12"))


def _hable(x):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _tonemap_bt2020_to_srgb(rgb16, transfer):
    """16-bit BT.2020 HLG/PQ-encoded RGB (uint16, RGB order) → 8-bit sRGB BGR."""
    x = rgb16.astype(np.float64) / 65535.0
    if transfer == "smpte2084":  # PQ EOTF (normalised so ~100-nit diffuse ≈ 1)
        m1, m2 = 0.1593017578125, 78.84375
        c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
        xp = np.power(np.clip(x, 0, 1), 1.0 / m2)
        lin = np.power(np.clip(xp - c1, 0, None) / (c2 - c3 * xp), 1.0 / m1)
        lin = lin * 100.0  # 0..10000 nits → 0..100 (diffuse-white reference)
    else:  # HLG inverse OETF (ARIB STD-B67)
        a = 0.17883277
        b = 1.0 - 4.0 * a
        c = 0.5 - a * np.log(4.0 * a)
        lin = np.where(x <= 0.5, x * x / 3.0, (np.exp((x - c) / a) + b) / 12.0)

    lin709 = np.clip(lin @ _M_2020_TO_709.T, 0, None) * HDR_EXPOSURE
    tm = np.clip(_hable(lin709) / _hable(np.float64(11.2)), 0, 1)
    srgb = np.where(tm <= 0.0031308, 12.92 * tm, 1.055 * np.power(tm, 1 / 2.4) - 0.055)
    out = (np.clip(srgb, 0, 1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _extract_frames_core(video_path, output_dir, fps, ss=None, dur=None):
    """Shared extraction. SDR → fast JPEG path; HDR → 16-bit + numpy tone-map.

    Returns sorted list of frame_*.jpg paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    for old in glob.glob(os.path.join(output_dir, "frame_*")):
        os.remove(old)

    pre = ["ffmpeg", "-y"]
    if ss is not None:
        pre += ["-ss", str(ss)]
    pre += ["-i", video_path]
    if dur is not None:
        pre += ["-t", str(dur)]

    transfer, primaries = _probe_color(video_path)
    if _is_hdr(transfer, primaries):
        # 16-bit BT.2020 PNG intermediate, then tone-map each frame in numpy.
        tmp = tempfile.mkdtemp(prefix="hdr_")
        try:
            subprocess.run(
                pre + ["-vf", f"yadif,fps={fps}", "-pix_fmt", "rgb48be",
                       os.path.join(tmp, "f_%04d.png")],
                capture_output=True)
            for i, p in enumerate(sorted(glob.glob(os.path.join(tmp, "f_*.png"))), 1):
                rgb16 = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED),
                                     cv2.COLOR_BGR2RGB)
                bgr8 = _tonemap_bt2020_to_srgb(rgb16, transfer)
                cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"),
                            bgr8, [cv2.IMWRITE_JPEG_QUALITY, 95])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        subprocess.run(
            pre + ["-vf", f"yadif,fps={fps}", "-q:v", "2",
                   os.path.join(output_dir, "frame_%04d.jpg")],
            capture_output=True)

    return sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))


def extract_frames_range(video_path, output_dir, start_time, duration, fps=10):
    """Extract frames from a specific time range (HDR-aware)."""
    return _extract_frames_core(video_path, output_dir, fps,
                                ss=start_time, dur=duration)


def extract_frames(video_path, output_dir, fps=10):
    """Extract frames from the whole video (HDR-aware)."""
    return _extract_frames_core(video_path, output_dir, fps)


# === Video stabilization (Part A — tech/PET_STABILIZATION_SPIKE.md) ===
# Trajectory smoothing for handheld camera shake. MUST run before SAM2/BiRefNet
# (see spike §4 red lines) so tracking & matting see stable frames — that alone
# removes most edge shimmer (B), per spike §0 ("A is cause, B is mostly effect").
#
# Algorithm:
#   1) estimateAffinePartial2D per frame on optical-flow-tracked features → [dx,dy,da]
#   2) cumulative camera trajectory
#   3) moving-average smooth the trajectory (NOT zero it — preserves intentional pans)
#   4) warp each frame by (smoothed - original)
#   5) light zoom (1.04) covers border-replicate edge artifacts
#
# Skip gate (shake_thresh): if mean inter-frame residual is tiny, the source is
# already stable — skip (don't lose canvas, don't waste ms).
#
# IMPORTANT: "subject-locked" stabilization (align every pet centroid) is FORBIDDEN
# here — would erase natural pet motion. Use trajectory smoothing only.

STABILIZE = True
STABILIZE_SMOOTH_RADIUS = 30   # ~3s @ 10fps; 15/30/45 swept in spike
STABILIZE_ZOOM = 1.04          # cover border artifacts; small enough not to hurt canvas
# PARTC §1 (2026-06-20): gate raised 1.2 → 2.5. IMG_0847 source residual is
# ~1.9px; the old 1.2 gate let Part A warp every frame on a near-stable source
# (INTER_LINEAR adds its own sub-pixel jitter). At 2.5, Part A skips on these
# clips — same effect as turning it off, no extra wobble introduced. Logic
# unchanged.
STABILIZE_SHAKE_THRESH = 2.5


def _estimate_interframe_shake(gray_frames):
    """Mean |dx|,|dy| across consecutive frames (goodFeaturesToTrack + PyrLK).
    Proxy for handheld shake. Used both by the skip gate and by A/B reporting."""
    n = len(gray_frames)
    if n < 2:
        return 0.0, np.zeros((max(n, 1), 3), dtype=np.float32)
    tr = []
    prev = gray_frames[0]
    prev_pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 30, blockSize=3)
    for i in range(1, n):
        cur = gray_frames[i]
        if prev_pts is None or len(prev_pts) < 8:
            tr.append([0.0, 0.0, 0.0])
            prev = cur
            prev_pts = cv2.goodFeaturesToTrack(cur, 200, 0.01, 30, blockSize=3)
            continue
        cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, prev_pts, None)
        ok = (st.flatten() == 1) if st is not None else np.zeros(prev_pts.shape[0], bool)
        if not np.any(ok):
            tr.append([0.0, 0.0, 0.0])
        else:
            m, _ = cv2.estimateAffinePartial2D(prev_pts[ok], cur_pts[ok])
            if m is None:
                tr.append([0.0, 0.0, 0.0])
            else:
                tr.append([m[0, 2], m[1, 2], float(np.arctan2(m[1, 0], m[0, 0]))])
        prev = cur
        prev_pts = cv2.goodFeaturesToTrack(cur, 200, 0.01, 30, blockSize=3)
    tr = np.asarray(tr, np.float32)  # n-1 × 3
    # include frame 0 (zero) so callers can index by frame number
    tr_full = np.zeros((n, 3), np.float32)
    tr_full[1:] = tr
    mean_dxdy = float(np.mean(np.abs(tr[:, :2]))) if len(tr) else 0.0
    return mean_dxdy, tr_full


def stabilize_frames(frame_paths, smooth_radius=STABILIZE_SMOOTH_RADIUS,
                     zoom=STABILIZE_ZOOM, shake_thresh=STABILIZE_SHAKE_THRESH,
                     progress_callback=None):
    """Trajectory-smooth stabilization (Part A). Mutates frame_paths in place.

    Returns:
        (frame_paths, did_stabilize, report_dict)

    report_dict keys: enabled, before_mean_dxdy_px, after_mean_dxdy_px,
    residual_reduction_pct, smooth_radius, zoom, skipped (bool + reason).
    """
    report = {
        "enabled": STABILIZE,
        "before_mean_dxdy_px": 0.0,
        "after_mean_dxdy_px": 0.0,
        "residual_reduction_pct": 0.0,
        "smooth_radius": int(smooth_radius),
        "zoom": float(zoom),
        "shake_thresh_px": float(shake_thresh),
        "skipped": False,
        "skip_reason": None,
    }
    if not STABILIZE or len(frame_paths) < 3:
        report["skipped"] = True
        report["skip_reason"] = "disabled" if not STABILIZE else "too_few_frames"
        return list(frame_paths), False, report

    # 1) read grayscale + measure pre-shake
    gray = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in frame_paths]
    h, w = gray[0].shape
    before, tr = _estimate_interframe_shake(gray)
    report["before_mean_dxdy_px"] = before

    if before < shake_thresh:
        report["skipped"] = True
        report["skip_reason"] = f"already_stable({before:.2f}px<{shake_thresh:.2f})"
        # No callback emit on skip — main() emits a single "skipped: ..." progress
        # itself (avoids duplicate throttled progress events on the CLI).
        return list(frame_paths), False, report

    # 2) cumulative trajectory  →  3) smooth with moving average
    # tr has shape n × 3 (frame 0 = 0, frames 1..n-1 = incremental dx/dy/da).
    # Spike's reference algorithm indexes `diff` by frame number, so cumsum the
    # full tr (length n) here — not the n-1 incremental slice.
    traj = np.cumsum(tr, axis=0)            # n × 3  (cumulative dx, dy, da)
    k_len = 2 * int(smooth_radius) + 1
    k = np.ones(k_len) / k_len
    n = traj.shape[0]
    if n >= k_len:
        # np.convolve mode='same' returns max(M,N) — trim back to traj length.
        sm = np.stack([np.convolve(traj[:, c], k, mode='same')[:n]
                       for c in range(3)], axis=1)
    else:
        # Short clip: kernel longer than signal. Reflect-pad signal so the box
        # filter sees reasonable neighbors at the boundaries, then take the
        # centre n samples. Without this the smoothed edges are biased toward
        # the centre (proven with n=5, k_len=11 → wrong vs reference).
        pad = k_len // 2
        sm_cols = []
        for c in range(3):
            xp = np.pad(traj[:, c], (pad, pad), mode='edge')
            full = np.convolve(xp, k, mode='full')
            sm_cols.append(full[pad:pad + n])
        sm = np.stack(sm_cols, axis=1)
    diff = sm - traj                        # correction per frame (frame 0 = 0 by construction)

    # 4) warp every frame by its correction + light zoom
    Z = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom) if zoom and zoom != 1.0 else None
    for i, p in enumerate(frame_paths):
        d = diff[i] if i > 0 else np.zeros(3, dtype=np.float32)
        dx, dy, da = float(d[0]), float(d[1]), float(d[2])
        ca, sa = np.cos(da), np.sin(da)
        M = np.array([[ca, -sa, dx], [sa, ca, dy]], np.float32)
        warped = cv2.warpAffine(cv2.imread(p), M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        if Z is not None:
            warped = cv2.warpAffine(warped, Z, (w, h), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(p, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if progress_callback and (i % 10 == 0 or i == len(frame_paths) - 1):
            progress_callback("stabilize", i + 1, len(frame_paths),
                             f"frame {i+1}/{len(frame_paths)} warped")

    # 5) measure post-shake
    gray_after = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in frame_paths]
    after, _ = _estimate_interframe_shake(gray_after)
    report["after_mean_dxdy_px"] = after
    report["residual_reduction_pct"] = (
        max(0.0, (before - after) / max(before, 1e-6)) * 100.0
    )
    if progress_callback:
        progress_callback("stabilize", len(frame_paths), len(frame_paths),
                         f"residual {before:.2f}px → {after:.2f}px "
                         f"(-{report['residual_reduction_pct']:.1f}%)")
    return list(frame_paths), True, report


def pass1_sam2(frames, click_point, progress_callback=None):
    """Pass 1: SAM2 tracking → logit collection.

    Runs SAM2 once on ALL frames and returns the per-frame tracking logits.
    Preview generation is NOT done here — the preview is built from the real
    BiRefNet output of the first N frames in pass 2 (see main), so what the
    user previews is identical to the final result (same model).

    Args:
        frames: list of frame paths
        click_point: (x, y) tuple for the pet location
        progress_callback: optional callback(phase, current, total, detail)

    Returns:
        dict mapping frame_idx → raw logit (float32)
    """
    import torch

    emit({"type": "phase", "name": "sam2_track", "detail": "loading SAM2"})
    sam2 = load_sam2()

    # Prepare frames for SAM2 (needs numbered files)
    sam2_dir = tempfile.mkdtemp(prefix="sam2_")
    for i, f in enumerate(frames):
        shutil.copy(f, f"{sam2_dir}/{i:05d}.jpg")

    # Initialize and add prompt (bbox if available, else point)
    inference_state = sam2.init_state(video_path=sam2_dir)
    cx, cy = click_point

    # Try to get pet bbox for better SAM2 tracking
    try:
        from pipeline.pet_detector import detect_pet_bbox
        pet_bbox = detect_pet_bbox(frames[0])
    except Exception:
        pet_bbox = None

    if pet_bbox:
        # Use bbox prompt (much better tracking)
        sam2.add_new_points_or_box(
            inference_state, 0, 1,
            box=np.array(pet_bbox),
        )
        emit({"type": "progress", "phase": "sam2_track",
              "detail": f"bbox prompt: {pet_bbox}"})
    else:
        # Fallback to point prompt
        sam2.add_new_points_or_box(
            inference_state, 0, 1,
            np.array([[cx, cy]]), np.array([1]),
        )
        emit({"type": "progress", "phase": "sam2_track",
              "detail": f"point prompt: ({cx}, {cy})"})

    # Propagate once on all frames, collecting logits
    logits = {}

    t0 = time.time()
    for frame_idx, obj_ids, mask_logits in sam2.propagate_in_video(inference_state):
        # Always store logit
        logit = mask_logits[0].cpu().numpy().squeeze()
        logits[frame_idx] = logit

        if progress_callback:
            progress_callback("sam2_track", frame_idx + 1, len(frames),
                              f"frame_{frame_idx:04d}")

    elapsed = time.time() - t0
    emit({"type": "progress", "phase": "sam2_track",
          "current": len(frames), "total": len(frames),
          "detail": f"done ({elapsed:.1f}s)"})

    # Free SAM2 model from GPU memory
    del sam2
    torch.mps.empty_cache()

    # Cleanup
    shutil.rmtree(sam2_dir, ignore_errors=True)

    return logits


def get_roi_from_logit(logit, h, w, margin=0.10):
    """Get ROI bounding box from SAM2 logit mask.

    Args:
        logit: raw logit array (float32)
        h, w: original frame dimensions
        margin: expansion margin as fraction of bbox size (default 10%)

    Returns:
        (x1, y1, x2, y2) or None if no foreground
    """
    # Convert logit to binary mask
    soft = 1.0 / (1.0 + np.exp(-logit * 2.0))
    binary = (soft > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get union bbox of all contours
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + cw)
        y_max = max(y_max, y + ch)

    # Expand by margin
    bw = x_max - x_min
    bh = y_max - y_min
    dx = int(bw * margin)
    dy = int(bh * margin)
    x1 = max(0, x_min - dx)
    y1 = max(0, y_min - dy)
    x2 = min(w, x_max + dx)
    y2 = min(h, y_max + dy)

    # Ensure minimum size (at least 64x64 for BiRefNet)
    if (x2 - x1) < 64 or (y2 - y1) < 64:
        # Expand to center with minimum size
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x1 = max(0, cx - 32)
        y1 = max(0, cy - 32)
        x2 = min(w, cx + 32)
        y2 = min(h, cy + 32)

    return (x1, y1, x2, y2)


def _birefnet_alpha(model, norm_transform, img_bgr, resolution):
    """Aspect-preserving BiRefNet alpha for a BGR image/crop.

    The pet crop is letterboxed to a square (BORDER_REPLICATE, so no fake
    foreground edge) BEFORE the resize to `resolution`, then the padding is
    removed. This avoids the square-squash distortion (tall/wide pets were
    stretched up to ~50%), which blurred whiskers and misaligned the matte
    edge against the RGB edge (the red/magenta fringe). Same inference cost
    as the old square resize.

    Returns uint8 alpha (H, W) matching img_bgr.
    """
    import torch
    from PIL import Image

    h, w = img_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    sq = cv2.copyMakeBorder(img_bgr, top, side - h - top, left, side - w - left,
                            cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if side > resolution else cv2.INTER_LINEAR
    sq = cv2.resize(sq, (resolution, resolution), interpolation=interp)

    inp = norm_transform(
        Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to("mps").half()
    with torch.no_grad():
        pred = model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        if USE_SIGMOID:
            pred = torch.sigmoid(pred)

    alpha_sq = torch.nn.functional.interpolate(
        pred, (side, side), mode="bilinear", align_corners=False
    ).squeeze().cpu().float().numpy()
    alpha = alpha_sq[top:top + h, left:left + w]
    return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)


def _fb_step(image, F, B, a, r):
    """One FB_blur_fusion iteration. image HxWx3 [0,1], a HxWx1 [0,1], r blur radius."""
    ba = cv2.blur(a[:, :, 0], (r, r))[:, :, None]
    bF = cv2.blur(F * a, (r, r)) / (ba + 1e-5)
    bB = cv2.blur(B * (1 - a), (r, r)) / ((1 - ba) + 1e-5)
    F = np.clip(bF + a * (image - a * bF - (1 - a) * bB), 0, 1)
    return F, bB


def fb_blur_fusion_fg(image, alpha, r=35):
    """Fast multi-level foreground estimation (Germer et al.).

    Args:
        image: HxWx3 float32/float64 in [0,1] (RGB or BGR — order doesn't matter, only
            relative ratio to alpha matters for the unmixing).
        alpha: HxW float32 in [0,1].
        r: blur radius. Full-frame recipe uses ~90; ROI crops need ~25-45.

    Returns:
        F: HxWx3 float [0,1] — "clean" foreground with the background color removed
        from soft edge pixels. ~0 cost (two cv2.blur passes).
    """
    a = alpha[:, :, None]
    F, blurB = _fb_step(image, image, image, a, r)
    F, _ = _fb_step(image, F, blurB, a, r)
    return F


def pass2_birefnet(frames, sam2_logits, output_dir, progress_callback=None,
                   preview_limit=None, on_preview_ready=None):
    """Pass 2: BiRefNet refinement with ROI cropping → final alpha.

    Uses SAM2 mask to crop each frame to the pet region before feeding
    to BiRefNet. This reduces pixel count by 50%+ and improves edge
    quality (same model resolution focused on the pet).

    Args:
        frames: list of frame paths
        sam2_logits: dict from pass1_sam2 (frame_idx → raw logit)
        output_dir: directory to save final RGBA PNGs
        progress_callback: optional callback(phase, current, total, detail)
        preview_limit: if set with on_preview_ready, fire the callback once the
            first N frames are matted (these ARE final frames — preview == final)
        on_preview_ready: callback(n) invoked after the first N frames are saved
    """
    import torch
    from torchvision import transforms

    # 1024 for ROI crops: crisper edge/fur matte than 768 (sharper alpha gradient,
    # ~10% tighter soft-edge band, more defined fur wisps). Costs ~1.8x the matte
    # time (≈141s vs ≈80s for a 72-frame clip) since cost scales with resolution².
    # 1536 was slower AND worse (over-resolution noise). The gain is real but
    # bounded by SOURCE resolution — a pet that's small in frame is detail-limited
    # regardless. This is the edge-precision lever for the soft-edge report.
    roi_resolution = 1024
    emit({"type": "phase", "name": "birefnet_refine",
          "detail": f"loading BiRefNet (ROI {roi_resolution})"})
    model, _ = load_birefnet(resolution=roi_resolution)

    # Normalize-only transform; resize is done aspect-preserving in
    # _birefnet_alpha (not via the model's bundled square Resize).
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    h, w = cv2.imread(frames[0]).shape[:2]
    dilate_size = max(3, int(w * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

    os.makedirs(output_dir, exist_ok=True)

    # Adaptive colour grade: measured ONCE over a sample of the clip, applied
    # identically to every saved frame (stable → no flicker). Near no-op on a
    # well-shot clip; standardises exposure / white balance / saturation on the
    # rest. See AUTO_GRADE notes for targets + clamps.
    grade = (compute_grade(measure_grade_stats(frames, logits=sam2_logits))
             if AUTO_GRADE else None)
    if grade is not None:
        emit({"type": "progress", "phase": "auto_grade",
              "detail": (f"wb={np.round(grade['wb'], 3).tolist()} "
                         f"a={grade['a']:.2f} b={grade['b']:.0f} "
                         f"gamma={grade['gamma']:.2f} sat={grade['sat']:.2f}")})

    t0 = time.time()
    total_pixels_saved = 0

    # PARTC §2.3: collect per-frame alpha + sam2_binary for temporal smoothing
    # after the BiRefNet loop. FB fusion + PNG write happen on the SMOOTHED
    # alpha (replaces per-frame loop body in the integration below).
    alpha_final_list = []
    sam2_binary_list = []
    img_list = []  # colour-graded RGB for FB fusion after smoothing
    bgr_pre_fb_list = []  # raw RGB (pre-FB) for FB fusion input

    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        # Missing logit: no SAM2 data → use full gate (BiRefNet alpha unchanged)
        logit = sam2_logits.get(i, None)

        # Resize logit if needed (guard against missing/None logit — a dropped
        # SAM2 frame must NOT crash the whole pass)
        if logit is not None and logit.shape != (h, w):
            logit = cv2.resize(logit, (w, h), interpolation=cv2.INTER_LINEAR)

        # Get ROI from SAM2 mask (None logit → no ROI, falls back to full image)
        roi = get_roi_from_logit(logit, h, w, margin=0.10) if logit is not None else None

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_w = x2 - x1
            roi_h = y2 - y1

            # Aspect-preserving BiRefNet on the pet crop
            alpha_crop = _birefnet_alpha(model, norm_transform,
                                         img[y1:y2, x1:x2], roi_resolution)
            birefnet_alpha = np.zeros((h, w), dtype=np.uint8)
            birefnet_alpha[y1:y2, x1:x2] = alpha_crop

            total_pixels_saved += h * w - roi_h * roi_w
        else:
            # No ROI found → aspect-preserving full-frame inference
            birefnet_alpha = _birefnet_alpha(model, norm_transform, img, roi_resolution)

        # SAM2 gate: binary → dilate → blur (original design, verified 06/07)
        # Gate = 1.0 inside pet + margin, 0.0 outside, blurred at edge
        # Preserves BiRefNet's soft alpha inside the pet (gate=1.0 is no-op)
        if logit is not None:
            sam2_soft = 1.0 / (1.0 + np.exp(-logit * 2.0))
            sam2_binary = (sam2_soft > 0.5).astype(np.uint8)
            sam2_dilated = cv2.dilate(sam2_binary, kernel, iterations=1)
            sam2_gate = np.clip(
                cv2.GaussianBlur(sam2_dilated.astype(np.float32), (0, 0), 3.0), 0, 1
            )
        else:
            # No SAM2 data → full gate (BiRefNet alpha unchanged)
            sam2_gate = np.ones((h, w), dtype=np.float32)
            sam2_binary = np.ones((h, w), dtype=np.uint8)  # always "in" → smooth allowed everywhere

        # Combine
        alpha_final = (birefnet_alpha.astype(float) * sam2_gate).astype(np.uint8)
        alpha_final = cv2.GaussianBlur(alpha_final, (0, 0), 0.5)

        # PARTC §2.3: collect for temporal smoothing; FB fusion + write happens
        # AFTER the loop using the smoothed alpha.
        alpha_final_list.append(alpha_final)
        sam2_binary_list.append(sam2_binary)

        bgr_out = apply_grade(img, grade) if grade is not None else img
        bgr_pre_fb_list.append(bgr_out)

        if progress_callback:
            progress_callback("birefnet_refine", i + 1, len(frames),
                              f"frame_{i:04d}")

        # PARTC §2.3 preview bookkeeping — the old design fired
        # on_preview_ready per-frame, but FB fusion + PNG write now happen
        # AFTER the BiRefNet loop (so alpha is post-temporal-smooth when
        # written). We remember the FIRST preview_limit indices instead and
        # trigger on_preview_ready at the end (see below).
        if (on_preview_ready is not None and preview_limit
                and i + 1 == min(preview_limit, len(frames))):
            # Defer — see bottom of pass2
            pass

    elapsed = time.time() - t0
    roi_pct = total_pixels_saved / (h * w * len(frames)) * 100 if frames else 0
    emit({"type": "progress", "phase": "birefnet_refine",
          "current": len(frames), "total": len(frames),
          "detail": f"done ({elapsed:.1f}s, ROI saved {roi_pct:.0f}% pixels)"})

    # PARTC §2.3: motion-gated temporal smoothing across all frames, then
    # FB fusion + PNG write. Smoothing before FB fusion matters: FB reads
    # alpha to decide which pixels to unmix, and we want that alpha to be
    # the SHIMMER-REMOVED version, not the per-frame raw.
    if TEMPORAL_ALPHA_SMOOTH and len(alpha_final_list) >= 3:
        emit({"type": "phase", "name": "temporal_smooth",
              "detail": f"motion-gated median, window ±{TEMPORAL_WINDOW} frames, "
                        f"soft band only, wrap at loop seam"})
        try:
            def _ts_prog(phase, current, total, detail):
                emit({"type": "progress", "phase": phase,
                      "current": current, "total": total, "detail": detail})
            t1 = time.time()
            smoothed = motion_gated_temporal_smooth(
                alpha_final_list, sam2_binary_list,
                window=TEMPORAL_WINDOW,
                progress_callback=_ts_prog,
            )
            # shimmer before/after measurement
            def _shimmer(alphas):
                K = np.ones((3, 3), np.uint8)
                disps = []
                for j in range(1, len(alphas)):
                    m0 = (alphas[j - 1] > 128).astype(np.uint8)
                    m1 = (alphas[j] > 128).astype(np.uint8)
                    e0 = cv2.dilate(m0, K) - m0
                    inv = 1 - m1
                    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
                    if np.any(e0 > 0):
                        disps.append(float(dt[e0 > 0].mean()))
                    else:
                        disps.append(0.0)
                d = np.array(disps)
                return float(d.mean()), float(np.percentile(d, 95)), float(d.max())
            sm_before_mean, sm_before_p95, _ = _shimmer(alpha_final_list)
            sm_after_mean, sm_after_p95, _ = _shimmer(smoothed)
            emit({"type": "progress", "phase": "temporal_smooth",
                  "detail": f"edge shimmer mean {sm_before_mean:.2f}→{sm_after_mean:.2f}px "
                            f"p95 {sm_before_p95:.2f}→{sm_after_p95:.2f}px "
                            f"in {time.time()-t1:.2f}s"})
            alpha_final_list = smoothed
            emit({"type": "temporal_smooth_done",
                  "before_mean": sm_before_mean, "after_mean": sm_after_mean,
                  "before_p95": sm_before_p95, "after_p95": sm_after_p95,
                  "window": int(TEMPORAL_WINDOW)})
        except Exception as e:
            emit({"type": "progress", "phase": "temporal_smooth",
                  "detail": f"failed ({type(e).__name__}: {e}), keeping raw alpha"})
            # alpha_final_list unchanged → fall through with raw

    # FB fusion + write PNG — using the (possibly smoothed) alpha_final.
    for i, (alpha_final, bgr_out) in enumerate(zip(alpha_final_list, bgr_pre_fb_list)):
        # Optional edge refine: remove the background color leaked into soft
        # edge pixels (BiRefNet's bundled handler.py does this, we didn't).
        # Runs on the *graded* RGB so the unmixing matches what the user sees.
        # r scales with the PET bbox width (subject-relative, not canvas-relative).
        # Cost: two cv2.blur per frame, ~ms. See tech/BIREFNET_EDGE_REFINE_SPIKE.md §3.
        if FG_ESTIMATE and alpha_final.max() > 0:
            ys_fg, xs_fg = np.where(alpha_final > 0)
            pet_w = int(xs_fg.max() - xs_fg.min()) if xs_fg.size else w
            fg_r = max(FG_RADIUS_MIN, int(pet_w * FG_RADIUS_RATIO))
            rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            a = alpha_final.astype(np.float32) / 255.0
            F = fb_blur_fusion_fg(rgb, a, r=fg_r)
            bgr_out = cv2.cvtColor(
                (np.clip(F, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )

        bgra = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_final
        cv2.imwrite(f"{output_dir}/frame_{i:04d}.png", bgra)

        # Preview: first N frames — fire callback when Nth is written
        if (on_preview_ready is not None and preview_limit
                and i + 1 == min(preview_limit, len(frames))):
            on_preview_ready(i + 1)

    elapsed = time.time() - t0
    roi_pct = total_pixels_saved / (h * w * len(frames)) * 100 if frames else 0
    emit({"type": "progress", "phase": "birefnet_refine",
          "current": len(frames), "total": len(frames),
          "detail": f"done ({elapsed:.1f}s, ROI saved {roi_pct:.0f}% pixels)"})


# === Quality Feedback Loop ===

def compute_frame_metrics(sam2_logit, birefnet_alpha, yolo_bbox, prev_mask, h, w):
    """Compute 6 quality metrics for a single frame.

    All metrics are derived from data already computed — no extra model calls.
    Total overhead: <5ms per frame.

    Returns dict with:
        coverage: mask coverage percentage (2%-70% = green)
        object_score: SAM2 confidence (from logit statistics)
        mask_iou: overlap with previous frame mask (stability)
        anchor_iou: overlap with detector bbox on first frame (correctness)
        bbox_stability: area/center change rate vs previous frame
        soft_edge_ratio: percentage of edge pixels (5%-20% = good)
        status: "green", "yellow", or "red"
    """
    # 1. Mask coverage (use BiRefNet alpha if SAM2 logit missing)
    if sam2_logit is not None:
        sam2_mask = (1.0 / (1.0 + np.exp(-sam2_logit * 2.0)) > 0.5).astype(np.uint8)
        object_score = float(np.mean(np.abs(sam2_logit)))
    else:
        sam2_mask = (birefnet_alpha > 128).astype(np.uint8)
        object_score = 0.0
    coverage = sam2_mask.sum() / (h * w) * 100

    # 3. Adjacent frame mask IoU
    mask_iou = 0.0
    if prev_mask is not None:
        intersection = (sam2_mask & prev_mask).sum()
        union = (sam2_mask | prev_mask).sum()
        mask_iou = intersection / max(union, 1)

    # 4. Anchor containment (how much of the SAM2 mask falls inside the detector
    # pet box). IoU against the rectangular bbox is the wrong metric — an
    # irregular cat fills only ~50-70% of its bounding rectangle, so IoU is
    # ~0.5 even when tracking is perfect, producing false red alerts. What we
    # actually want to know is "is the tracked mask the detected pet?" =
    # fraction of the mask contained in the box. ~1.0 when correct, low only
    # if SAM2 wandered onto something outside the pet box.
    anchor_iou = 1.0
    if yolo_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        intersection = (sam2_mask & bbox_mask).sum()
        anchor_iou = intersection / max(int(sam2_mask.sum()), 1)

    # 4. Bbox stability (area/center change)
    bbox_stability = 1.0  # 1.0 = perfectly stable
    if prev_mask is not None:
        # Current bbox
        ys, xs = np.where(sam2_mask > 0)
        if len(ys) > 0:
            curr_area = len(ys)
            curr_cx, curr_cy = xs.mean(), ys.mean()
        else:
            curr_area, curr_cx, curr_cy = 0, 0, 0

        # Previous bbox
        pys, pxs = np.where(prev_mask > 0)
        if len(pys) > 0:
            prev_area = len(pys)
            prev_cx, prev_cy = pxs.mean(), pys.mean()
        else:
            prev_area, prev_cx, prev_cy = 0, 0, 0

        if prev_area > 0:
            area_change = abs(curr_area - prev_area) / prev_area
            center_dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            center_change = center_dist / max(w, h)
            bbox_stability = 1.0 - min(1.0, area_change + center_change)

    # 6. Soft edge ratio
    alpha = birefnet_alpha if birefnet_alpha is not None else (sam2_mask * 255)
    edge_pixels = ((alpha > 20) & (alpha < 240)).sum()
    soft_edge_ratio = edge_pixels / max((alpha > 0).sum(), 1) * 100

    # Classify status
    status = "green"
    reasons = []

    if coverage < 2:
        status = "red"
        reasons.append(f"coverage={coverage:.1f}% (<2%)")
    elif coverage > 70:
        status = "yellow"
        reasons.append(f"coverage={coverage:.1f}% (>70%)")

    if mask_iou < 0.7 and prev_mask is not None:
        if status == "green":
            status = "yellow"
        reasons.append(f"mask_iou={mask_iou:.2f} (<0.7)")

    if bbox_stability < 0.7:
        if status == "green":
            status = "yellow"
        reasons.append(f"bbox_stability={bbox_stability:.2f} (<0.7)")

    if anchor_iou < 0.5 and yolo_bbox is not None:
        status = "red"
        reasons.append(f"anchor_iou={anchor_iou:.2f} (<0.5)")

    return {
        "coverage": round(coverage, 1),
        "object_score": round(object_score, 2),
        "mask_iou": round(mask_iou, 3),
        "anchor_iou": round(anchor_iou, 3),
        "bbox_stability": round(bbox_stability, 3),
        "soft_edge_ratio": round(soft_edge_ratio, 1),
        "status": status,
        "reasons": reasons,
    }


def run_quality_feedback(alphas, logits, yolo_bbox, h, w, frame_paths=None, emit_fn=None):
    """Run quality feedback loop on all frames.

    Classifies each frame as green/yellow/red and repairs bad frames
    using neighboring good frames via optical flow warp.

    Detects scene changes by comparing consecutive source frames.

    Returns:
        alphas: repaired alpha list
        metrics: list of per-frame metrics
        red_count: number of red frames (need user input)
    """
    import cv2

    total = len(alphas)
    metrics = []
    prev_mask = None
    consecutive_yellow = 0
    red_frames = []

    # Phase 1: compute metrics for all frames
    for i in range(total):
        sam2_logit = logits.get(i, None)
        # Only check anchor IoU on first frame (after that, cat has moved)
        anchor_bbox = yolo_bbox if i == 0 else None
        m = compute_frame_metrics(sam2_logit, alphas[i], anchor_bbox, prev_mask, h, w)
        metrics.append(m)
        prev_mask = (alphas[i] > 128).astype(np.uint8)

    # Phase 2: classification only — surface red frames to the UI.
    #
    # Optical-flow warp repair was REMOVED. Even with a correct remap grid it
    # drags a neighbour's mask along the motion vector on fast frames, producing
    # a background "tail-trail" halo + double contours (confirmed on test3
    # frame 74). Warping a single mask can't repair a moving frame cleanly.
    #
    # Real defects are handled downstream by mechanisms that copy a WHOLE good
    # neighbour (RGB+alpha together), so they can never expose background:
    #   - stabilize_alpha_temporal: coverage spikes / empty frames
    #   - repair_opacity_flashes:   per-frame opacity dips (drop-and-hold)
    # Yellow frames are usually just fast motion (low mask-IoU / bbox stability)
    # — i.e. correct BiRefNet output — so they are left untouched.
    red_frames = [i for i in range(total) if metrics[i]["status"] == "red"]

    return alphas, metrics, len(red_frames)


def stabilize_alpha_temporal(alphas, spike_thresh=0.40, neighbor_agree=0.15):
    """Remove single-frame alpha flashes / empty frames (temporal deflicker).

    Targets the whole "flash" class the product suffers from:
      - empty frames (pet alpha drops to ~0 for one frame)
      - opening/closing flashes (anchor frame coverage differs from the rest)
      - floor / background misclassified as foreground for one frame
    All three share the same signature: a frame whose foreground *coverage*
    deviates sharply from BOTH temporal neighbors while the neighbors agree
    with each other. Smooth motion does NOT have this signature — it changes
    gradually frame to frame — so only true one-frame anomalies are flagged.

    Correction is a per-pixel temporal median-of-3 applied ONLY to flagged
    frames. Median is self-guarding: a real dropout (0 vs 255/255) is
    restored to 255, while genuine disagreement yields the middle value.
    Clean frames are returned untouched (identity), so fast motion is never
    ghosted.

    Args:
        alphas: list of uint8 alpha arrays (H, W)
        spike_thresh: min relative coverage deviation from the neighbor mean
            for a frame to count as a flash (0.40 = 40%, well above normal
            motion, catches dropouts/spikes which are 100%+)
        neighbor_agree: max relative coverage difference between the two
            neighbors for them to count as "agreeing" — during real fast
            motion neighbors disagree, which (safely) suppresses correction

    Returns:
        (stabilized_alphas, flagged_indices)
    """
    n = len(alphas)
    if n < 3:
        return alphas, []

    hw = float(alphas[0].size)
    cov = np.array([(a > 128).sum() / hw for a in alphas], dtype=np.float64)

    out = list(alphas)
    flagged = []
    for i in range(n):
        if i == 0:
            na, nb = 1, 2
        elif i == n - 1:
            na, nb = n - 2, n - 3
        else:
            na, nb = i - 1, i + 1

        base = (cov[na] + cov[nb]) / 2.0
        neighbors_agree = abs(cov[na] - cov[nb]) <= neighbor_agree * max(cov[na], cov[nb], 1e-6)
        dev = abs(cov[i] - base) / max(base, 1e-6)

        if neighbors_agree and dev >= spike_thresh:
            flagged.append(i)
            stack = np.stack([alphas[i], alphas[na], alphas[nb]], axis=0)
            out[i] = np.median(stack, axis=0).astype(np.uint8)

    return out, flagged


def repair_opacity_flashes(final_dir, n, emit_fn=None, ratio=0.88, solid=205):
    """Detect & repair per-frame opacity flashes (drop-and-hold).

    A distinct failure from coverage spikes / empty frames: the silhouette
    is normal but a region of the body renders translucent for one frame
    (e.g. test3 frame 74: mean foreground alpha 188 vs neighbours 246),
    because BiRefNet's confidence dips on a motion-blurred / hard frame and
    the new pipeline no longer pins the interior opaque. On the transparent
    desktop this is a visible blink.

    It can't be cleanly reconstructed (the translucent region is exactly
    where the pet is moving, so neither spatial fill nor temporal warp has a
    reliable source). For a 10fps looping pet the robust fix is to replace
    the whole flagged frame (RGB+alpha) with the nearest good neighbour — a
    1-frame hold is imperceptible, a translucency blink is not.

    Detection: a frame whose mean foreground alpha is < `ratio` of the
    smaller neighbour, when both neighbours are solid (> `solid`). Validated
    to flag exactly the real flash on test3 with zero false positives on
    clean clips (test2, dog).
    """
    if n < 3:
        return []
    paths = [os.path.join(final_dir, f"frame_{i:04d}.png") for i in range(n)]
    mf = np.zeros(n, dtype=np.float64)
    for i, p in enumerate(paths):
        bgra = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
            a = bgra[:, :, 3]
            mf[i] = a[a > 10].mean() if (a > 10).any() else 0.0

    flagged = []
    for i in range(n):
        lo, hi = (1, 2) if i == 0 else (n - 2, n - 3) if i == n - 1 else (i - 1, i + 1)
        nb = min(mf[lo], mf[hi])
        if nb > solid and mf[i] < ratio * nb:
            flagged.append(i)

    flagset = set(flagged)
    for i in flagged:
        donor = None
        for off in range(1, n):
            if i - off >= 0 and (i - off) not in flagset:
                donor = i - off
                break
            if i + off < n and (i + off) not in flagset:
                donor = i + off
                break
        if donor is not None:
            shutil.copy(paths[donor], paths[i])

    if emit_fn and flagged:
        emit_fn({"type": "progress", "phase": "opacity_repair",
                 "detail": f"held {len(flagged)} opacity-flash frame(s): {flagged}"})
    return flagged


def make_background(h, w, color=(30, 30, 30)):
    """Generate solid dark background for video preview.

    Uses solid color instead of checkerboard to avoid visual artifacts
    in semi-transparent edge regions.
    """
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = color
    return bg


# === Motion-gated temporal alpha smoothing (Part C #2b — tech/PET_STABILIZATION_PARTC.md) ===
# Why: BiRefNet predicts each frame independently, so the soft alpha band on
# edges shimmers frame-to-frame (~3.3 / 8 mean/p95 px on IMG_0847). The cat's
# body is mostly static across the loop, so most of this shimmer happens on
# pixels that ARE static — perfect target for temporal median.
#
# Critical safety rule (PARTC §3): never smooth a pixel whose SAM2 mask is
# flipping (i.e. the pet is moving there). Unconditional temporal smoothing
# of moving alpha = frozen alpha + moving RGB = background border leak (see
# project_skipframe_rejected). The SAM2 mask is the motion gate; RGB diff
# is useless here (edge pixels have constant high diff).
#
# Window MUST wrap (PARTC §2.2): pet loops ping-pong, so frame 0's neighbours
# include frame N-1. Without wrap, the loop seam has un-smoothed shimmer
# exactly where the eye notices it.
#
# Only smooth pixels in the SOFT band (0 < alpha < 255). Solid interior
# (alpha=255) and full background (alpha=0) are left untouched — they don't
# shimmer anyway, and smoothing them risks dragging interior alpha down.

TEMPORAL_ALPHA_SMOOTH = True
TEMPORAL_WINDOW = 2     # ±2 frames = 5-frame window (1+2*2); swept 1/2/3 in §4


def motion_gated_temporal_smooth(alphas, sam2_binaries,
                                 window=TEMPORAL_WINDOW,
                                 soft_lo=1, soft_hi=255,
                                 progress_callback=None):
    """Per-pixel motion-gated temporal median of soft-edge alpha.

    Args:
        alphas: list of N uint8 H×W alpha arrays (the `alpha_final` pre-FB-fusion)
        sam2_binaries: list of N uint8 H×W {0,1} arrays (the pre-dilate SAM2 mask)
        window: half-width of the temporal window (so full window is 2*window+1)
        soft_lo, soft_hi: alpha band eligible for smoothing. Defaults 1..254
            (anything strictly between full-transparent and full-opaque).
        progress_callback(phase, current, total, detail) optional.

    Returns:
        list of N uint8 H×W smoothed alphas. Frames where motion was detected
        in the local window at that pixel are passed through unchanged.

    Implementation note: vectorised via numpy stride tricks for speed.
    The wrap trick is `np.roll(stack, k, axis=0)` to address out-of-bounds
    frames cheaply (no copies needed for reads).
    """
    n = len(alphas)
    if n < 3 or window < 1:
        return alphas

    H, W = alphas[0].shape
    if any(a.shape != (H, W) for a in alphas) or any(s.shape != (H, W) for s in sam2_binaries):
        return alphas  # shape mismatch → no-op (don't crash the pipeline)

    # Stack into 3D arrays for vectorised indexing. (n, H, W)
    a_stack = np.stack(alphas, axis=0).astype(np.uint8)
    s_stack = np.stack(sam2_binaries, axis=0).astype(np.uint8)

    # 1) motion mask: per-pixel "did sam2_binary flip anywhere in the local window"
    # s_stack is {0,1}; rolling-window max == 1 if any sample is 1, min == 0 if any is 0.
    # Flipped iff max != min.
    flipped = np.zeros((n, H, W), dtype=bool)
    for k in range(-window, window + 1):
        if k == 0:
            continue
        rolled = np.roll(s_stack, k, axis=0)
        flipped |= (rolled != s_stack)

    # 2) soft-band mask: 0 < alpha < 255 (per-pixel, per-frame)
    soft = (a_stack > soft_lo) & (a_stack < soft_hi)

    # 3) eligible-for-smooth mask: soft AND not-flipped
    eligible = soft & ~flipped

    # 4) per-pixel median across the window. Build a (n, W*H, k_len) tensor and
    #    take median along last axis. Memory: n*W*H*k_len uint8. For 72 frames
    #    at 1920×1080 with window=2: 72*2M*5 ≈ 720MB uint8 ≈ 720MB. Too big.
    #    Chunk along the H axis instead.
    out = a_stack.copy()  # default: unchanged
    chunk_rows = 64
    n_chunks = (H + chunk_rows - 1) // chunk_rows
    for ci in range(n_chunks):
        y0 = ci * chunk_rows
        y1 = min(y0 + chunk_rows, H)
        # slice on this chunk
        a_chunk = a_stack[:, y0:y1, :].astype(np.float32)   # (n, h, W)
        e_chunk = eligible[:, y0:y1, :]                     # (n, h, W)
        # build window stack (k_len, n, h, W) → reshape to (n*h*W, k_len) for median
        k_len = 2 * window + 1
        win_stack = np.empty((k_len, n, y1 - y0, W), dtype=np.float32)
        for ki, k in enumerate(range(-window, window + 1)):
            win_stack[ki] = np.roll(a_chunk, k, axis=0)
        # median across k_len axis
        med = np.median(win_stack, axis=0)  # (n, h, W)
        # write back ONLY at eligible pixels (others keep original)
        result_chunk = np.where(e_chunk, med, a_chunk).astype(np.uint8)
        out[:, y0:y1, :] = result_chunk
        if progress_callback and (ci % 8 == 0 or ci == n_chunks - 1):
            progress_callback("temporal_smooth", ci + 1, n_chunks,
                              f"chunk {ci+1}/{n_chunks}")

    return [out[i] for i in range(n)]


# === Output position stabilization (Part B #1 — tech/PET_STABILIZATION_PARTB.md) ===
# Why: even after Part A (camera shake), the BiRefNet+SAM2 alpha matte itself
# jitters — the cutout's centroid jumps ~3.3px frame-to-frame and the contour
# shimmers 3.3/8 (mean/p95) px. Evaluator confirmed the source is mostly stable
# (IMG_0847 raw = 1.9px) so the residual is the matting pipeline's own variance.
#
# This is the SAFE fix: rigid-translate the whole RGBA cutout so its centroid
# tracks a smoothed centroid trajectory. RGB and alpha are warped together —
# no edge mixing, no halos, no "frozen alpha + moving RGB" failure mode (see
# project_skipframe_rejected). Smoothed (not zeroed) so low-freq real cat
# motion survives; only the high-freq jitter is removed.
#
# Border MUST be (0,0,0,0) — fully transparent. Never black, never replicated
# (a transparent border is what lets display-side union-bbox crop cleanly
# without showing a frame-of-background strip).

# PARTC §1 (2026-06-20): DISABLED. Part B #1 was a category error — rigid
# output translation fixed a centroid noise problem by amplifying it into a
# whole-cat translation, which the eye reads as worse jitter than the original
# edge shimmer. Code preserved so PARTC §2's different approach can use the
# same scaffolding if needed; switch stays False until/unless PARTC revives it.
STABILIZE_OUTPUT_POS = False
STABILIZE_OUTPUT_RADIUS = 8    # ~0.8s @ 10fps; swept 5/8/12 in §6
STABILIZE_OUTPUT_THRESH_PX = 0.3   # skip micro-shifts (no-op cost)
STABILIZE_OUTPUT_CLAMP_PX = 20     # safety clamp so a bad frame doesn't yank cutout 100s px


def stabilize_output_position(final_dir, n, smooth_radius=STABILIZE_OUTPUT_RADIUS,
                              area_thresh=128, progress_callback=None):
    """Rigid-translate each RGBA frame so the alpha centroid tracks a smoothed
    trajectory. Mutates `final_dir/frame_*.png` in place.

    Returns:
        (did_stabilize, report_dict)  — report has before/after centroid jitter.

    Skips (no-op, did=False) when inter-frame centroid diff std is already below
    `STABILIZE_OUTPUT_THRESH_PX` — saving ms on stable clips.
    """
    import time as _t
    report = {
        "before_centroid_std_x_px": 0.0,
        "after_centroid_std_x_px": 0.0,
        "before_centroid_std_y_px": 0.0,
        "after_centroid_std_y_px": 0.0,
        "reduction_pct_x": 0.0,
        "reduction_pct_y": 0.0,
        "smooth_radius": int(smooth_radius),
        "n": int(n),
        "n_valid_frames": 0,
        "frames_warped": 0,
        "clamped_frames": 0,
        "skipped": False,
        "skip_reason": None,
        "dt_s": 0.0,
    }
    if not STABILIZE_OUTPUT_POS or n < 3:
        report["skipped"] = True
        report["skip_reason"] = "disabled" if not STABILIZE_OUTPUT_POS else "too_few_frames"
        return False, report

    paths = [os.path.join(final_dir, f"frame_{i:04d}.png") for i in range(n)]
    # 1) read RGBA + measure raw centroid trajectory
    rgba = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]
    if any(s is None or s.ndim != 3 or s.shape[2] != 4 for s in rgba):
        report["skipped"] = True
        report["skip_reason"] = "bad_rgba"
        return False, report

    h, w = rgba[0].shape[:2]
    cx = np.zeros(n, dtype=np.float64)
    cy = np.zeros(n, dtype=np.float64)
    # Track which frames have non-empty alpha — empty-alpha frames get a
    # NaN centroid so they're excluded from smoothing (otherwise a single
    # empty frame would yank the entire smoothed trajectory to 0,0 and
    # clamp every other frame by 20px in the wrong direction).
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(rgba):
        ys, xs = np.where(s[:, :, 3] > area_thresh)
        if xs.size >= 16:  # ignore tiny noise blobs (< 16 px isn't a pet)
            cx[i], cy[i] = xs.mean(), ys.mean()
            valid[i] = True
        else:
            cx[i], cy[i] = np.nan, np.nan

    n_valid = int(valid.sum())
    if n_valid < 3:
        report["skipped"] = True
        report["skip_reason"] = f"too_few_valid_frames({n_valid}<3)"
        return False, report

    # baseline: inter-frame diff std (the "high-frequency jitter" per §5)
    # Use NaN-aware diff so empty frames don't dominate the std.
    diff_cx = np.diff(cx)
    diff_cy = np.diff(cy)
    valid_diff = ~(np.isnan(diff_cx) | np.isnan(diff_cy))
    before_x = float(diff_cx[valid_diff].std()) if valid_diff.any() else 0.0
    before_y = float(diff_cy[valid_diff].std()) if valid_diff.any() else 0.0
    report["before_centroid_std_x_px"] = before_x
    report["before_centroid_std_y_px"] = before_y
    report["n_valid_frames"] = n_valid

    if before_x < STABILIZE_OUTPUT_THRESH_PX and before_y < STABILIZE_OUTPUT_THRESH_PX:
        report["skipped"] = True
        report["skip_reason"] = (f"already_stable(x={before_x:.2f},y={before_y:.2f}"
                                 f"<{STABILIZE_OUTPUT_THRESH_PX:.2f}px)")
        return False, report

    # 2) smooth the centroid trajectory with a proper moving average.
    # Naive `np.convolve(x, k, mode='same')` is wrong here: it returns the
    # centre slice of the full convolution (length max(M,N)), AND the boundary
    # values are biased toward the signal centre (only half the kernel sees
    # real data at the edges). Found this in the Part B #1 verification —
    # centroids actually got WORSE after #1 because the smoothed trajectory
    # was offset by hundreds of pixels from the raw trajectory.
    #
    # Correct: cumsum-based MA with reflection at boundaries so each output
    # sample averages exactly `k_len` values (clipped + mirrored at edges).
    # This is the same pattern as the trajectory smoother in Part A, but
    # applied per-axis on the 1D centroid signal.
    r = int(smooth_radius)
    k_len = 2 * r + 1

    def moving_avg_reflect(sig, r):
        """Moving average with reflective boundary handling, NaN-safe.

        At edges the window is clipped (not mirrored — for centroid data that
        would mean averaging the same point twice, which biases the centre).
        Clipped windows are divided by ACTUAL count, not kernel length.

        NaN values are skipped — a frame with no alpha contributes 0 to the
        window rather than dragging the mean down.
        """
        n = len(sig)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            a = max(0, i - r)
            b = min(n - 1, i + r)
            win = sig[a:b + 1]
            # NaN-safe: skip invalid samples
            ok = ~np.isnan(win)
            if not ok.any():
                out[i] = np.nan
            else:
                out[i] = float(win[ok].sum() / ok.sum())
        return out

    sm_x = moving_avg_reflect(cx, r)
    sm_y = moving_avg_reflect(cy, r)

    # 3) per-frame correction (clamped + tiny-shift skip + skip empty-alpha frames)
    t0 = _t.time()
    wrote = 0
    clamped_count = 0
    for i, s in enumerate(rgba):
        if not valid[i]:
            # Skip — alpha was empty, no centroid to align to. Leaving the
            # frame untouched avoids yanking it to (0,0).
            continue
        if np.isnan(sm_x[i]) or np.isnan(sm_y[i]):
            continue
        dx_raw = sm_x[i] - cx[i]
        dy_raw = sm_y[i] - cy[i]
        dx = float(np.clip(dx_raw, -STABILIZE_OUTPUT_CLAMP_PX, STABILIZE_OUTPUT_CLAMP_PX))
        dy = float(np.clip(dy_raw, -STABILIZE_OUTPUT_CLAMP_PX, STABILIZE_OUTPUT_CLAMP_PX))
        if abs(dx_raw) > STABILIZE_OUTPUT_CLAMP_PX or abs(dy_raw) > STABILIZE_OUTPUT_CLAMP_PX:
            clamped_count += 1
        if abs(dx) < STABILIZE_OUTPUT_THRESH_PX and abs(dy) < STABILIZE_OUTPUT_THRESH_PX:
            continue
        M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        # TRANSPARENT border (0,0,0,0). NEVER black — display would show a
        # black strip at the cutout edge.
        warped = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        cv2.imwrite(paths[i], warped)
        wrote += 1
    report["dt_s"] = round(_t.time() - t0, 3)
    report["frames_warped"] = wrote
    report["clamped_frames"] = clamped_count

    # 4) measure post centroid jitter from newly-written frames
    cx2 = np.full(n, np.nan, dtype=np.float64)
    cy2 = np.full(n, np.nan, dtype=np.float64)
    for i, p in enumerate(paths):
        s = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        ys, xs = np.where(s[:, :, 3] > area_thresh)
        if xs.size >= 16:
            cx2[i], cy2[i] = xs.mean(), ys.mean()
    diff_cx2 = np.diff(cx2)
    diff_cy2 = np.diff(cy2)
    valid_diff2 = ~(np.isnan(diff_cx2) | np.isnan(diff_cy2))
    after_x = float(diff_cx2[valid_diff2].std()) if valid_diff2.any() else 0.0
    after_y = float(diff_cy2[valid_diff2].std()) if valid_diff2.any() else 0.0
    report["after_centroid_std_x_px"] = after_x
    report["after_centroid_std_y_px"] = after_y
    report["reduction_pct_x"] = max(0.0, (before_x - after_x) / max(before_x, 1e-6)) * 100.0
    report["reduction_pct_y"] = max(0.0, (before_y - after_y) / max(before_y, 1e-6)) * 100.0

    if progress_callback:
        progress_callback("stabilize_output",
                          n, n,
                          f"centroid std x {before_x:.2f}→{after_x:.2f}px "
                          f"(-{report['reduction_pct_x']:.1f}%)  "
                          f"y {before_y:.2f}→{after_y:.2f}px "
                          f"(-{report['reduction_pct_y']:.1f}%)  "
                          f"warped {wrote}/{n} in {report['dt_s']}s")
    return True, report


def make_video(frame_dir, output_path, fps=10):
    """Generate MP4 video from BGRA PNGs on solid dark background."""
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frames:
        return

    h, w = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED).shape[:2]
    bg = make_background(h, w)

    # Per-output intermediate dir (preview vs final must not overwrite each
    # other) using a lossless PNG intermediate — JPEG q92 double-compression
    # was a needless source of the "PNG clean but MP4 dirty" artifacts.
    rgb_dir = os.path.join(os.path.dirname(frame_dir),
                           "rgb_" + os.path.splitext(os.path.basename(output_path))[0])
    os.makedirs(rgb_dir, exist_ok=True)
    for old in glob.glob(os.path.join(rgb_dir, "*.png")):
        os.remove(old)

    for f in frames:
        bgra = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        a = bgra[:, :, 3].astype(float) / 255
        disp = (bg * (1 - a[:, :, None]) + bgr * a[:, :, None]).astype(np.uint8)
        base = os.path.basename(f)  # keep .png (lossless)
        cv2.imwrite(f"{rgb_dir}/{base}", disp)

    # scenecut=0 + fixed keyint stops x264 from inserting mid-clip keyframes
    # at motion bursts (the 7s "flash"); -bf 0 removes B-frame quality
    # oscillation. These keep brightness/quality constant across the clip.
    cmd = [
        "ffmpeg", "-y", "-f", "image2", "-framerate", str(fps),
        "-i", f"{rgb_dir}/frame_%04d.png",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-x264-params", "scenecut=0:keyint=600:min-keyint=600",
        "-bf", "0",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)


def find_pet_center(frames):
    """Auto-detect pet center using torchvision Faster R-CNN.

    Raises RuntimeError if no pet is detected (caller should prompt user
    for --click instead of silently using frame center).
    """
    from pipeline.pet_detector import detect_pet_centroid

    result = detect_pet_centroid(frames[0])
    if result is None:
        raise RuntimeError("No pet detected in first frame. Please use --click to specify pet location.")
    cx, cy, name, conf = result
    return (cx, cy, name, conf)


def check_dependencies():
    """Verify all required packages are available."""
    missing = []
    try:
        import torch
        if not torch.backends.mps.is_available():
            missing.append("MPS (Apple Silicon GPU) not available")
    except ImportError:
        missing.append("torch")

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        missing.append("sam2 (run: pip install git+https://github.com/facebookresearch/sam2.git)")

    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        missing.append("transformers")

    try:
        import torchvision
    except ImportError:
        missing.append("torchvision (run: pip install torchvision)")

    checkpoint = _sam2_checkpoint()
    if not os.path.exists(checkpoint):
        missing.append(f"SAM2 weights: {checkpoint} (run: python scripts/download_weights.py)")

    return missing


def main():
    # Pre-flight dependency check
    missing = check_dependencies()
    if missing:
        emit({"type": "error", "message": f"Missing dependencies: {', '.join(missing)}"})
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Track-then-Matte pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Frame extraction FPS")
    parser.add_argument("--click", type=str, default=None,
                        help="Pet click point as 'x,y' (auto-detect if not set)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction (reuse existing)")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip quality check")
    parser.add_argument("--preview-seconds", type=float, default=5.0,
                        help="Seconds of video to process for preview (default: 5)")
    parser.add_argument("--max-seconds", type=float, default=15.0,
                        help="Max seconds to process (longer videos are trimmed, default: 15)")
    parser.add_argument("--start", type=float, default=None,
                        help="Explicit clip start time (s); skips smart auto-select")
    parser.add_argument("--duration", type=float, default=None,
                        help="Explicit clip duration (s), used with --start")
    parser.add_argument("--preview-only", action="store_true",
                        help="Only run SAM2 pass (preview)")
    args = parser.parse_args()

    output_dir = args.output_dir
    extract_dir = os.path.join(output_dir, "extracted")
    preview_dir = os.path.join(output_dir, "preview")
    final_dir = os.path.join(output_dir, "segmented")

    # Step 1: Check duration and decide processing range
    video_duration = _get_video_duration(args.video)

    if args.start is not None and not args.skip_extract:
        # User picked a specific segment — honor it exactly, no auto-select.
        dur = args.duration if args.duration is not None else args.max_seconds
        # Honor the user's range, clamped to [8s, max_seconds]. The dual-handle
        # clip picker already enforces this, but mirror it here so any caller
        # (or a future short selection) can't slip a sub-8s or over-budget clip.
        dur = max(8.0, min(dur, args.max_seconds))
        start = max(0.0, min(args.start, max(0.0, video_duration - 1.0)))
        emit({"type": "phase", "name": "extract",
              "detail": f"extracting {dur:.1f}s from {start:.1f}s (user-selected)"})
        trim_dir = os.path.join(extract_dir, "trim")
        frames = extract_frames_range(args.video, trim_dir, start, dur, args.fps)
        emit({"type": "progress", "phase": "extract",
              "current": len(frames), "total": len(frames), "detail": "done"})
    elif video_duration <= args.max_seconds:
        # Short video: process entire video
        if not args.skip_extract:
            emit({"type": "phase", "name": "extract",
                  "detail": f"extracting {video_duration:.1f}s video"})
            frames = extract_frames(args.video, extract_dir, fps=args.fps)
            emit({"type": "progress", "phase": "extract",
                  "current": len(frames), "total": len(frames), "detail": "done"})
        else:
            frames = sorted(glob.glob(os.path.join(extract_dir, "frame_*.jpg")))
    else:
        # Long video: auto-select best segment
        emit({"type": "phase", "name": "analyze",
              "detail": f"video is {video_duration:.1f}s, selecting best {args.max_seconds}s segment"})
        try:
            from pipeline.smart_clip import analyze_video
            result = analyze_video(args.video, target_duration=args.max_seconds)
            clip = result.recommended_clip
            trim_dir = os.path.join(extract_dir, "trim")
            os.makedirs(trim_dir, exist_ok=True)
            frames = extract_frames_range(args.video, trim_dir,
                                          clip.start_time, clip.duration, args.fps)
            emit({"type": "progress", "phase": "analyze",
                  "detail": f"selected {clip.start_time:.1f}-{clip.end_time:.1f}s "
                           f"({len(frames)} frames, score={clip.avg_score:.2f})"})
        except Exception as e:
            emit({"type": "progress", "phase": "analyze",
                  "detail": f"auto-select failed: {e}, keeping first {args.max_seconds}s"})
            frames = extract_frames(args.video, extract_dir, fps=args.fps)
            frames = frames[:int(args.max_seconds * args.fps)]

    if not frames:
        emit({"type": "error", "message": "No frames extracted"})
        sys.exit(1)

    # Step 1.5: Video stabilization (Part A — tech/PET_STABILIZATION_SPIKE.md).
    # MUST run before check_quality / SAM2 / BiRefNet so tracking & matting see
    # stable frames. Mutates `frames` in place. Skips if STABILIZE=False or if
    # the source is already below STABILIZE_SHAKE_THRESH px residual.
    if STABILIZE:
        emit({"type": "phase", "name": "stabilize",
              "detail": f"shake gate {STABILIZE_SHAKE_THRESH}px, "
                        f"smooth_r={STABILIZE_SMOOTH_RADIUS}, zoom={STABILIZE_ZOOM}"})
        try:
            def _stab_prog(phase, current, total, detail):
                emit({"type": "progress", "phase": phase,
                      "current": current, "total": total, "detail": detail})
            frames, did_stab, stab_report = stabilize_frames(
                frames,
                smooth_radius=STABILIZE_SMOOTH_RADIUS,
                zoom=STABILIZE_ZOOM,
                shake_thresh=STABILIZE_SHAKE_THRESH,
                progress_callback=_stab_prog,
            )
            if stab_report["skipped"]:
                emit({"type": "progress", "phase": "stabilize",
                      "detail": f"skipped: {stab_report['skip_reason']}"})
            else:
                emit({"type": "stabilize_done", "report": stab_report})
        except Exception as e:
            # Stabilization is best-effort — never block the pipeline.
            emit({"type": "progress", "phase": "stabilize",
                  "detail": f"failed ({type(e).__name__}: {e}), continuing with raw frames"})

    # Step 2: Quality check
    if not args.skip_qa:
        emit({"type": "phase", "name": "quality_check", "detail": "checking quality"})
        qa = check_quality(args.video, frames, max_duration=1000)
        if not qa.passed:
            emit({"type": "quality_failed", "result": qa.to_dict()})
            sys.exit(2)
        emit({"type": "progress", "phase": "quality_check", "detail": "passed"})

    # Step 3: Determine click point
    if args.click:
        cx, cy = map(int, args.click.split(","))
        click_point = (cx, cy)
        emit({"type": "progress", "phase": "detect",
              "detail": f"pet at ({cx}, {cy})"})
    else:
        emit({"type": "phase", "name": "detect", "detail": "auto-detecting pet"})
        cx, cy, name, conf = find_pet_center(frames)
        click_point = (cx, cy)
        emit({"type": "progress", "phase": "detect",
              "detail": f"{name} ({conf:.0%}) at ({cx}, {cy})"})

    # Step 4: SAM2 tracking (single pass, collect logits)
    preview_limit = int(args.preview_seconds * args.fps)

    def sam2_progress(phase, current, total, detail):
        emit({"type": "progress", "phase": phase,
              "current": current, "total": total, "detail": detail})

    all_logits = pass1_sam2(frames, click_point,
                            progress_callback=sam2_progress)

    # Step 5: Pass 2 - BiRefNet refinement (final). The first N matted frames
    # double as the preview: same model as the final output, so what the user
    # sees early is byte-identical to what lands later (no model mismatch).
    preview_video = os.path.join(output_dir, "preview.mp4")

    def on_preview_ready(n):
        # Snapshot the first n final frames into preview_dir so the preview is
        # stable while pass 2 keeps writing the rest into final_dir.
        os.makedirs(preview_dir, exist_ok=True)
        for i in range(n):
            shutil.copy2(os.path.join(final_dir, f"frame_{i:04d}.png"),
                         os.path.join(preview_dir, f"frame_{i:04d}.png"))
        make_video(preview_dir, preview_video, fps=args.fps)
        emit({"type": "preview_ready",
              "preview_dir": os.path.abspath(preview_dir),
              "preview_video": os.path.abspath(preview_video),
              "frames": n})

    def birefnet_progress(phase, current, total, detail):
        emit({"type": "progress", "phase": phase,
              "current": current, "total": total, "detail": detail})

    if args.preview_only:
        # Only matte the first N frames, emit preview, stop before final.
        pass2_birefnet(frames[:preview_limit], all_logits, final_dir,
                       birefnet_progress, preview_limit=preview_limit,
                       on_preview_ready=on_preview_ready)
        emit({"type": "complete", "preview": preview_video,
              "frames": min(preview_limit, len(frames))})
        return

    pass2_birefnet(frames, all_logits, final_dir, birefnet_progress,
                   preview_limit=preview_limit,
                   on_preview_ready=on_preview_ready)

    # Step 6: Quality feedback loop
    emit({"type": "phase", "name": "quality_feedback", "detail": "checking output quality"})
    h, w = cv2.imread(frames[0]).shape[:2]

    # Load alphas from segmented output
    alphas = []
    for i in range(len(frames)):
        seg_path = os.path.join(final_dir, f"frame_{i:04d}.png")
        bgra = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if bgra is not None and len(bgra.shape) == 3 and bgra.shape[2] == 4:
            alphas.append(bgra[:, :, 3])
        else:
            alphas.append(np.zeros((h, w), dtype=np.uint8))

    # Get pet bbox for anchor validation
    yolo_bbox_for_qa = None
    try:
        from pipeline.pet_detector import detect_pet_bbox
        yolo_bbox_for_qa = detect_pet_bbox(frames[0])
    except Exception:
        pass

    # Run quality feedback
    alphas, metrics, red_count = run_quality_feedback(
        alphas, all_logits, yolo_bbox_for_qa, h, w,
        frame_paths=frames, emit_fn=emit
    )

    # Final temporal deflicker: surgically remove single-frame alpha
    # flashes / empty frames that survive (or are introduced by) the
    # steps above. Only true one-frame anomalies are touched; motion is
    # left intact. This is what the desktop pet actually renders, so it
    # is the layer that matters for the user-visible flicker.
    alphas, flash_frames = stabilize_alpha_temporal(alphas)
    if flash_frames:
        emit({"type": "progress", "phase": "deflicker",
              "detail": f"stabilized {len(flash_frames)} flash frame(s): {flash_frames}"})

    # Re-save repaired alphas
    for i in range(len(alphas)):
        seg_path = os.path.join(final_dir, f"frame_{i:04d}.png")
        bgra = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if bgra is not None and len(bgra.shape) == 3 and bgra.shape[2] == 4:
            bgra[:, :, 3] = alphas[i]
            cv2.imwrite(seg_path, bgra)

    # Repair per-frame opacity flashes (body renders translucent for one
    # frame — different failure from coverage spikes, so the deflicker above
    # is blind to it). Operates on the saved RGBA files: holds the nearest
    # good neighbour over each flagged frame.
    opacity_flashes = repair_opacity_flashes(final_dir, len(alphas), emit_fn=emit)

    # Part B #1 (tech/PET_STABILIZATION_PARTB.md §2): rigid-translate each RGBA
    # cutout so the alpha centroid tracks a smoothed trajectory. Kills the
    # "whole-cat position bouncing ~3px" residue left after Part A. Safe — RGB
    # and alpha move together; border is transparent; low-freq real motion
    # survives (only high-freq jitter is removed).
    if STABILIZE_OUTPUT_POS:
        emit({"type": "phase", "name": "stabilize_output",
              "detail": f"rigid-translate cutouts (smooth_r={STABILIZE_OUTPUT_RADIUS},"
                        f" threshold={STABILIZE_OUTPUT_THRESH_PX}px)"})
        try:
            def _stab_out_prog(phase, current, total, detail):
                emit({"type": "progress", "phase": phase,
                      "current": current, "total": total, "detail": detail})
            # stabilize_output_position returns (did: bool, report: dict) — 2-tuple.
            did_out, out_report = stabilize_output_position(
                final_dir, len(alphas),
                smooth_radius=STABILIZE_OUTPUT_RADIUS,
                progress_callback=_stab_out_prog,
            )
            if out_report["skipped"]:
                emit({"type": "progress", "phase": "stabilize_output",
                      "detail": f"skipped: {out_report['skip_reason']}"})
            else:
                emit({"type": "stabilize_output_done", "report": out_report})
        except Exception as e:
            emit({"type": "progress", "phase": "stabilize_output",
                  "detail": f"failed ({type(e).__name__}: {e}), continuing"})

    # Re-collect alphas if #1 moved the cutouts (centroid shifted) — downstream
    # quality_summary / make_video read from final_dir/frame_*.png so they're
    # already seeing the moved frames; only metrics that use the in-memory
    # `alphas` list would be stale (we don't re-use `alphas` past this point,
    # so no-op).

    # Emit quality summary
    green = sum(1 for m in metrics if m["status"] == "green")
    yellow = sum(1 for m in metrics if m["status"] == "yellow")
    red = sum(1 for m in metrics if m["status"] == "red")
    emit({"type": "quality_summary",
          "green": green, "yellow": yellow, "red": red,
          "total": len(metrics),
          "detail": f"green={green}, yellow={yellow}, red={red}"})

    if red > 0:
        emit({"type": "quality_alert",
              "level": "red",
              "message": f"{red} frames have severe quality issues. Consider re-selecting the pet."})

    # Post-processing gate: if too many red frames, the pet's edges are too
    # complex for the segmentation models. This is the ONLY reliable way to
    # detect "edge too cluttered" — pre-QC cannot cheaply predict SAM2/BiRefNet
    # failure. See tech/QC_QUALITY_GATE_PLAN.md §0 "关键诚实点".
    total_frames = len(metrics)
    if total_frames > 0:
        red_ratio = red / total_frames
        if red_ratio > MAX_RED_RATIO:
            emit({"type": "segmentation_poor",
                  "red_count": red, "total": total_frames,
                  "message": f"分割质量不达标（{red}/{total_frames} 帧异常），"
                             f"素材边缘可能过于复杂或主体不清晰，建议换一段画面更干净的素材"})
            # Hard gate: do NOT generate the final video or emit "complete".
            # Emitting complete here would overwrite the failure on the Swift
            # side and drop a bad pet onto the desktop (self-contradictory).
            return

    # Generate final video
    emit({"type": "phase", "name": "final_video", "detail": "generating final"})
    final_video = os.path.join(output_dir, "final.mp4")
    make_video(final_dir, final_video, fps=args.fps)
    emit({"type": "progress", "phase": "final_video",
          "detail": f"saved to {final_video}"})

    emit({"type": "complete",
          "preview": preview_video,
          "final": final_video,
          "frames": len(frames),
          "frames_dir": os.path.abspath(final_dir),
          "segmented_dir": os.path.abspath(final_dir),
          "frame_count": len(frames),
          "quality": {"green": green, "yellow": yellow, "red": red}})


if __name__ == "__main__":
    main()
