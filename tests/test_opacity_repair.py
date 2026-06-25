"""Unit test for repair_opacity_flashes (per-frame opacity-flash drop-and-hold).

Builds a synthetic RGBA sequence of an opaque blob, drops ONE frame's
interior to translucent (the flash signature), and verifies the repair
holds the nearest good neighbour over it — while leaving clean clips
untouched.

Run:  python tests/test_opacity_repair.py
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.track_then_matte import repair_opacity_flashes

H, W = 80, 80


def write_clip(d, alphas):
    os.makedirs(d, exist_ok=True)
    for i, a in enumerate(alphas):
        bgra = np.zeros((H, W, 4), np.uint8)
        bgra[:, :, :3] = 120          # dummy RGB
        bgra[:, :, 3] = a
        cv2.imwrite(os.path.join(d, f"frame_{i:04d}.png"), bgra)


def mean_fg(d, i):
    a = cv2.imread(os.path.join(d, f"frame_{i:04d}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3]
    return float(a[a > 10].mean()) if (a > 10).any() else 0.0


def solid_blob(val=255):
    a = np.zeros((H, W), np.uint8)
    a[20:60, 20:60] = val
    return a


def main():
    tmp = tempfile.mkdtemp(prefix="opacity_test_")
    try:
        # 8 opaque frames; frame 4's interior drops to translucent (flash)
        alphas = [solid_blob(255) for _ in range(8)]
        flash = solid_blob(255)
        flash[20:60, 20:60] = 150       # whole blob translucent
        alphas[4] = flash

        clip = os.path.join(tmp, "seg")
        write_clip(clip, alphas)

        before = mean_fg(clip, 4)
        flagged = repair_opacity_flashes(clip, 8)
        after = mean_fg(clip, 4)

        assert flagged == [4], f"expected only frame 4 flagged, got {flagged}"
        assert after > before + 50, f"frame 4 not repaired: {before} -> {after}"
        assert abs(after - mean_fg(clip, 3)) < 1, "repaired frame should match neighbour"

        # Clean clip: nothing flagged
        clean = os.path.join(tmp, "clean")
        write_clip(clean, [solid_blob(255) for _ in range(8)])
        assert repair_opacity_flashes(clean, 8) == [], "clean clip falsely flagged"

        print(f"PASS  flagged={flagged}  frame4 meanFG {before:.0f} -> {after:.0f}  "
              f"(neighbour {mean_fg(clip, 3):.0f}); clean clip: no flags")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
