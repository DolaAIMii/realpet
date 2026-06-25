"""Unit test for stabilize_alpha_temporal (temporal alpha deflicker).

Verifies the deflicker:
  1. flags + repairs an empty frame (alpha drops to 0)
  2. flags + repairs an opening flash (coverage spikes up, e.g. floor
     misclassified as foreground)
  3. leaves smooth-motion frames byte-for-byte untouched (no ghosting)

Run:  python tests/test_deflicker.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.track_then_matte import stabilize_alpha_temporal

H, W = 64, 64


def disc(cx, r=15):
    yy, xx = np.ogrid[:H, :W]
    m = (xx - cx) ** 2 + (yy - 32) ** 2 <= r * r
    a = np.zeros((H, W), np.uint8)
    a[m] = 255
    return a


def cov(a):
    return int((a > 128).sum())


def main():
    # 10 frames: a disc drifting right by 1px/frame = smooth motion
    clean = [disc(20 + i) for i in range(10)]
    seq = [a.copy() for a in clean]

    # Inject anomalies
    seq[0] = np.full((H, W), 255, np.uint8)   # opening flash: full-frame fg spike
    seq[5] = np.zeros((H, W), np.uint8)        # empty frame: total dropout

    out, flagged = stabilize_alpha_temporal(seq)

    # 1+2. Both anomalies flagged
    assert 0 in flagged, f"opening flash not flagged: {flagged}"
    assert 5 in flagged, f"empty frame not flagged: {flagged}"

    # Empty frame restored to roughly its true coverage
    assert cov(out[5]) > 0.5 * cov(clean[5]), (
        f"empty frame not restored: got {cov(out[5])}, expected ~{cov(clean[5])}")

    # Opening flash pulled back down to disc-scale coverage (not full frame)
    assert cov(out[0]) < 2 * cov(clean[1]), (
        f"opening flash not suppressed: got {cov(out[0])}, full={H * W}")

    # 3. Smooth-motion frames untouched (identity — no ghosting)
    for i in (2, 3, 4, 6, 7, 8):
        assert np.array_equal(out[i], seq[i]), f"motion frame {i} was modified"

    # Sanity: nothing flagged in a fully clean sequence
    _, flagged_clean = stabilize_alpha_temporal(clean)
    assert flagged_clean == [], f"clean sequence falsely flagged: {flagged_clean}"

    print("PASS  flagged =", flagged,
          "| empty restored:", cov(clean[5]), "->", cov(out[5]),
          "| flash suppressed:", H * W, "->", cov(out[0]),
          "| clean-seq flags:", flagged_clean)


if __name__ == "__main__":
    main()
