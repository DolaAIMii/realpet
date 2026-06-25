#!/usr/bin/env python3
"""Unified weight downloader for realpet.

Downloads the SAM2 checkpoint to the project's weights/ directory.
BiRefNet and Faster R-CNN are auto-downloaded by their respective libraries
(HuggingFace from_pretrained and torchvision), so only SAM2 needs manual fetch.

Usage:
    python scripts/download_weights.py           # download all
    python scripts/download_weights.py --check   # only verify, don't download
"""
import argparse
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configurable via REALPET_WEIGHTS_DIR env var
WEIGHTS_DIR = os.environ.get(
    "REALPET_WEIGHTS_DIR", os.path.join(PROJECT_ROOT, "weights"))

SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
SAM2_SUBDIR = "sam2"
SAM2_FILENAME = "sam2.1_hiera_tiny.pt"
SAM2_EXPECTED_SIZE = 156_000_000  # ~156MB, used for rough validation


def _sam2_path():
    return os.path.join(WEIGHTS_DIR, SAM2_SUBDIR, SAM2_FILENAME)


def download_sam2(force=False):
    """Download SAM2 checkpoint. Returns True if already present or downloaded."""
    dest = _sam2_path()
    if os.path.exists(dest) and not force:
        size = os.path.getsize(dest)
        if size > SAM2_EXPECTED_SIZE * 0.9:
            print(f"SAM2 checkpoint already exists: {dest} ({size:,} bytes)")
            return True
        print(f"Existing file too small ({size:,} bytes), re-downloading...")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading SAM2 checkpoint to {dest} ...")
    print(f"  URL: {SAM2_URL}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(SAM2_URL, dest, reporthook=_progress)
        print()
        size = os.path.getsize(dest)
        print(f"Done. ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("You can manually download from:")
        print(f"  {SAM2_URL}")
        print(f"  and place it at: {dest}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def check_all():
    """Check which weights are present. Returns list of missing items."""
    missing = []

    # SAM2
    sam2 = _sam2_path()
    if not os.path.exists(sam2):
        missing.append(f"SAM2 checkpoint: {sam2}")
    else:
        size = os.path.getsize(sam2)
        print(f"✓ SAM2 checkpoint: {sam2} ({size:,} bytes)")

    # BiRefNet (auto-downloaded by HuggingFace)
    print("✓ BiRefNet-matting: auto-downloaded by HuggingFace from_pretrained()")
    print("  (first run requires internet; cached at ~/.cache/huggingface/)")

    # Faster R-CNN (auto-downloaded by torchvision)
    print("✓ Faster R-CNN: auto-downloaded by torchvision on first use")
    print("  (cached at ~/.cache/torch/hub/)")

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for realpet")
    parser.add_argument("--check", action="store_true",
                        help="Only check which weights exist, don't download")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if file exists")
    args = parser.parse_args()

    if args.check:
        missing = check_all()
        if missing:
            print(f"\nMissing: {', '.join(missing)}")
            sys.exit(1)
        else:
            print("\nAll weights present.")
        return

    # Download SAM2
    if not download_sam2(force=args.force):
        sys.exit(1)

    print()
    check_all()


if __name__ == "__main__":
    main()
