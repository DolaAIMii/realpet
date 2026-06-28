#!/usr/bin/env python3
"""Bundle ALL model weights for the v0.2.0 double-click DMG.

Layout (under --out):
  weights/sam2/sam2.1_hiera_tiny.pt
  weights/hf/hub/models--ZhengPeng7-BiRefNet-matting/snapshots/<sha>/
  weights/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco32689bffd.pth

Idempotent + --verify-only + honors HF_ENDPOINT (auto via huggingface_hub).
"""
import argparse
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.environ.get(
    "REALPET_WEIGHTS_DIR", os.path.join(PROJECT_ROOT, "weights"))

BIREFNET_REPO = "ZhengPeng7/BiRefNet-matting"
RCNN_URL = ("https://download.pytorch.org/models/"
            "fasterrcnn_resnet50_fpn_v2_coco32689bffd.pth")
RCNN_EXPECTED_SIZE = 173_000_000


def _ensure_env(out_dir):
    """download_weights.py reads REALPET_WEIGHTS_DIR at import time."""
    os.environ.setdefault("REALPET_WEIGHTS_DIR", out_dir)


def download_birefnet(out_dir, force=False):
    from huggingface_hub import snapshot_download
    dest = os.path.join(out_dir, "hf")
    if not force and os.path.isdir(os.path.join(dest, "hub")):
        print("BiRefNet already present")
        return
    print(f"Downloading BiRefNet-matting to {dest} ...")
    snapshot_download(
        repo_id=BIREFNET_REPO,
        local_dir=dest,
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.bin",
                        "*.txt", "*.md"],
    )


def download_faster_rcnn(out_dir, force=False):
    dest_dir = os.path.join(out_dir, "torch", "hub", "checkpoints")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(RCNN_URL))
    if os.path.exists(dest) and not force:
        size = os.path.getsize(dest)
        if size >= RCNN_EXPECTED_SIZE * 0.9:
            print(f"Faster R-CNN already present: {dest} ({size:,} bytes)")
            return
        print(f"Existing Faster R-CNN too small ({size:,} bytes), "
              "re-downloading...")
    print(f"Downloading Faster R-CNN to {dest} ...")
    urllib.request.urlretrieve(RCNN_URL, dest)
    size = os.path.getsize(dest)
    print(f"Done. ({size:,} bytes)")


def verify(out_dir):
    """Exit 0 if all three weights are present and sized, 1 otherwise."""
    _ensure_env(out_dir)
    from download_weights import SAM2_EXPECTED_SIZE

    sam2_path = os.path.join(out_dir, "sam2", "sam2.1_hiera_tiny.pt")
    checks = [
        ("SAM2", sam2_path, SAM2_EXPECTED_SIZE, "file"),
        ("BiRefNet", os.path.join(out_dir, "hf", "hub"),
         100_000_000, "dir"),
        ("Faster R-CNN", os.path.join(out_dir, "torch", "hub", "checkpoints",
                                      os.path.basename(RCNN_URL)),
         RCNN_EXPECTED_SIZE, "file"),
    ]
    ok = True
    for label, path, floor, kind in checks:
        if not os.path.exists(path):
            print(f"  !! {label}: missing at {path}")
            ok = False
            continue
        if kind == "file":
            size = os.path.getsize(path)
        else:
            size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path)
                for f in files
            )
        marker = "ok" if size >= floor * 0.9 else "!!"
        if marker == "!!":
            ok = False
        print(f"  {marker} {label}: {size / 1024 / 1024:,.0f} MB")
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle all model weights for the RealPet double-click DMG.")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output directory for weights (default: weights/ or "
             "REALPET_WEIGHTS_DIR).")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify existing weights and exit.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _ensure_env(args.out)

    if args.verify_only:
        verify(args.out)
        return

    from download_weights import download_sam2

    print("Bundling SAM2...")
    download_sam2(force=args.force)
    print()
    print("Bundling BiRefNet...")
    download_birefnet(args.out, force=args.force)
    print()
    print("Bundling Faster R-CNN...")
    download_faster_rcnn(args.out, force=args.force)
    print()
    verify(args.out)


if __name__ == "__main__":
    main()
