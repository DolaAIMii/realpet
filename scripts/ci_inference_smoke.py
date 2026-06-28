#!/usr/bin/env python3
"""CI inference smoke test.

This script intentionally avoids the full pipeline (no real pet video needed)
and instead proves that the key AI dependencies are importable and runnable:

1. PyTorch can create a device (mps if available, else cpu).
2. BiRefNet-matting can load from HuggingFace and run a forward pass.
3. torchvision Faster R-CNN can load and run a forward pass.

It exits 0 on success and 1 on failure.
"""
import os
import sys
import time

import torch
import torchvision
from PIL import Image
from transformers import AutoModelForImageSegmentation


def _device():
    # Force CPU on macOS CI: GitHub Actions macos-14 runners advertise MPS but
    # the MPS memory pool (~7.9 GiB hard cap) is too tight for BiRefNet
    # (~900 MB weights) plus a real forward pass. Forcing CPU here keeps the
    # smoke test about "do the imports + forward path work?" rather than
    # "can MPS fit this model in the runner's shared memory?". On a real
    # user machine the pipeline uses mps when available and falls back to
    # cpu automatically (see pipeline/segment.py + scripts/track_then_matte.py).
    return torch.device("cpu")


def smoke_birefnet(device):
    print("Loading BiRefNet-matting...")
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet-matting", trust_remote_code=True
    )
    model.to(device)
    model.eval()

    # Synthesize a tiny RGB image
    img = Image.new("RGB", (256, 256), color=(120, 90, 60))
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = torch.sigmoid(pred)

    assert pred.shape[0] == 1
    print(f"  BiRefNet OK on {device} (output shape {tuple(pred.shape)})")


def smoke_faster_rcnn(device):
    print("Loading Faster R-CNN...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    model.to(device)
    model.eval()

    # Synthesize a tiny batch
    dummy = torch.rand(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model([dummy.squeeze(0)])

    assert isinstance(out, list)
    print(f"  Faster R-CNN OK on {device} (found {len(out[0].get('boxes', []))} boxes)")


def main():
    device = _device()
    print(f"Selected device: {device}")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'default')}")

    # Optional: let HF cache go to a temp dir for isolation
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".ci_cache", "hf"))

    t0 = time.time()
    smoke_birefnet(device)
    smoke_faster_rcnn(device)
    elapsed = time.time() - t0
    print(f"All smoke tests passed in {elapsed:.1f}s on {device}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
