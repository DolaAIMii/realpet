# RealPet / 真实桌宠

Turn a video of your real pet into a transparent animated desktop companion on macOS.

RealPet uses SAM2 video tracking + BiRefNet matting to extract your pet from any video, then displays it as a borderless, always-on-top, draggable window on your desktop.

> **v0.2.0 — Double-click to use**
> - All model weights and ffmpeg are bundled inside the .app. **No first-launch downloads.**
> - First launch only creates a Python virtual environment (~2 minutes, one-time).
> - Download the DMG, drag to /Applications, launch.

## Features

- **Video → Desktop Pet**: Import any video of your pet, get a transparent animated companion
- **AI-Powered Extraction**: SAM2 tracking + BiRefNet matting for high-quality alpha extraction
- **Smart Clip Selection**: Long videos are automatically analyzed for the best pet segments
- **Transparent Window**: Borderless, always-on-top, click-through window with real transparency
- **Drag & Drop**: Move your pet anywhere on the desktop
- **Persistent Pets**: Pets are saved and relaunched on app restart
- **Resident Daemon**: Python daemon keeps models loaded for fast processing

## Requirements

- macOS 14.0+ (Sonoma)
- ~3 GB free disk space (bundled model weights + extracted frames)

> **⚠️ Windows is not supported.** The app uses Swift/AppKit/SwiftUI and Apple's Metal GPU framework (MPS). The Python pipeline could theoretically run on other platforms, but the desktop app requires macOS.

### Tested on

| Date | macOS | Chip | Memory | Notes |
|------|-------|------|--------|-------|
| 2026-06-28 | 14.3.1 | Apple M2 Pro | 16 GB | v0.2.0 clean-machine fresh-clone verify; all weights bundled. |
| 2026-06-27 | 14.3.1 | Apple M2 Pro | 16 GB | Agent-verified on maintainer's machine; not a clean-machine fresh-clone. See `docs/RELEASE.md` for the reproducibility protocol. |

### Performance Recommendations

RealPet runs real-time AI models (SAM2 tracking + BiRefNet matting + Faster R-CNN detection) on every frame. Performance matters.

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Chip** | Apple M1 | Apple M1 Pro or better | MPS (Metal) GPU acceleration is critical. Intel Macs work but are **significantly slower** (CPU-only inference ~5-10× slower). |
| **Memory** | 8GB | 16GB+ | SAM2 + BiRefNet + Faster R-CNN models together use ~3-4GB. With the OS and app, 8GB gets tight. 16GB gives comfortable headroom. |
| **Disk** | 3GB free | 5GB+ free | Model weights: SAM2 ~156MB, BiRefNet ~900MB, Faster R-CNN ~175MB. Plus space for extracted frames and output. |
| **GPU** | Apple M1 integrated | Apple M1 Pro/Max/Ultra or M2/M3/M4 | More GPU cores = faster inference. M1 base (~8 GPU cores) processes ~1 frame/2s. M1 Pro+ (~16 cores) processes ~1 frame/1s. |

**Processing time estimates (10-second video):**

| Chip | QC Gate | Pet Detection | Full Pipeline |
|------|---------|---------------|---------------|
| M1 (8 GPU) | ~4s | ~3s | ~60-90s |
| M1 Pro (16 GPU) | ~2s | ~1.5s | ~30-45s |
| M2 Pro/M3 Pro | ~1.5s | ~1s | ~20-35s |

**If processing feels slow:**
- Use shorter videos (5-15 seconds is ideal)
- Close other GPU-intensive apps (Final Cut, games, etc.)
- Ensure you're on Apple Silicon (Intel Macs will be very slow)
- The app shows progress — SAM2 tracking is the longest step

## Quick Start

1. Download `RealPet.dmg` from the [latest release](https://github.com/DolaAIMii/realpet/releases/latest).
2. Double-click the DMG and drag `RealPet.app` to `/Applications`.
3. Launch from `/Applications`. First run sets up Python (~2 minutes, one-time).

Import a video of your pet through the app's Import button, and RealPet will extract, segment, and display it as a desktop companion.

For developers who prefer to build from source, see `docs/RELEASE.md`.

## Architecture

```
┌─────────────────────────────────────────────┐
│  macOS App (Swift/SwiftUI)                  │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ ControlPanel │  │ TransparentPetWindow │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │              │
│  ┌──────┴────────────────────┴───────────┐  │
│  │ PythonBridge / PythonDaemon           │  │
│  │ (stdin/stdout NDJSON to Python)       │  │
│  └──────┬────────────────────────────────┘  │
└─────────┼───────────────────────────────────┘
          │
┌─────────┴───────────────────────────────────┐
│  Python Pipeline                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Faster   │  │   SAM2   │  │ BiRefNet │  │
│  │ R-CNN    │→ │ Tracking │→ │ Matting  │  │
│  │ (detect) │  │ (track)  │  │ (alpha)  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

**Pipeline**: `detect_pet` → `analyze_clips` → `track_then_matte` (SAM2 + BiRefNet) → transparent display

## Project Structure

```
realpet/
├── RealPet/              # Swift macOS app
│   ├── DeskPetApp.swift  # App entry (file name kept for SwiftPM target)
│   ├── Models/           # Pet model
│   ├── Services/         # PetLauncher, PetStorage, PythonBridge, PythonDaemon
│   ├── ViewModels/       # PetListViewModel
│   ├── Views/            # MainPanelView, PetRowView
│   └── Package.swift
├── pipeline/             # Python AI pipeline
│   ├── pet_detector.py   # Faster R-CNN pet detection
│   ├── segment.py        # BiRefNet segmentation
│   ├── display.py        # Transparent window rendering
│   ├── cli.py            # AI-animate pipeline entry
│   └── smart_clip.py     # Long-video clip selection
├── scripts/              # CLI tools
│   ├── track_then_matte.py  # Main pipeline (SAM2 + BiRefNet)
│   ├── detect_pet.py     # Pet detection
│   ├── quality_check.py  # QC gate
│   ├── analyze_clips.py  # Clip selection for long videos
│   ├── daemon.py         # Resident Python worker
│   └── download_weights.py # Weight downloader
├── tests/                # Unit tests (pytest)
├── weights/              # Model weights (git-ignored, downloaded)
├── requirements.txt      # Python dependencies
├── install.sh            # One-click installer
├── build_app.sh          # .app builder
├── LICENSE               # MIT
└── NOTICE                # Third-party licenses
```

## Model Weights

All model weights are bundled inside `RealPet.app`. No downloads are required on first launch.

| Model | Size | Source |
|-------|------|--------|
| SAM2 (`sam2.1_hiera_tiny.pt`) | ~156MB | [Meta](https://github.com/facebookresearch/sam2) |
| BiRefNet-matting | ~900MB | [ZhengPeng7](https://huggingface.co/ZhengPeng7/BiRefNet-matting) |
| Faster R-CNN | ~175MB | [PyTorch](https://pytorch.org/vision/stable/models.html) |

If you are building from source, `scripts/bundle_weights.py` downloads all three into the `weights/` directory.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `REALPET_WEIGHTS_DIR` | `weights/` | Directory for model weights |
| `REALPET_VENV` | `.venv` (project root) | Python virtual environment path |
| `REALPET_PROJECT_ROOT` | auto-detected | Project root directory override |

## Building the App

```bash
./build_app.sh
# Output: dist/RealPet.app (ad-hoc signed)
```

## License

[MIT](LICENSE)

## Acknowledgments

- [SAM 2](https://github.com/facebookresearch/sam2) — Meta (Apache-2.0)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — ZhengPeng7 (MIT)
- [PyTorch](https://github.com/pytorch/pytorch) — Meta (BSD-3)
