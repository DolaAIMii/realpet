# RealPet / 真实桌宠

Turn a video of your real pet into a transparent animated desktop companion on macOS.

RealPet uses SAM2 video tracking + BiRefNet matting to extract your pet from any video, then displays it as a borderless, always-on-top, draggable window on your desktop.

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
- Python 3.10+ (for the AI pipeline)
- ffmpeg (for video frame extraction)
- ~2GB free disk space (model weights)

## Quick Start

```bash
# Clone
git clone https://github.com/yourname/realpet.git
cd realpet

# Install dependencies
./install.sh

# Run
cd DeskPet && swift run
```

Import a video of your pet through the app's Import button, and RealPet will extract, segment, and display it as a desktop companion.

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
├── DeskPet/              # Swift macOS app
│   ├── Services/         # PythonBridge, PythonDaemon, PetStorage
│   ├── ViewModels/       # PetListViewModel
│   ├── Views/            # MainPanelView, PetRowView
│   └── Package.swift
├── pipeline/             # Python AI pipeline
│   ├── pet_detector.py   # Faster R-CNN pet detection
│   ├── segment.py        # BiRefNet segmentation
│   └── display.py        # Transparent window rendering
├── scripts/              # CLI tools
│   ├── track_then_matte.py  # Main pipeline (SAM2 + BiRefNet)
│   ├── detect_pet.py     # Pet detection
│   ├── quality_check.py  # QC gate
│   ├── analyze_clips.py  # Clip selection for long videos
│   ├── daemon.py         # Resident Python worker
│   └── download_weights.py # Weight downloader
├── weights/              # Model weights (git-ignored, downloaded)
├── requirements.txt      # Python dependencies
├── install.sh            # One-click installer
├── build_app.sh          # .app builder
├── LICENSE               # MIT
└── NOTICE                # Third-party licenses
```

## Model Weights

| Model | Size | Auto-download | Source |
|-------|------|---------------|--------|
| SAM2 (`sam2.1_hiera_tiny.pt`) | ~156MB | `download_weights.py` | [Meta](https://github.com/facebookresearch/sam2) |
| BiRefNet-matting | ~900MB | HuggingFace `from_pretrained()` | [ZhengPeng7](https://huggingface.co/ZhengPeng7/BiRefNet-matting) |
| Faster R-CNN | ~175MB | torchvision auto-download | [PyTorch](https://pytorch.org/vision/stable/models.html) |

Run `python scripts/download_weights.py` to download SAM2. BiRefNet and Faster R-CNN are fetched automatically on first use.

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
