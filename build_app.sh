#!/usr/bin/env bash
set -euo pipefail

# --stub-weights: skip the ~1.1 GB weight download and bundle empty
# placeholder weights/{sam2,hf,torch} dirs instead. For CI structural
# verification of the .app bundle only — never ship a stubbed build.
STUB_WEIGHTS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --stub-weights) STUB_WEIGHTS=1 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="RealPet"
BUILD_DIR="$SCRIPT_DIR/RealPet/.build/release"
DIST_DIR="$SCRIPT_DIR/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"

GREEN='\033[0;32m'
NC='\033[0m'

ok() { echo -e "${GREEN}✓${NC} $1"; }

echo "=== Building $APP_NAME.app ==="
echo ""

# 1. Swift release build
echo "--- Building Swift ---"
cd "$SCRIPT_DIR/RealPet"
swift build -c release 2>&1 | tail -3
ok "Swift build complete"

# 2. Find the binary
BINARY=$(find "$BUILD_DIR" -name "$APP_NAME" -type f -perm +111 | head -1)
if [ -z "$BINARY" ]; then
    # Fallback: look anywhere under .build/release for the executable
    BINARY=$(find "$SCRIPT_DIR/RealPet/.build" -name "$APP_NAME" -type f -perm +111 | head -1)
fi
if [ -z "$BINARY" ]; then
    echo "Error: could not find built binary in $BUILD_DIR"
    exit 1
fi
ok "Binary: $BINARY"

# 3. Create .app bundle
echo ""
echo "--- Assembling .app bundle ---"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copy binary
cp "$BINARY" "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>RealPet</string>
    <key>CFBundleIdentifier</key>
    <string>com.realpet.app</string>
    <key>CFBundleName</key>
    <string>RealPet</string>
    <key>CFBundleDisplayName</key>
    <string>RealPet</string>
    <key>CFBundleVersion</key>
    <string>0.2.2</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.2</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.entertainment</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
</dict>
</plist>
PLIST

# --- App icon ---
cp "$SCRIPT_DIR/assets/icon/AppIcon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
ok "Bundled app icon"

# --- Bundle Python pipeline source (required at runtime) ---
# PythonBridge.projectRoot locates the pipeline by finding a `pipeline/`
# directory under Resources; SetupWizard reads `requirements.txt` from the
# bundle. Without these the installed .app cannot run the pipeline and the
# first-launch wizard aborts.
RES_DIR="$APP_BUNDLE/Contents/Resources"
for src in pipeline scripts; do
    rsync -a --exclude '__pycache__' --exclude '*.pyc' \
        "$SCRIPT_DIR/$src/" "$RES_DIR/$src/"
done
cp "$SCRIPT_DIR/requirements.txt" "$RES_DIR/requirements.txt"
ok "Bundled Python pipeline source (pipeline/, scripts/, requirements.txt)"

# --- Bundle all model weights (NEW v0.2.0) ---
WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [ "$STUB_WEIGHTS" = "1" ]; then
    # Placeholder tree in a temp dir so the repo's real weights/ is untouched
    echo "--- Stubbing model weights (--stub-weights) ---"
    WEIGHTS_DIR="$(mktemp -d)/weights"
    for d in sam2 hf torch; do
        mkdir -p "$WEIGHTS_DIR/$d"
        touch "$WEIGHTS_DIR/$d/.stub-weights-ci-only"
    done
    ok "Stub weights created (structural check only, app will NOT run)"
elif [ ! -d "$WEIGHTS_DIR/sam2" ] || [ ! -d "$WEIGHTS_DIR/hf" ] \
   || [ ! -d "$WEIGHTS_DIR/torch" ]; then
    echo "--- Bundling model weights (first build, downloads ~1 GB) ---"
    # 用临时 venv 跑 bundle_weights.py(避免要求 build 机器已装 pip)
    TMP_VENV=$(mktemp -d)/venv
    python3 -m venv "$TMP_VENV" >/dev/null
    "$TMP_VENV/bin/pip" install -q huggingface_hub
    REALPET_WEIGHTS_DIR="$WEIGHTS_DIR" \
        "$TMP_VENV/bin/python" "$SCRIPT_DIR/scripts/bundle_weights.py" \
        --out "$WEIGHTS_DIR"
    rm -rf "$(dirname "$TMP_VENV")"
    ok "Weights bundled"
else
    ok "Weights already present, skipping"
fi
cp -r "$WEIGHTS_DIR" "$APP_BUNDLE/Contents/Resources/"
ok "Bundled weights ($(du -sh "$WEIGHTS_DIR" | cut -f1))"

# --- Bundle static ffmpeg (NEW v0.2.0) ---
FFMPEG_DIR="$APP_BUNDLE/Contents/Resources/bin"
mkdir -p "$FFMPEG_DIR"
FFMPEG_TMP=$(mktemp -d)
curl -fsSL -o "$FFMPEG_TMP/ffmpeg.zip" \
    https://evermeet.cx/ffmpeg/getrelease/zip
unzip -q "$FFMPEG_TMP/ffmpeg.zip" -d "$FFMPEG_DIR"
chmod +x "$FFMPEG_DIR/ffmpeg"
rm -rf "$FFMPEG_TMP"
"$FFMPEG_DIR/ffmpeg" -version | head -1
ok "Bundled static ffmpeg"

ok "App bundle assembled"

# 4. Ad-hoc code sign
echo ""
echo "--- Code signing ---"
codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null
ok "Ad-hoc signed"

echo ""
echo -e "${GREEN}=== Done! ===${NC}"
echo "Output: $APP_BUNDLE"
echo ""
echo "To run: open $APP_BUNDLE"
echo ""
echo "First launch sets up Python (~2 min, one-time)."
echo "All model weights and ffmpeg are bundled. No downloads."
echo ""
echo "Ad-hoc-signed .app: right-click → Open the first time (Gatekeeper)."
