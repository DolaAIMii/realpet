#!/usr/bin/env bash
set -euo pipefail

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
    # Fallback: look for the target name
    BINARY=$(find "$BUILD_DIR" -name "RealPet" -type f | head -1)
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
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.entertainment</string>
</dict>
</plist>
PLIST

# Copy pipeline and scripts as resources
cp -r "$SCRIPT_DIR/pipeline" "$APP_BUNDLE/Contents/Resources/"
cp -r "$SCRIPT_DIR/scripts" "$APP_BUNDLE/Contents/Resources/"

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
echo "Weights are NOT bundled (too large, ~1 GB)."
echo "Point the app at your weights directory before first run:"
echo "  export REALPET_WEIGHTS_DIR=/path/to/your/weights"
echo "Default search path (if env unset): <repo>/weights/"
echo "  python scripts/download_weights.py  # to fetch the official set"
