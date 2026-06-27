#!/usr/bin/env bash
# Build a drag-to-install DMG for RealPet (end-user friendly).
#
# Usage:
#   ./build_dmg.sh          # builds dist/RealPet.dmg using existing dist/RealPet.app
#
# Prerequisites:
#   1. ./build_app.sh must have run successfully (produces dist/RealPet.app).
#   2. The user must set REALPET_WEIGHTS_DIR before launching the .app
#      (or place weights under <repo>/weights/).
#
# What this does:
#   - Creates a temporary read-only DMG with a symlink to /Applications,
#     so the user can drag the .app into /Applications.
#   - For ad-hoc signed .apps (default), users will need to right-click → Open
#     the first time (Gatekeeper). For Developer ID-signed .apps, no
#     workaround needed — see docs/RELEASE.md.
#
# NOT a notarized build. See docs/RELEASE.md for the notarization recipe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_BUNDLE="$SCRIPT_DIR/dist/RealPet.app"
DMG_OUT="$SCRIPT_DIR/dist/RealPet.dmg"
VOL_NAME="RealPet"
STAGE_DIR=$(mktemp -d)
trap "rm -rf '$STAGE_DIR'" EXIT

if [ ! -d "$APP_BUNDLE" ]; then
    echo "Error: $APP_BUNDLE not found. Run ./build_app.sh first."
    exit 1
fi

echo "=== RealPet DMG Builder ==="
echo ""

# Stage the .app + Applications symlink
echo "Staging .app and Applications symlink..."
cp -R "$APP_BUNDLE" "$STAGE_DIR/RealPet.app"
ln -s /Applications "$STAGE_DIR/Applications"

# Create read-only DMG
echo "Creating DMG (this may take ~30s)..."
rm -f "$DMG_OUT"
hdiutil create -ov -volname "$VOL_NAME" \
    -fs HFS+ -fsargs "-c c=64,a=16,e=16" \
    -srcfolder "$STAGE_DIR" \
    -format UDZO \
    "$DMG_OUT" 2>&1 | tail -5

echo ""
echo "=== Done! ==="
echo "DMG: $DMG_OUT"
ls -lh "$DMG_OUT"
echo ""
echo "End-user install:"
echo "  1. Double-click RealPet.dmg"
echo "  2. Drag RealPet to /Applications"
echo "  3. Eject the disk image"
echo "  4. Launch from /Applications (or Spotlight)"
echo ""
echo "NOTE: First launch downloads ~1 GB of model weights via HuggingFace."
echo "      Optionally set HF_ENDPOINT=https://hf-mirror.com before launching"
echo "      if HuggingFace is slow or blocked in your region."
echo ""
echo "Ad-hoc-signed .app: right-click → Open the first time (Gatekeeper)."
echo "For Developer ID signing + notarization, see docs/RELEASE.md."
