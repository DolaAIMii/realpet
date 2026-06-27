#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REALPET_VENV:-$SCRIPT_DIR/.venv}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }

echo "=== RealPet Installer ==="
echo ""

# 1. Check Python 3.10+ (prefer 3.12 > 3.11 > 3.10 > system python3)
echo "--- Checking dependencies ---"
PYTHON=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" &>/dev/null; then
        PY_VER=$("$cand" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON="$cand"
            ok "Python $PY_VER ($cand)"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    if command -v python3 &>/dev/null; then
        CUR=$("python3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        fail "Python 3.10+ required, found $CUR (python3). Install with: brew install python@3.12"
    else
        fail "Python 3 not found. Install with: brew install python@3.12"
    fi
fi

# 2. Check ffmpeg
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    fail "ffmpeg not found. Install with: brew install ffmpeg"
fi

# 3. Create venv
echo ""
echo "--- Setting up Python environment ---"
if [ -d "$VENV_DIR" ]; then
    warn "Venv already exists at $VENV_DIR, skipping creation"
else
    echo "Creating venv at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Venv created"
fi

# 4. Install Python dependencies
echo ""
echo "--- Installing Python dependencies ---"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q
ok "Python dependencies installed"

# 5. Install SAM2
echo ""
echo "--- Installing SAM2 ---"
if "$VENV_DIR/bin/python" -c "import sam2" 2>/dev/null; then
    ok "SAM2 already installed"
else
    echo "Installing SAM2 from Meta ..."
    "$VENV_DIR/bin/pip" install sam2 -q 2>/dev/null || {
        warn "SAM2 pip install failed, trying from source ..."
        "$VENV_DIR/bin/pip" install "git+https://github.com/facebookresearch/sam2.git" -q
    }
    ok "SAM2 installed"
fi

# 6. Download weights
echo ""
echo "--- Downloading model weights ---"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/download_weights.py"

# 7. Verify
echo ""
echo "--- Verification ---"
MISSING=$("$VENV_DIR/bin/python" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from scripts.track_then_matte import check_dependencies
m = check_dependencies()
print('\n'.join(m))
" 2>/dev/null)

if [ -z "$MISSING" ]; then
    ok "All dependencies satisfied"
else
    fail "Missing dependencies:\n$MISSING"
fi

echo ""
echo -e "${GREEN}=== Installation complete! ===${NC}"
echo ""
echo "To run the app:"
echo "  cd RealPet && swift build -c release && swift run"
echo ""
echo "Or build the .app bundle:"
echo "  ./build_app.sh"
