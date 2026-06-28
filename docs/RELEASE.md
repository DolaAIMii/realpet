# RealPet — Release & Distribution Guide

This document is for **maintainers** shipping a public release of RealPet.
End users only need `README.md`.

## v0.2.0 — Double-click to use (2026-06-28)

### What's new

- **All model weights bundled** in the .app: SAM2 (~156 MB) + BiRefNet-matting (~900 MB) + Faster R-CNN (~175 MB).
- **Static ffmpeg bundled** — no `brew install ffmpeg` required.
- **First-launch SetupWizard** auto-creates the Python venv. Users only need Python 3.10+ installed (one `brew install python@3.12`, no sudo on Apple Silicon).
- DMG compression switched to ULFO (LZFSE) on Apple Silicon; Intel fallback remains UDZO.

### End-user install

1. Download `RealPet.dmg`.
2. Double-click and drag `RealPet.app` to `/Applications`.
3. Launch. First run sets up Python (~2 minutes, one-time).

### Maintainer build

```bash
./build_app.sh   # downloads weights on first run; produces dist/RealPet.app
./build_dmg.sh   # produces dist/RealPet.dmg
```

Expected DMG size: ~1.05–1.15 GB (was 139 MB in v0.1.0).

---

## What this repo ships

- A Swift/SwiftUI macOS app (`RealPet/`)
- A Python AI pipeline (`pipeline/` + `scripts/`)
- An ad-hoc-signed `.app` builder (`build_app.sh`)
- A drag-to-install `.dmg` builder (`build_dmg.sh`)
- **NOT** a notarized, Developer-ID-signed, or App Store build.

Out of the box, the `.app` is **ad-hoc signed** (your machine's local identity).
macOS Gatekeeper will warn end users the first time they open it. To ship
without the warning, you need a **paid Apple Developer Program membership**
(US$99/yr), sign with your Developer ID, and **notarize** with Apple.

This document walks you through that.

## TL;DR

| Audience | What to ship | How |
|----------|--------------|-----|
| Friends / testers | `RealPet.dmg` (ad-hoc signed) | `./build_dmg.sh` |
| Public release | `RealPet.dmg` (Developer ID + notarized) | See below |
| App Store | Out of scope (this guide, not the codebase) | — |

## 1. Build the ad-hoc-signed .app + DMG

This is what the current `build_app.sh` and `build_dmg.sh` produce. It works
for testing on your own machine and for sharing with a small group of
people who don't mind right-click → Open the first time.

```bash
./install.sh              # one-time setup
cd RealPet && swift build -c release && cd ..   # build binary
./build_app.sh            # produces dist/RealPet.app (ad-hoc signed)
./build_dmg.sh            # produces dist/RealPet.dmg (drag-to-install)
```

End-user experience (ad-hoc):

1. Download `RealPet.dmg`, double-click.
2. Drag `RealPet.app` to `/Applications`.
3. Eject the disk image.
4. Launch from `/Applications` (or Spotlight).
5. **First-launch Gatekeeper prompt**: right-click the .app → Open → Open
   (only needed once; macOS records the exception).
6. **First-launch SetupWizard**: if Python 3.10+ is not found, copy the
   `brew install python@3.12` command into Terminal (no sudo on Apple Silicon),
   then click **重新检测 / Retry**. The wizard creates a venv and installs
   dependencies (~2 minutes, one-time).
7. Done.

## 2. Build a Developer-ID-signed + notarized DMG (public release)

Prerequisites:

- **Apple Developer Program** membership (US$99/year): <https://developer.apple.com/programs/>
- Your **Team ID** (10-character alphanumeric, e.g. `ABCDE12345`)
- A **Developer ID Application** certificate installed in your keychain
  (Xcode → Settings → Accounts → select team → Manage Certificates → +)
- An **app-specific password** for `notarytool` (Apple ID → Sign-In & Security
  → App-Specific Passwords → Generate)

### 2a. Sign with your Developer ID

Replace `TEAMID` with your actual Team ID:

```bash
codesign --force --deep --options=runtime \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    dist/RealPet.app

# Verify
codesign --verify --deep --strict --verbose=2 dist/RealPet.app
spctl --assess --type execute --verbose dist/RealPet.app
# Expected: dist/RealPet.app: accepted
```

### 2b. Notarize

Save your app-specific password in the keychain (one-time):

```bash
xcrun notarytool store-credentials "notary-profile" \
    --apple-id "you@example.com" \
    --team-id "TEAMID" \
    --password "abcd-efgh-ijkl-mnop"   # the app-specific password
```

Submit for notarization:

```bash
# Zip the .app first (notarytool prefers zip / dmg; zip is simpler)
ditto -c -k --keepParent dist/RealPet.app dist/RealPet.zip

xcrun notarytool submit dist/RealPet.zip \
    --keychain-profile "notary-profile" \
    --wait
# Expected: status: Accepted
```

If notarization fails, get the log:

```bash
xcrun notarytool log <submission-id> --keychain-profile "notary-profile"
```

Common rejections and fixes:

| Rejection | Cause | Fix |
|-----------|-------|-----|
| "The binary uses an SDK older than the 14.0 SDK" | Built on older Xcode | Build with Xcode 15+ |
| "The signature does not include a secure timestamp" | Forgot `--options=runtime` | Re-sign with the flag |
| "Unsealed contents present" | Code signature broken by zip manipulation | Use `ditto`, not `zip` |

### 2c. Staple the notarization ticket to the .app

After notarization succeeds, attach the ticket so end users don't need
internet to verify:

```bash
xcrun stapler staple dist/RealPet.app
xcrun stapler validate dist/RealPet.app
```

### 2d. Rebuild the DMG and re-sign

The DMG itself must also be signed. Easiest path:

```bash
./build_dmg.sh                                   # rebuild
codesign --force --sign "Developer ID Application: Your Name (TEAMID)" dist/RealPet.dmg
xcrun notarytool submit dist/RealPet.dmg --keychain-profile "notary-profile" --wait
xcrun stapler staple dist/RealPet.dmg
```

### 2e. Publish on GitHub Releases

1. Push the tag:
   ```bash
   git tag -s v0.1.0 -m "RealPet v0.1.0"   # signed tag
   git push origin v0.1.0
   ```
2. On GitHub → Releases → Draft a new release → pick the tag → upload
   `dist/RealPet.dmg` and a SHA256 checksum:
   ```bash
   shasum -a 256 dist/RealPet.dmg
   ```
3. Mark it "Set as the latest release".

## 3. Network considerations on first launch

**RESOLVED in v0.2.0.** All model weights are now bundled in the `.app`:

- **SAM2** (~156 MB) — bundled at `Resources/weights/sam2/`
- **BiRefNet-matting** (~900 MB) — bundled at `Resources/weights/hf/`
- **Faster R-CNN** (~175 MB) — bundled at `Resources/weights/torch/`

No internet connection is required for first launch. The only first-launch
setup is creating the Python venv and installing pip dependencies, which uses
PyPI. If PyPI is slow or blocked in your region, set a mirror before launching:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
open /Applications/RealPet.app
```

(The default PyPI endpoint is `https://pypi.org/simple`.)

`HF_ENDPOINT` is still honored for any code path that does hit HuggingFace,
but it is no longer required for normal use.

## 4. Reproducibility — what was actually tested

This release guide assumes the maintainer has a clean machine. The
`Tested on` table in `README.md` records what was verified, when, and on
what hardware. Update it whenever you re-verify on a new machine:

```
| Date | macOS | Chip | Memory | Notes |
|------|-------|------|--------|-------|
| YYYY-MM-DD | 14.x.x | Apple M2 Pro | 16 GB | Maintainer fresh-clone verify |
```

A "fresh-clone verify" means:

1. Wipe a test Mac (or use a throwaway VM).
2. Install only Xcode Command Line Tools (`xcode-select --install`).
3. Clone the repo, run `./install.sh`, then `./build_app.sh`.
4. Launch the .app, import a pet video, verify a desktop pet appears.
5. Time each step (this is your TTHW / "time to hello world").

## 5. Known gaps (out of scope for this script)

These are **not** solved by `build_dmg.sh`. They are listed here for
transparency so users filing issues don't get bounced:

- **Auto-update**: not implemented. Users download new releases manually
  from GitHub Releases.

These are minimum-viable-release limitations, not blockers for an initial
public release.

## 6. Security checklist before publishing

- [ ] `./install.sh` runs cleanly on a clean machine
- [ ] `./build_app.sh` produces a launching .app
- [ ] `./build_dmg.sh` produces a usable DMG
- [ ] If shipping Developer-ID-signed: `spctl --assess` accepts the .app
- [ ] If shipping notarized: `xcrun stapler validate` passes
- [ ] DMG SHA256 checksum published alongside the release
- [ ] `README.md` "Tested on" row updated with date + hardware
- [ ] No `*.pkl`, `*.pt`, or personal paths accidentally committed
  (`git log -p HEAD~5..HEAD | grep -E "\.pkl|/\w+/\w+/Desktop"` should
  print nothing)
