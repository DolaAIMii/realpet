# RealPet — Release & Distribution Guide

This document is for **maintainers** shipping a public release of RealPet.
End users only need `README.md`.

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
6. The app downloads ~1 GB of model weights on first run (see "Network" below).
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

The `.app` downloads ~1 GB of model weights on first run:

- **SAM2** (~156 MB) from `dl.fbaipublicfiles.com` — direct download via `scripts/download_weights.py`
- **BiRefNet-matting** (~900 MB) from **HuggingFace** — auto-downloaded by `transformers.AutoModelForImageSegmentation`
- **Faster R-CNN** (~175 MB) from torchvision — auto-downloaded on first use

If end users are in regions where HuggingFace is slow or blocked, instruct
them to set the mirror **before launching**:

```bash
export HF_ENDPOINT=https://hf-mirror.com
open /Applications/RealPet.app
```

(This works because `transformers` respects the `HF_ENDPOINT` env var.)

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

- **First-launch automatic weight download inside the .app**: the current
  flow requires the user to either (a) run `python scripts/download_weights.py`
  in the repo, or (b) place weights under `~/.cache/huggingface/` manually.
  A Swift-side auto-download + progress UI + mirror switching is a future
  feature (issue TBD).
- **ffmpeg-missing alert UI**: if the user's machine has no `ffmpeg`, the
  app currently fails silently. A `NSAlert` on first launch pointing at
  `brew install ffmpeg` is a future feature (issue TBD).
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
