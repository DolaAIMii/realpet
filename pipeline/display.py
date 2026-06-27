"""Transparent desktop pet window using PyObjC + NSWindow with per-pixel alpha."""
import os
import math
import json
import numpy as np
from PIL import Image

import objc
from Cocoa import (
    NSApplication, NSApp, NSWindow, NSView, NSImage, NSColor, NSBezierPath,
    NSCursor, NSTrackingArea, NSEvent, NSScreen,
    NSBitmapImageRep, NSDeviceRGBColorSpace, NSBitmapFormatAlphaNonpremultiplied,
    NSBorderlessWindowMask, NSFloatingWindowLevel,
    NSCompositingOperationSourceOver,
    NSMakeRect, NSMakePoint, NSSize,
    NSTrackingMouseEnteredAndExited, NSTrackingActiveInActiveApp,
    NSTrackingInVisibleRect,
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _crop_to_foreground(frames, margin=20):
    """Find the union bounding box of all foreground (alpha > 0) regions."""
    min_r, min_c, max_r, max_c = None, None, None, None
    for img in frames:
        arr = np.array(img)
        alpha = arr[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if not rows.any() or not cols.any():
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        min_r = r0 if min_r is None else min(min_r, r0)
        min_c = c0 if min_c is None else min(min_c, c0)
        max_r = r1 if max_r is None else max(max_r, r1)
        max_c = c1 if max_c is None else max(max_c, c1)
    if min_r is None:
        return frames
    w, h = frames[0].size
    min_r = max(0, min_r - margin)
    min_c = max(0, min_c - margin)
    max_r = min(h - 1, max_r + margin)
    max_c = min(w - 1, max_c + margin)
    return [img.crop((min_c, min_r, max_c + 1, max_r + 1)) for img in frames]


def _nsimage_from_raw(raw, w, h, logical_size=None):
    """Build an NSImage from raw straight-alpha RGBA bytes (w*h*4).

    Goes straight to an NSBitmapImageRep — no PNG/TIFF round-trip. PIL gives
    STRAIGHT (non-premultiplied) alpha, so the bitmap must be tagged
    NSBitmapFormatAlphaNonpremultiplied. Without it Cocoa assumes premultiplied
    and over-shows the RGB at soft edges → a light/colour halo on dark
    backgrounds.

    If logical_size (w, h) in points is given, the NSImage keeps that point
    size while retaining the bitmap's full pixel resolution — so on a Retina
    (2x) screen the high-res bitmap renders crisp instead of being upscaled.
    """
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel_(
        None, w, h, 8, 4, True, False, NSDeviceRGBColorSpace,
        int(NSBitmapFormatAlphaNonpremultiplied), w * 4, 32)
    rep.bitmapData()[:] = raw
    size = NSSize(logical_size[0], logical_size[1]) if logical_size else NSSize(w, h)
    img = NSImage.alloc().initWithSize_(size)
    img.addRepresentation_(rep)
    return img


def _pil_to_nsimage(rgba_img, logical_size=None):
    """Convert a PIL RGBA Image to NSImage (preserving alpha)."""
    w, h = rgba_img.size
    return _nsimage_from_raw(rgba_img.tobytes("raw", "RGBA"), w, h, logical_size)


# ---------------------------------------------------------------------------
# Display cache — packs the cropped + Retina-resized frames as one raw blob so
# re-opening a pet skips PNG decode + LANCZOS resize (the ~1.9s open delay) and
# just memcpys bytes into NSImages. Keyed by the source frames + screen scale;
# invalidated automatically if either changes.
# ---------------------------------------------------------------------------

_CACHE_DIR = ".petcache"
_CACHE_VERSION = 3  # bumped: native-res master + subject-area normalized window

# Long-side cap (px) for the master frame bitmap. Keeps the pet at native
# resolution for crisp base + zoom display while bounding memory / open latency.
MAX_MASTER_PX = 1000

# Subject-size normalization. Every pet opens so its foreground occupies a fixed
# on-screen area (points²), so the cat looks the SAME size regardless of source
# resolution or landscape/portrait orientation — instead of "fit longest side to
# 400pt", which made a tall sitting pose render ~2x bigger than a wide lying one.
# Anchored to IMG_0847 at the size the user approved ("适中"): the prior
# longest-side-to-400pt rule gave scale 400/786 ≈ 0.509 on its union crop, so the
# cat's foreground (≈101389px) occupied 101389 × 0.509² ≈ 26250 pt² on screen.
# Matching that area reproduces the current size exactly while normalizing every
# other clip to it. Scale is clamped so we never blow up a tiny pet or shrink a
# frame-filling one to absurdity.
TARGET_FG_AREA_PT2 = 26250
SUBJECT_SCALE_MIN, SUBJECT_SCALE_MAX = 0.12, 4.0


def _median_fg_area(frames, sample=7):
    """Median foreground (alpha>40) pixel count over a sample of RGBA frames."""
    n = len(frames)
    if n == 0:
        return 0
    idx = (range(n) if n <= sample
           else [int(round(i * (n - 1) / (sample - 1))) for i in range(sample)])
    areas = []
    for i in idx:
        a = np.array(frames[i])[:, :, 3]
        areas.append(int((a > 40).sum()))
    return int(np.median(areas)) if areas else 0


def _src_signature(frame_paths):
    """Cheap fingerprint of the source frames (count + first/last mtime+size)."""
    try:
        s0 = os.stat(frame_paths[0])
        s1 = os.stat(frame_paths[-1])
        return "%d:%d:%d:%d:%d" % (len(frame_paths), int(s0.st_mtime),
                                   s0.st_size, int(s1.st_mtime), s1.st_size)
    except OSError:
        return None


def _cache_paths(frames_dir):
    cdir = os.path.join(frames_dir, _CACHE_DIR)
    return cdir, os.path.join(cdir, "meta.json"), os.path.join(cdir, "frames.raw")


def _load_display_cache(frames_dir, scale, sig):
    """Return (list[NSImage], (tw, th)) from cache, or None on miss/mismatch."""
    _, meta_p, raw_p = _cache_paths(frames_dir)
    if not (os.path.exists(meta_p) and os.path.exists(raw_p)):
        return None
    try:
        meta = json.load(open(meta_p))
    except (ValueError, OSError):
        return None
    if (meta.get("version") != _CACHE_VERSION or meta.get("scale") != scale
            or meta.get("sig") != sig):
        return None
    pw, ph, tw, th, n = (meta.get(k) for k in ("pw", "ph", "tw", "th", "count"))
    if not all(isinstance(v, int) for v in (pw, ph, tw, th, n)):
        return None
    fsz = pw * ph * 4
    try:
        with open(raw_p, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) != fsz * n:
        return None
    imgs = [_nsimage_from_raw(data[i * fsz:(i + 1) * fsz], pw, ph, (tw, th))
            for i in range(n)]
    return imgs, (tw, th)


def _write_display_cache(frames_dir, scale, sig, resized_imgs, tw, th):
    """Pack the prepared frames to disk. Best-effort; never raises."""
    if sig is None:
        return
    cdir, meta_p, raw_p = _cache_paths(frames_dir)
    pw, ph = resized_imgs[0].size
    try:
        os.makedirs(cdir, exist_ok=True)
        tmp = raw_p + ".tmp"
        with open(tmp, "wb") as f:
            for im in resized_imgs:
                f.write(im.tobytes("raw", "RGBA"))
        os.replace(tmp, raw_p)
        json.dump({"version": _CACHE_VERSION, "scale": scale, "sig": sig,
                   "count": len(resized_imgs), "pw": pw, "ph": ph,
                   "tw": tw, "th": th}, open(meta_p, "w"))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# PetView — renders frames, handles interaction
# ---------------------------------------------------------------------------

class PetView(NSView):
    """Custom view that renders animated RGBA frames with close button and resize handle."""

    def initWithFrame_(self, frame):
        self = objc.super(PetView, self).initWithFrame_(frame)
        if self is None:
            return None

        self._frames = []        # list of NSImage
        self._frame_idx = 0
        self._direction = 1      # 1=forward, -1=backward (ping-pong)
        self._is_hovering_close = False
        self._is_dragging_resize = False
        self._is_dragging_window = False
        self._drag_start_screen = None
        self._drag_win_origin = None
        self._resize_start = None
        self._resize_win_start = None

        # Base (initial) window size in points — the subject-normalized size,
        # which is also the MAXIMUM (100%). The pet can only be scaled DOWN from
        # here, to a floor of 60% of its area (= sqrt(0.6) ≈ 0.775 linear), so
        # every pet stays predictable and never larger than the normalized size.
        self._base_w = 0.0
        self._base_h = 0.0
        self._min_area_frac = 0.40                  # smallest allowed area vs base
        self._min_scale = math.sqrt(self._min_area_frac)  # linear floor ≈ 0.632
        self._max_scale = 1.0                       # base size is the ceiling
        # Subject normalization keeps every pet a consistent size; the user may
        # now nudge a single pet smaller within [60% area, 100%] via scroll,
        # trackpad pinch, or the drag handle. Set True to lock size entirely.
        self._size_locked = False
        # Persisted per-pet scale (loaded from frames dir on open).
        self._scale_store_path = None

        # Close button geometry (relative to view bounds)
        self._close_radius = 10.0
        self._close_margin = 6.0

        # Resize handle geometry
        self._resize_size = 16.0

        # Tracking area for hover detection
        ta = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            self.bounds(),
            NSTrackingMouseEnteredAndExited | NSTrackingActiveInActiveApp | NSTrackingInVisibleRect,
            self, None
        )
        self.addTrackingArea_(ta)

        return self

    def acceptsFirstMouse_(self, event):
        return True

    # -- Frame management --

    def set_frames(self, nsimages):
        """Set animation frames (list of NSImage)."""
        self._frames = nsimages
        self._frame_idx = 0
        self._direction = 1

    def set_base_size(self, w, h):
        """Record the normalized (100%) window size: aspect ratio + scale ceiling."""
        self._base_w = float(w)
        self._base_h = float(h)

    def set_scale_store(self, frames_dir):
        """Point persistence at this pet's frames dir and return a saved scale
        (linear, relative to base) if one exists, else None."""
        if not frames_dir:
            return None
        self._scale_store_path = os.path.join(frames_dir, ".pet_scale.json")
        try:
            data = json.load(open(self._scale_store_path))
            s = float(data.get("scale", 1.0))
            # Re-clamp against the live floor/ceiling in case bounds changed.
            return max(self._min_scale, min(s, self._max_scale))
        except (OSError, ValueError, KeyError):
            return None

    def _save_scale(self):
        """Persist the current linear scale (window width / base width)."""
        if not self._scale_store_path or not self._base_w:
            return
        scale = self.window().frame().size.width / self._base_w
        try:
            json.dump({"scale": round(scale, 4)}, open(self._scale_store_path, "w"))
        except OSError:
            pass

    def _clamp_size(self, w, h):
        """Clamp a target window size to [60% area, 100%] of the base size,
        preserving the pet's aspect ratio (so it never stretches). The base
        (subject-normalized) size is the ceiling; the floor is sqrt(0.6) of it.
        """
        # Always derive from the base aspect — this both preserves shape and
        # corrects any earlier distortion.
        if self._base_w and self._base_h:
            aspect = self._base_w / self._base_h
        else:
            aspect = w / max(1.0, h)

        # Floor and ceiling are both tied to the base size. Fall back to a small
        # absolute floor only if base isn't known yet (shouldn't happen in use).
        if self._base_w and self._base_h:
            min_w = self._base_w * self._min_scale
            max_w = self._base_w * self._max_scale
            min_h = self._base_h * self._min_scale
            max_h = self._base_h * self._max_scale
        else:
            min_w, max_w = 64.0, 1e9
            min_h, max_h = 64.0, 1e9

        # Never exceed the visible screen either.
        vis = NSScreen.mainScreen().visibleFrame()
        max_w = min(max_w, vis.size.width * 0.9)
        max_h = min(max_h, vis.size.height * 0.9)

        # Drive by width, derive height from aspect, then re-check the height caps.
        w = max(min_w, min(w, max_w))
        h = w / aspect
        if h > max_h:
            h = max_h
            w = h * aspect
        if h < min_h:
            h = min_h
            w = h * aspect
        return w, h

    def next_frame(self):
        """Advance one frame (ping-pong)."""
        if not self._frames:
            return
        self._frame_idx += self._direction
        if self._frame_idx >= len(self._frames):
            self._frame_idx = len(self._frames) - 2
            self._direction = -1
        elif self._frame_idx < 0:
            self._frame_idx = 1
            self._direction = 1
        if self._frame_idx < 0:
            self._frame_idx = 0
        self.setNeedsDisplay_(True)

    # -- Drawing --

    def drawRect_(self, rect):
        bounds = self.bounds()

        # Draw current animation frame
        if self._frames and 0 <= self._frame_idx < len(self._frames):
            frame = self._frames[self._frame_idx]
            frame.drawInRect_fromRect_operation_fraction_(
                bounds,
                NSMakeRect(0, 0, frame.size().width, frame.size().height),
                NSCompositingOperationSourceOver,
                1.0
            )

        # Draw close button
        self._draw_close_button(bounds)

    def _draw_close_button(self, bounds):
        cx = bounds.size.width - self._close_margin - self._close_radius
        cy = bounds.size.height - self._close_margin - self._close_radius
        r = self._close_radius

        if self._is_hovering_close:
            # Active: white circle + dark X
            NSColor.colorWithWhite_alpha_(1.0, 0.85).setFill()
            bg = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - r, cy - r, r * 2, r * 2)
            )
            bg.fill()

            NSColor.colorWithWhite_alpha_(0.2, 1.0).setStroke()
        else:
            # Inactive: faint circle
            NSColor.colorWithWhite_alpha_(1.0, 0.2).setFill()
            bg = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - r, cy - r, r * 2, r * 2)
            )
            bg.fill()

            NSColor.colorWithWhite_alpha_(0.5, 0.3).setStroke()

        # Draw X
        d = r * 0.4
        line = NSBezierPath.bezierPath()
        line.setLineWidth_(1.5)
        line.moveToPoint_(NSMakePoint(cx - d, cy - d))
        line.lineToPoint_(NSMakePoint(cx + d, cy + d))
        line.moveToPoint_(NSMakePoint(cx + d, cy - d))
        line.lineToPoint_(NSMakePoint(cx - d, cy + d))
        line.stroke()

    # -- Hit testing --

    def _close_btn_rect(self):
        bounds = self.bounds()
        cx = bounds.size.width - self._close_margin - self._close_radius
        cy = bounds.size.height - self._close_margin - self._close_radius
        r = self._close_radius
        return NSMakeRect(cx - r, cy - r, r * 2, r * 2)

    def _resize_handle_rect(self):
        bounds = self.bounds()
        s = self._resize_size
        return NSMakeRect(bounds.size.width - s, 0, s, s)

    def _point_in_rect(self, pt, rect):
        return (rect.origin.x <= pt.x <= rect.origin.x + rect.size.width and
                rect.origin.y <= pt.y <= rect.origin.y + rect.size.height)

    def _point_in_circle(self, pt, rect):
        cx = rect.origin.x + rect.size.width / 2
        cy = rect.origin.y + rect.size.height / 2
        r = rect.size.width / 2
        return (pt.x - cx) ** 2 + (pt.y - cy) ** 2 <= r * r

    # -- Mouse events --

    def mouseDown_(self, event):
        pt = self.convertPoint_fromView_(event.locationInWindow(), None)

        # Check close button
        if self._point_in_circle(pt, self._close_btn_rect()):
            self.window().close()
            return

        # Check resize handle (disabled when the size is locked)
        if not self._size_locked and self._point_in_rect(pt, self._resize_handle_rect()):
            self._is_dragging_resize = True
            self._resize_start = event.locationInWindow()
            self._resize_win_start = self.window().frame().size
            return

        # Start window drag (manual tracking for borderless window)
        self._is_dragging_window = True
        self._drag_start_screen = NSEvent.mouseLocation()
        win_frame = self.window().frame()
        self._drag_win_origin = NSMakePoint(win_frame.origin.x, win_frame.origin.y)

    def mouseDragged_(self, event):
        if self._is_dragging_window:
            screen = NSEvent.mouseLocation()
            dx = screen.x - self._drag_start_screen.x
            dy = screen.y - self._drag_start_screen.y
            new_x = self._drag_win_origin.x + dx
            new_y = self._drag_win_origin.y + dy
            # Clamp to visible screen so pet stays fully on screen
            from AppKit import NSScreen
            vis = NSScreen.mainScreen().visibleFrame()
            wf = self.window().frame()
            new_x = max(vis.origin.x, min(new_x, vis.origin.x + vis.size.width - wf.size.width))
            new_y = max(vis.origin.y, min(new_y, vis.origin.y + vis.size.height - wf.size.height))
            self.window().setFrameOrigin_(NSMakePoint(new_x, new_y))

        elif self._is_dragging_resize:
            current = event.locationInWindow()
            dx = current.x - self._resize_start.x
            dy = current.y - self._resize_start.y
            # Drive by the larger drag delta; _clamp_size enforces aspect + cap.
            if abs(dx) >= abs(dy):
                target = self._resize_win_start.width + dx
                new_w, new_h = self._clamp_size(target, target)
            else:
                target = self._resize_win_start.height + dy
                aspect = (self._base_w / self._base_h) if self._base_h else 1.0
                new_w, new_h = self._clamp_size(target * aspect, target)
            frame = self.window().frame()
            frame.size.width = new_w
            frame.size.height = new_h
            self.window().setFrame_display_animate_(frame, True, False)
            self.setNeedsDisplay_(True)

    def mouseUp_(self, event):
        if self._is_dragging_resize:
            self._save_scale()
        self._is_dragging_resize = False
        self._is_dragging_window = False

    def mouseEntered_(self, event):
        pt = self.convertPoint_fromView_(event.locationInWindow(), None)
        if self._point_in_circle(pt, self._close_btn_rect()):
            self._is_hovering_close = True
            self.setNeedsDisplay_(True)

    def mouseExited_(self, event):
        if self._is_hovering_close:
            self._is_hovering_close = False
            self.setNeedsDisplay_(True)

    def mouseMoved_(self, event):
        pt = self.convertPoint_fromView_(event.locationInWindow(), None)

        # Update close button hover state
        in_close = self._point_in_circle(pt, self._close_btn_rect())
        if in_close != self._is_hovering_close:
            self._is_hovering_close = in_close
            self.setNeedsDisplay_(True)

        # Update cursor
        if not self._size_locked and self._point_in_rect(pt, self._resize_handle_rect()):
            NSCursor.crosshairCursor().set()
        elif in_close:
            NSCursor.pointingHandCursor().set()
        else:
            NSCursor.arrowCursor().set()

    # -- Scroll wheel zoom --

    def scrollWheel_(self, event):
        """Mouse scroll wheel → resize window (disabled when size is locked)."""
        if self._size_locked:
            return
        delta = event.scrollingDeltaY()
        if event.hasPreciseScrollingDeltas():
            delta *= 10.0
        factor = 1.0 + delta * 0.01
        self._scale_window(factor)

    def magnifyWithEvent_(self, event):
        """Trackpad pinch gesture → resize window (disabled when size is locked)."""
        if self._size_locked:
            return
        factor = 1.0 + event.magnification()
        self._scale_window(factor)

    def _scale_window(self, factor):
        vis = NSScreen.mainScreen().visibleFrame()
        frame = self.window().frame()
        # Aspect-preserving + capped (fixes stretch-on-zoom and runaway size).
        new_w, new_h = self._clamp_size(frame.size.width * factor,
                                        frame.size.height * factor)
        # Keep center position
        cx = frame.origin.x + frame.size.width / 2
        cy = frame.origin.y + frame.size.height / 2
        new_x = max(vis.origin.x, min(cx - new_w / 2, vis.origin.x + vis.size.width - new_w))
        new_y = max(vis.origin.y, min(cy - new_h / 2, vis.origin.y + vis.size.height - new_h))
        new_frame = NSMakeRect(new_x, new_y, new_w, new_h)
        self.window().setFrame_display_animate_(new_frame, True, False)
        self.setNeedsDisplay_(True)
        self._save_scale()


# ---------------------------------------------------------------------------
# PetWindow — transparent borderless always-on-top
# ---------------------------------------------------------------------------

class PetWindow(NSWindow):
    """Borderless, transparent, always-on-top window."""

    def initWithContentRect_(self, rect):
        self = objc.super(PetWindow, self).initWithContentRect_styleMask_backing_defer_(
            rect,
            NSBorderlessWindowMask,
            2,  # NSBackingStoreBuffered
            False
        )
        if self is None:
            return None

        self.setOpaque_(False)
        self.setBackgroundColor_(NSColor.clearColor())
        self.setLevel_(NSFloatingWindowLevel)
        self.setMovableByWindowBackground_(False)
        self.setHasShadow_(False)
        self.setAcceptsMouseMovedEvents_(True)

        return self

    def canBecomeKeyWindow(self):
        return True

    def canBecomeMainWindow(self):
        return True


# ---------------------------------------------------------------------------
# Display entry point
# ---------------------------------------------------------------------------

def _load_frames(frame_paths):
    """Load frames from paths, supporting both RGBA PNG and JPEG pair formats.

    JPEG pairs: frame_NNNN.jpg (RGB) + frame_NNNN_a.jpg (alpha grayscale).
    Falls back to standard PNG loading if no alpha pair found.
    """
    if not frame_paths:
        return []

    # Detect JPEG pair format: check if alpha companion exists for first frame
    first = frame_paths[0]
    stem = os.path.splitext(first)[0]
    alpha_path = stem + "_a.jpg"
    if os.path.exists(alpha_path):
        frames = []
        for p in frame_paths:
            s = os.path.splitext(p)[0]
            rgb = Image.open(p).convert("RGB")
            a = Image.open(s + "_a.jpg").convert("L")
            # Merge RGB + alpha into RGBA
            rgba = rgb.convert("RGBA")
            rgba.putalpha(a)
            frames.append(rgba)
        return frames

    # Standard loading (PNG or other)
    return [Image.open(p).convert("RGBA") for p in frame_paths]


class DesktopPet:
    """Manages the pet window and animation loop."""

    def __init__(self, frame_paths, fps=24, window_size=None, alpha=1.0):
        self.fps = fps
        self.delay = 1.0 / fps

        # Screen backing scale (Retina) — frames are rendered at physical pixels
        # so they stay crisp; the NSImage keeps its logical (point) size.
        try:
            scale = max(1.0, float(NSScreen.mainScreen().backingScaleFactor()))
        except Exception:
            scale = 1.0

        # Fast path: load the packed display cache (cropped + resized frames),
        # skipping PNG decode + LANCZOS resize entirely.
        frames_dir = os.path.dirname(frame_paths[0]) if frame_paths else ""
        sig = _src_signature(frame_paths)
        nsimages = None
        if frames_dir and sig and window_size is None:
            cached = _load_display_cache(frames_dir, scale, sig)
            if cached is not None:
                nsimages, (tw, th) = cached
                print(f"  Loaded {len(nsimages)} frames from display cache "
                      f"({tw}x{th}pt @ {scale:g}x)")

        if nsimages is None:
            # Slow path: decode, crop to the common foreground box, resize to
            # Retina pixels, then pack to the cache for next time.
            raw = _load_frames(frame_paths)
            raw = _crop_to_foreground(raw, margin=30)
            print(f"  Cropped to foreground: {raw[0].size[0]}x{raw[0].size[1]}")

            cw, ch = raw[0].size
            # Window (point) size: normalize so the SUBJECT occupies a fixed
            # on-screen area (consistent pet size across clips/orientations).
            if window_size:
                fit = min(window_size[0] / cw, window_size[1] / ch)
            else:
                fg_area = _median_fg_area(raw)
                if fg_area > 0:
                    fit = math.sqrt(TARGET_FG_AREA_PT2 / fg_area)
                    fit = max(SUBJECT_SCALE_MIN, min(fit, SUBJECT_SCALE_MAX))
                else:
                    fit = min(1.0, 400.0 / max(cw, ch))  # fallback: fit longest side
            # Safety cap: never open larger than 90% of the visible screen.
            vis = NSScreen.mainScreen().visibleFrame()
            fit = min(fit, vis.size.width * 0.9 / cw, vis.size.height * 0.9 / ch)
            tw, th = max(1, int(round(cw * fit))), max(1, int(round(ch * fit)))

            # Master bitmap resolution is DECOUPLED from the window size: keep the
            # pet's native pixels (don't throw detail away — that was the old
            # 400pt-cap softness), capped at MAX_MASTER_PX on the long side to
            # bound memory, and never smaller than the window's physical size.
            # This keeps the pet crisp at base size AND when zoomed up.
            win_phys = max(tw, th) * scale
            master_long = max(min(max(cw, ch), MAX_MASTER_PX), win_phys)
            ms = master_long / max(cw, ch)
            pw, ph = max(1, int(round(cw * ms))), max(1, int(round(ch * ms)))
            print(f"  Loading {len(raw)} frames: window {tw}x{th}pt, master {pw}x{ph}px (@ {scale:g}x) ...")
            resized = [img.resize((pw, ph), Image.LANCZOS) for img in raw]
            nsimages = [_pil_to_nsimage(im, logical_size=(tw, th)) for im in resized]
            if frames_dir and sig and window_size is None:
                _write_display_cache(frames_dir, scale, sig, resized, tw, th)

        self._window_size = (tw, th)

        # Create window
        screen = NSScreen.mainScreen().frame()
        x = (screen.size.width - tw) / 2
        y = (screen.size.height - th) / 2

        rect = NSMakeRect(x, y, tw, th)
        self.window = PetWindow.alloc().initWithContentRect_(rect)

        # Create and set content view
        self.view = PetView.alloc().initWithFrame_(NSMakeRect(0, 0, tw, th))
        self.view.set_frames(nsimages)
        self.view.set_base_size(tw, th)
        # Restore this pet's saved scale (if any), re-centered. Base size is the
        # 100% ceiling, so a saved scale only ever shrinks the window.
        saved_scale = self.view.set_scale_store(frames_dir)
        if saved_scale is not None and abs(saved_scale - 1.0) > 1e-3:
            sw, sh = self.view._clamp_size(tw * saved_scale, th * saved_scale)
            nx = x + (tw - sw) / 2
            ny = y + (th - sh) / 2
            self.window.setFrame_display_(NSMakeRect(nx, ny, sw, sh), False)
        self.window.setContentView_(self.view)
        self.window.makeKeyAndOrderFront_(None)

        # Start animation timer
        self.timer = None

    def tick(self):
        self.view.next_frame()

    def run(self):
        print("  Desktop pet running. Drag to move, right-corner resize, top-right X to close.")

        # Schedule animation timer
        from Foundation import NSTimer as FT
        self.timer = FT.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self.delay, self, "tick", None, True
        )

        # Run the Cocoa event loop
        NSApp().run()


def display_frames(frame_paths, fps=24, window_size=None, alpha=1.0):
    """Display frame sequence as a looping desktop pet.

    Args:
        frame_paths: list of PNG file paths
        fps: playback framerate
        window_size: optional (width, height) tuple
        alpha: unused (per-pixel alpha from images)
    """
    if not frame_paths:
        print("  No frames to display.")
        return

    # Initialize NSApplication
    NSApplication.sharedApplication()

    pet = DesktopPet(frame_paths, fps=fps, window_size=window_size, alpha=alpha)
    pet.run()
