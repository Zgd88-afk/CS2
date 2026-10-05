# CS2 adaptation config for the Counter-Strike Behavioural Cloning project.
#
# The original scripts assume the CSGO window title and the CSGO HUD layout.
# CS2 (Source 2) changes both, so this module centralises the CS2-specific
# knobs:
#   * window discovery (fuzzy title match, also supports csgo_legacy for A/B tests)
#   * screen-crop offsets, overridable via cs2_crop_offsets.json produced by
#     tools_screen_calibrate_cs2.py (CS2 HUD sits at different pixels than CSGO)
#   * MOUSE_SCALE, compensating for CS2 forcing raw input ON while the model
#     was trained with raw input OFF (see docs/CS2迁移测试计划.md Phase 2)
#
# Used by: dm_run_agent_cs2.py, tools_screen_calibrate_cs2.py

import ctypes
import json
import os

import cv2
import numpy as np
import win32con
import win32gui
import win32ui

from config import csgo_game_res, csgo_img_dimension

# BitBlt on the CS2 window returns a stale, never-updating frame (Source 2 uses
# DXGI flip-model presentation). PrintWindow with PW_RENDERFULLCONTENT forces
# DWM to render fresh window content and is the reliable path on CS2; legacy
# CSGO windows (csgo_branch) work with either, so BitBlt stays as fallback.
PW_RENDERFULLCONTENT = 0x00000002

# ---------------------------------------------------------------- window title
# CS2 window is titled 'Counter-Strike 2'. The legacy CSGO branch is kept as a
# fallback so dm_run_agent_cs2.py also runs on csgo_legacy for A/B comparison.
CS2_TITLE_SUBSTRINGS = (
    'counter-strike 2',
    'counter-strike: global offensive',  # csgo_legacy window
)

# ---------------------------------------------------------------- crop offsets
# Defaults mirror screen_input.py's CSGO values; run the calibration tool to
# adapt them to CS2's HUD layout and your OS title-bar height.
DEFAULT_CROP_OFFSETS = {
    'bar_height': 35,        # window title-bar height
    'offset_top': 135,       # pixels of game view ignored at the top
    'offset_bottom': 135,    # ... at the bottom
    'offset_sides': 100,     # ... on each side
    'game_resolution': list(csgo_game_res),
}

OFFSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cs2_crop_offsets.json')

# ------------------------------------------------------------- mouse calibration
# deg-per-delta in CS2 vs the training environment (CSGO, raw input OFF,
# sensitivity 2.5). Measure with `dm_run_agent_cs2.py --calibrate` plus
# `cl_showpos 1`, then set MOUSE_SCALE = trained_deg / cs2_deg.
MOUSE_SCALE = 1.0

# ---------------------------------------------------------------- style matching
# Test-time visual domain-gap mitigation: recolour each captured CS2 frame so
# its per-channel histogram matches the CSGO training-data distribution.
# Reference distribution is built once with tools_model_check.py --build-ref.
STYLE_MATCH = False  # when True, grab_window_cs2 recolours frames automatically
STYLE_REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'cs2_style_ref.npz')
_style_ref_cache = None


def build_style_reference(data_npy, n_frames=200, out_path=None):
    """Compute per-channel reference histograms from CSGO training frames."""
    import warnings
    warnings.filterwarnings('ignore')
    data = np.load(data_npy, allow_pickle=True)
    idx = np.linspace(0, len(data) - 1, min(n_frames, len(data))).astype(int)
    hists = np.zeros((3, 256), dtype=np.float64)
    for i in idx:
        img = np.asarray(data[i][0])
        for c in range(3):
            hists[c] += np.bincount(img[:, :, c].ravel(), minlength=256)
    hists /= hists.sum(axis=1, keepdims=True)
    out_path = out_path or STYLE_REF_FILE
    np.savez_compressed(out_path, hist=hists)
    print('style reference saved: %s (%d frames from %s)'
          % (out_path, len(idx), os.path.basename(data_npy)))
    return hists


def _style_ref():
    global _style_ref_cache
    if _style_ref_cache is None:
        if not os.path.isfile(STYLE_REF_FILE):
            raise RuntimeError(
                'style reference missing - run: python tools_model_check.py '
                '--build-ref <csgo training .npy>')
        with np.load(STYLE_REF_FILE) as z:
            _style_ref_cache = z['hist']
    return _style_ref_cache


def match_style(img):
    """Recolour a BGR uint8 frame to match the CSGO training histogram."""
    ref = _style_ref()
    ref_cdf = np.cumsum(ref, axis=1)
    out = np.empty_like(img)
    for c in range(3):
        hist = np.bincount(img[:, :, c].ravel(), minlength=256).astype(np.float64)
        cdf = np.cumsum(hist)
        if cdf[-1] == 0:
            continue
        cdf /= cdf[-1]
        lut = np.clip(np.interp(cdf, ref_cdf[c], np.arange(256)), 0, 255).astype(np.uint8)
        out[:, :, c] = cv2.LUT(img[:, :, c], lut)
    return out


def load_crop_offsets():
    """Return crop offsets, preferring cs2_crop_offsets.json if it exists."""
    offsets = dict(DEFAULT_CROP_OFFSETS)
    if os.path.isfile(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, 'r') as f:
                offsets.update(json.load(f))
        except (OSError, ValueError) as e:
            print('could not read %s (%s), using defaults' % (OFFSET_FILE, e))
    return offsets


def find_cs2_window():
    """Find the CS2 (or legacy CSGO) window.

    Returns (hwnd, title) or (None, None). Fuzzy substring match so engine
    suffixes like ' - Direct3D 9' do not break discovery.
    """
    matches = []

    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            low = title.lower()
            for i, sub in enumerate(CS2_TITLE_SUBSTRINGS):
                if sub in low:
                    matches.append((i, hwnd, title))
                    break

    win32gui.EnumWindows(_enum, None)
    if not matches:
        return None, None
    # prefer CS2 over legacy if both are open
    matches.sort(key=lambda m: m[0])
    _, hwnd, title = matches[0]
    return hwnd, title


def grab_window_cs2(hwin, game_resolution=None, SHOW_IMAGE=False):
    """Capture the CS2 window and return the 150x280 BGR input image.

    Same pipeline as screen_input.grab_window (win32 BitBlt -> crop -> resize),
    but crop offsets come from load_crop_offsets() so they can be recalibrated
    for CS2 without touching the original scripts.
    """
    if game_resolution is None:
        game_resolution = csgo_game_res

    off = load_crop_offsets()
    bar_height = off['bar_height']
    offset_top = off['offset_top']
    offset_bottom = off['offset_bottom']
    offset_sides = off['offset_sides']
    if 'game_resolution' in off:
        game_resolution = tuple(off['game_resolution'])

    crop_w = game_resolution[0] - 2 * offset_sides
    crop_h = game_resolution[1] - offset_top - offset_bottom
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(
            'crop offsets leave no image (w=%d, h=%d) - rerun '
            'tools_screen_calibrate_cs2.py' % (crop_w, crop_h))

    l, t, r, b = win32gui.GetWindowRect(hwin)
    win_w, win_h = r - l, b - t
    if win32gui.IsIconic(hwin) or win_h < crop_h or win_w < crop_w:
        raise RuntimeError(
            'CS2 window not visible (minimized or closed? rect %dx%d)'
            % (win_w, win_h))

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, win_w, win_h)
    memdc.SelectObject(bmp)

    ok = ctypes.windll.user32.PrintWindow(hwin, memdc.GetSafeHdc(),
                                          PW_RENDERFULLCONTENT)
    if ok:
        # render whole window, then crop in numpy
        signed_ints_array = bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (win_h, win_w, 4)
        img = img[bar_height + offset_top:bar_height + offset_top + crop_h,
                  offset_sides:offset_sides + crop_w]
    else:
        # legacy fallback (BitBlt): crop at copy time via source origin
        bmp.DeleteObject()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, crop_w, crop_h)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (crop_w, crop_h), srcdc,
                     (offset_sides, bar_height + offset_top), win32con.SRCCOPY)
        signed_ints_array = bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (crop_h, crop_w, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_small = cv2.resize(img, csgo_img_dimension[::-1])
    if STYLE_MATCH:
        img_small = match_style(img_small)

    if SHOW_IMAGE:
        target_width = 800
        scale = target_width / img_small.shape[1]
        dim = (target_width, int(img_small.shape[0] * scale))
        resized = cv2.resize(img_small, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('grab_window_cs2 (q to quit)', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return img_small
