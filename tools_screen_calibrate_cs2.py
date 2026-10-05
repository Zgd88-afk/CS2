# Screen-crop calibration tool for running the BC agent on CS2.
#
# The model expects a 150x280 crop of the game view. On CSGO the crop was
# (1024x768 window, minus 135px top / 135px bottom / 100px each side). CS2's
# HUD/radar layout differs, so the crop must be re-tuned once per machine.
#
# Usage:
#   1. Start CS2 windowed at 1024x768 and join a match (so the HUD is visible)
#   2. python tools_screen_calibrate_cs2.py
#   3. Move the trackbars until the red rectangle frames the game view the way
#      the model should see it: crosshair centred, health/ammo HUD included,
#      no title bar, no dead borders
#   4. Press 's' to save cs2_crop_offsets.json (dm_run_agent_cs2.py reads it)
#   5. Press 'q' to quit
#
# NOT YET VALIDATED ON A LIVE CS2 INSTALL.

import argparse
import json
import os

import cv2
import numpy as np
import win32con
import win32gui
import win32ui

from config import csgo_game_res, csgo_img_dimension
from cs2_config import (find_cs2_window, load_crop_offsets, DEFAULT_CROP_OFFSETS,
                        OFFSET_FILE)

PREVIEW_WIDTH = 768


def capture_full_window(hwin, width, height):
    """Grab the whole window (title bar included) as a BGRA->BGR numpy image."""
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    signed_ints_array = bmp.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser(
        description='Calibrate the CS2 screen crop for the BC agent')
    ap.add_argument('--width', type=int, default=csgo_game_res[0],
                    help='game window width (default 1024)')
    ap.add_argument('--height', type=int, default=csgo_game_res[1],
                    help='game window height (default 768)')
    args = ap.parse_args()
    res = (args.width, args.height)

    hwnd, title = find_cs2_window()
    if hwnd is None:
        print('CS2 window not found - start the game (windowed) first.')
        return
    print('calibrating window: %r (hwnd=%s)' % (title, hwnd))

    defaults = load_crop_offsets()
    cv2.namedWindow('controls', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('top', 'controls', defaults.get('offset_top', 135), 300, nothing)
    cv2.createTrackbar('bottom', 'controls', defaults.get('offset_bottom', 135), 300, nothing)
    cv2.createTrackbar('sides', 'controls', defaults.get('offset_sides', 100), 300, nothing)
    cv2.createTrackbar('bar', 'controls', defaults.get('bar_height', 35), 100, nothing)

    print('\nadjust the sliders so the red rectangle frames the game view')
    print('  (crosshair centred, HUD visible, no title bar / dead borders)')
    print("press 's' to save, 'q' to quit\n")

    while True:
        img = capture_full_window(hwnd, res[0], res[1])

        off_top = cv2.getTrackbarPos('top', 'controls')
        off_bottom = cv2.getTrackbarPos('bottom', 'controls')
        off_sides = cv2.getTrackbarPos('sides', 'controls')
        bar = cv2.getTrackbarPos('bar', 'controls')

        x0, y0 = off_sides, bar + off_top
        x1, y1 = res[0] - off_sides, res[1] - off_bottom

        vis = img.copy()
        if x1 > x0 and y1 > y0:
            crop = img[y0:y1, x0:x1]
            # mark the crop region and centre lines
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        else:
            crop = None
            cv2.putText(vis, 'invalid crop (offsets too large)', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        scale = PREVIEW_WIDTH / img.shape[1]
        preview = cv2.resize(vis, (PREVIEW_WIDTH, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        cv2.putText(preview, 's: save  q: quit  |  crop %dx%d -> model %s'
                    % (max(x1 - x0, 0), max(y1 - y0, 0), tuple(csgo_img_dimension)),
                    (10, preview.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow('preview (full window, red = crop region)', preview)

        if crop is not None and crop.size > 0:
            model_view = cv2.resize(crop, csgo_img_dimension[::-1])
            big = cv2.resize(model_view, (csgo_img_dimension[1] * 2,
                                          csgo_img_dimension[0] * 2),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, 'model input %s' % (tuple(csgo_img_dimension),), (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow('model input (nearest x2)', big)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print('quit without saving')
            break
        if key == ord('s'):
            if crop is None:
                print('cannot save: crop region is invalid')
                continue
            payload = {
                'bar_height': int(bar),
                'offset_top': int(off_top),
                'offset_bottom': int(off_bottom),
                'offset_sides': int(off_sides),
                'game_resolution': [res[0], res[1]],
            }
            with open(OFFSET_FILE, 'w') as f:
                json.dump(payload, f, indent=2)
            print('saved %s: %s' % (OFFSET_FILE, payload))
            print('defaults in cs2_config.DEFAULT_CROP_OFFSETS were: %s' % DEFAULT_CROP_OFFSETS)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
