# CS2 finetune-data recorder (Stage 2 of docs/CS2迁移测试计划.md).
#
# You play CS2 normally (offline dust2 deathmatch, AK47, sensitivity 2.5);
# this script captures (screen frame + your real key/mouse state) pairs at
# 16 fps - the same pairing the original CSGO project trained on.
#
# Per frame it stores:
#   imgs     (1000,150,280,3) uint8  - what the model will see
#   keys     (1000,11) uint8         - w a s d space ctrl shift 1 2 3 r
#   clicks   (1000,2)  uint8         - left, right mouse button
#   mouse_raw(1000,2) float32       - GetCursorPos deltas (may be unreliable
#                                      in raw-input mode; authoritative mouse
#                                      labels come from demo parsing later)
#   ts       (1000,)  float64       - unix time per frame (for demo alignment)
#
# Files: <folder>/cs2_ft_<n>.npz every 1000 frames (~1 minute). Numbering
# auto-continues so you can record across multiple sessions.
#
# Before starting (in game):
#   1. offline dust2 deathmatch with bots, AK47 equipped
#   2. console:  record cs2_ft_1     (demo gives precise action labels later;
#      stop each session with  stop  and start the next number next time)
#
# Usage:
#   python dm_record_data_cs2.py                # record until Q pressed
#   python dm_record_data_cs2.py --minutes 60   # auto-stop after 60 minutes
#
# NOTE: runs with STYLE_MATCH OFF - finetune frames must look exactly like
# deployment-time frames.

import os
import sys
import time
import argparse

import numpy as np
import win32api
import win32gui

from config import loop_fps, get_highest_num
from cs2_config import find_cs2_window, grab_window_cs2, STYLE_MATCH

if STYLE_MATCH:
    raise RuntimeError('STYLE_MATCH must be False when recording training data')

KEY_ORDER = ['w', 'a', 's', 'd', 'space', 'ctrl', 'shift', '1', '2', '3', 'r']
KEY_VK = {'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44, 'space': 0x20,
          'ctrl': 0x11, 'shift': 0x10, '1': 0x31, '2': 0x32, '3': 0x33,
          'r': 0x52}
VK_Q, VK_LBUTTON, VK_RBUTTON = 0x51, 0x01, 0x02


def read_input():
    keys = np.zeros(len(KEY_ORDER), dtype=np.uint8)
    pressed = []
    for i, k in enumerate(KEY_ORDER):
        if win32api.GetAsyncKeyState(KEY_VK[k]) & 0x8000:
            keys[i] = 1
            pressed.append(k)
    clicks = np.array([1 if win32api.GetAsyncKeyState(VK_LBUTTON) & 0x8000 else 0,
                       1 if win32api.GetAsyncKeyState(VK_RBUTTON) & 0x8000 else 0],
                      dtype=np.uint8)
    return keys, pressed, clicks


def read_cursor():
    pt = win32gui.GetCursorPos() if hasattr(win32gui, 'GetCursorPos') else \
        win32api.GetCursorPos()
    return float(pt[0]), float(pt[1])


def main():
    ap = argparse.ArgumentParser(description='Record CS2 finetune data')
    ap.add_argument('--folder', default=r'D:\csgo_data\cs2_ft')
    ap.add_argument('--minutes', type=float, default=0,
                    help='auto-stop after N minutes (0 = until Q)')
    args = ap.parse_args()

    hwnd, title = find_cs2_window()
    if hwnd is None:
        print('CS2 window not found - start the game first.')
        return
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    if win32gui.GetForegroundWindow() != hwnd:
        print('CS2 window is not in the foreground - click on the game once '
          'and start again.')
        return

    os.makedirs(args.folder, exist_ok=True)
    file_num = get_highest_num('cs2_ft_', args.folder) + 1

    print('recording window: %r (hwnd=%s)' % (title, hwnd))
    print('output: %s\\cs2_ft_<n>.npz (1000 frames each, ~62 s)' % args.folder)
    print('remember the demo: console  record cs2_ft_1  (stop with  stop)')
    print('play normally - walk, aim, fight. press Q to stop.\n')

    buf_imgs, buf_keys, buf_clicks, buf_mouse, buf_ts = [], [], [], [], []
    n_loops = 0
    t_start = time.time()
    last_pos = read_cursor()

    while True:
        loop_start = time.time()
        n_loops += 1

        keys, pressed, clicks = read_input()
        pos = read_cursor()
        mouse_raw = (pos[0] - last_pos[0], pos[1] - last_pos[1])
        last_pos = pos

        try:
            img = grab_window_cs2(hwnd)
        except RuntimeError as e:
            print('\n%s - waiting for the CS2 window...' % e)
            time.sleep(2)
            continue

        buf_imgs.append(np.asarray(img, dtype=np.uint8))
        buf_keys.append(keys)
        buf_clicks.append(clicks)
        buf_mouse.append(np.array(mouse_raw, dtype=np.float32))
        buf_ts.append(time.time())

        if len(buf_imgs) >= 1000:
            out = os.path.join(args.folder, 'cs2_ft_%d.npz' % file_num)
            np.savez(out,
                     imgs=np.stack(buf_imgs),
                     keys=np.stack(buf_keys),
                     clicks=np.stack(buf_clicks),
                     mouse_raw=np.stack(buf_mouse),
                     ts=np.array(buf_ts))
            print('\nsaved %s (total %.1f min)' % (out, (time.time() - t_start) / 60))
            buf_imgs, buf_keys, buf_clicks, buf_mouse, buf_ts = [], [], [], [], []
            file_num += 1

        if n_loops % 100 == 0:
            print('recording... frame %5d | %.1f min | keys %s | click %d'
                  % (n_loops, (time.time() - t_start) / 60,
                     ','.join(pressed) or '-', clicks[0]), end='\r')

        if win32api.GetAsyncKeyState(VK_Q) & 0x8000:
            print('\nQ pressed - stopping.')
            break
        if args.minutes and (time.time() - t_start) >= args.minutes * 60:
            print('\n--minutes reached - stopping.')
            break

        while time.time() < loop_start + 1.0 / loop_fps:
            time.sleep(0.001)

    if buf_imgs:
        out = os.path.join(args.folder, 'cs2_ft_%d.npz' % file_num)
        np.savez(out,
                 imgs=np.stack(buf_imgs),
                 keys=np.stack(buf_keys),
                 clicks=np.stack(buf_clicks),
                 mouse_raw=np.stack(buf_mouse),
                 ts=np.array(buf_ts))
        print('saved partial %s (%d frames)' % (out, len(buf_imgs)))
    print('done. total time: %.1f min' % ((time.time() - t_start) / 60))


if __name__ == '__main__':
    main()
