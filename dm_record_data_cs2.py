# CS2 finetune-data recorder v2 (Stage 2 of docs/CS2迁移测试计划.md).
#
# You play CS2 normally (offline dust2 deathmatch, AK47, sensitivity 2.5);
# this script captures (screen frame + your real key/mouse input) pairs at
# 16 fps - the same pairing the original CSGO project trained on.
#
# v2: mouse deltas now come from a WM_INPUT raw-input listener (OS-level,
# works while CS2 keeps the cursor locked) - the v1 GetCursorPos approach
# captured nothing because CS2 raw input never moves the Windows cursor.
# Demo recording is NOT needed (demoparser2 cannot parse POV demos).
#
# Per frame it stores:
#   imgs  (1000,150,280,3) uint8  - what the model will see
#   keys  (1000,11) uint8         - w a s d space ctrl shift 1 2 3 r
#   clicks(1000,2) uint8          - left, right mouse button
#   mouse (1000,2) float32        - raw-input mouse deltas summed per frame
#   ts    (1000,)  float64        - unix time per frame
#
# Files: <folder>/cs2_ft_<n>.npz every 1000 frames (~62 s); numbering
# auto-continues across sessions.
#
# Usage:
#   python dm_record_data_cs2.py --pilot          # 20 s capture test, no save
#   python dm_record_data_cs2.py                  # record until Q pressed
#   python dm_record_data_cs2.py --minutes 60
#
# NOTE: runs with STYLE_MATCH OFF - finetune frames must look exactly like
# deployment-time frames.

import os
import sys
import time
import argparse
import ctypes
import threading
from ctypes import wintypes

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

# ------------------------------------------------------------------ raw input
WM_INPUT = 0x00FF
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPE_MOUSE = 0x00
MOUSE_MOVE_ABSOLUTE = 0x0001
HWND_MESSAGE = -3  # message-only window parent (missing from pywin32)


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [('usUsagePage', ctypes.c_ushort),
                ('usUsage', ctypes.c_ushort),
                ('dwFlags', ctypes.c_ulong),
                ('hwndTarget', ctypes.c_void_p)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [('dwType', ctypes.c_ulong),
                ('dwSize', ctypes.c_ulong),
                ('hDevice', ctypes.c_void_p),
                ('wParam', ctypes.c_size_t)]


class RAWMOUSE(ctypes.Structure):
    # NOTE: empirically this machine's Win64 RAWMOUSE is 24 bytes with an
    # extra always-zero ULONG at offset 8 (docs say 20 bytes without it).
    # Without the reserved field, lLastX reads 0 and lLastY reads the X
    # deltas (verified by horizontal-only movement test, 2026-10-05).
    _fields_ = [('usFlags', ctypes.c_ushort),
                ('ulRawButtons', ctypes.c_ulong),
                ('reserved32', ctypes.c_ulong),
                ('lLastX', ctypes.c_long),
                ('lLastY', ctypes.c_long),
                ('ulExtraInformation', ctypes.c_ulong)]


class RAWINPUT(ctypes.Structure):
    class _Data(ctypes.Union):
        _fields_ = [('mouse', RAWMOUSE),
                    ('pad', ctypes.c_byte * 40)]
    _fields_ = [('header', RAWINPUTHEADER),
                ('data', _Data)]


_user32 = ctypes.windll.user32


class RawMouseListener(threading.Thread):
    """Daemon thread receiving OS-level relative mouse deltas via WM_INPUT.

    Works regardless of which window has focus and of CS2 hiding the cursor.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._events = []
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None

    def run(self):
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._on_input(lparam)
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        wc = win32gui.WNDCLASS()
        wc.lpszClassName = 'cs2_bc_rawinput_sink'
        wc.lpfnWndProc = wndproc
        wc.hInstance = win32api.GetModuleHandle(None)
        win32gui.RegisterClass(wc)
        hwnd = win32gui.CreateWindowEx(0, wc.lpszClassName, 'raw_input_sink',
                                       0, 0, 0, 0, 0, HWND_MESSAGE,
                                       None, None, None)

        rid = RAWINPUTDEVICE(1, 2, RIDEV_INPUTSINK, hwnd)  # generic mouse
        if not _user32.RegisterRawInputDevices(ctypes.byref(rid), 1,
                                               ctypes.sizeof(RAWINPUTDEVICE)):
            self.error = 'RegisterRawInputDevices failed (err %d)' % ctypes.GetLastError()
            return
        self.ready.set()
        win32gui.PumpMessages()

    def _on_input(self, lparam):
        buf = RAWINPUT()
        size = wintypes.UINT(ctypes.sizeof(RAWINPUT))
        ret = _user32.GetRawInputData(wintypes.HANDLE(lparam), RID_INPUT,
                                      ctypes.byref(buf), ctypes.byref(size),
                                      ctypes.sizeof(RAWINPUTHEADER))
        if ret == -1 or buf.header.dwType != RIM_TYPE_MOUSE:
            return
        m = buf.data.mouse
        if m.usFlags & MOUSE_MOVE_ABSOLUTE:
            return  # injected/absolute devices are not the player's mouse
        with self._lock:
            self._events.append((time.time(), m.lLastX, m.lLastY))

    def consume(self):
        """Return (dx, dy, n_events) summed since the previous call."""
        with self._lock:
            evs, self._events = self._events, []
        dx = float(sum(e[1] for e in evs))
        dy = float(sum(e[2] for e in evs))
        return dx, dy, len(evs)


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


def save_file(folder, file_num, imgs, keys, clicks, mouse, ts):
    out = os.path.join(folder, 'cs2_ft_%d.npz' % file_num)
    np.savez(out, imgs=np.stack(imgs), keys=np.stack(keys),
             clicks=np.stack(clicks), mouse=np.stack(mouse), ts=np.array(ts))
    return out


def main():
    ap = argparse.ArgumentParser(description='Record CS2 finetune data')
    ap.add_argument('--folder', default=r'D:\csgo_data\cs2_ft')
    ap.add_argument('--minutes', type=float, default=0,
                    help='auto-stop after N minutes (0 = until Q)')
    ap.add_argument('--pilot', action='store_true',
                    help='20 s capture test, prints mouse stats, saves nothing')
    args = ap.parse_args()

    listener = RawMouseListener()
    listener.start()
    if not listener.ready.wait(5):
        print('FATAL: raw-input listener failed:', listener.error or 'timeout')
        sys.exit(1)
    print('raw-input mouse listener ready.')

    hwnd, title = find_cs2_window()
    if hwnd is None:
        print('CS2 window not found - start the game first '
              '(not fatal for --pilot).')

    if args.pilot:
        print('PILOT: move the mouse and press keys for 20 seconds...')
        evs, frames = [], 0
        t0 = time.time()
        while time.time() - t0 < 20:
            loop_start = time.time()
            evs.append(listener.consume())
            frames += 1
            while time.time() < loop_start + 1.0 / loop_fps:
                time.sleep(0.001)
        arr = np.array(evs)
        moved = (np.abs(arr[:, 0]) + np.abs(arr[:, 1])) > 0
        print('frames: %d | mouse events total: %d (%.1f/frame)'
              % (frames, arr[:, 2].sum(), arr[:, 2].mean()))
        print('frames with movement: %.0f%% | |dx| mean %.1f p95 %.0f | '
              '|dy| mean %.1f p95 %.0f'
              % (100 * moved.mean(), np.abs(arr[:, 0]).mean(),
                 np.percentile(np.abs(arr[:, 0]), 95),
                 np.abs(arr[:, 1]).mean(), np.percentile(np.abs(arr[:, 1]), 95)))
        print('VERDICT:', 'mouse capture OK' if moved.mean() > 0.3 and
              arr[:, 2].sum() > frames else 'mouse capture NOT working')
        return

    if hwnd is None:
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
    print('play normally - walk, aim, fight. press Q to stop.\n')

    buf_imgs, buf_keys, buf_clicks, buf_mouse, buf_ts = [], [], [], [], []
    n_loops = 0
    t_start = time.time()
    listener.consume()  # drop anything accumulated during setup

    while True:
        loop_start = time.time()
        n_loops += 1

        keys, pressed, clicks = read_input()
        dx, dy, n_ev = listener.consume()

        try:
            img = grab_window_cs2(hwnd)
        except RuntimeError as e:
            print('\n%s - waiting for the CS2 window...' % e)
            time.sleep(2)
            continue

        buf_imgs.append(np.asarray(img, dtype=np.uint8))
        buf_keys.append(keys)
        buf_clicks.append(clicks)
        buf_mouse.append(np.array([dx, dy], dtype=np.float32))
        buf_ts.append(time.time())

        if len(buf_imgs) >= 1000:
            out = save_file(args.folder, file_num, buf_imgs, buf_keys,
                            buf_clicks, buf_mouse, buf_ts)
            print('\nsaved %s (total %.1f min)' % (out, (time.time() - t_start) / 60))
            buf_imgs, buf_keys, buf_clicks, buf_mouse, buf_ts = [], [], [], [], []
            file_num += 1

        if n_loops % 100 == 0:
            print('recording... frame %5d | %.1f min | keys %-12s | click %d | '
                  'mouse ev/frame %.0f'
                  % (n_loops, (time.time() - t_start) / 60,
                     ','.join(pressed) or '-', clicks[0], n_ev), end='\r')

        if win32api.GetAsyncKeyState(VK_Q) & 0x8000:
            print('\nQ pressed - stopping.')
            break
        if args.minutes and (time.time() - t_start) >= args.minutes * 60:
            print('\n--minutes reached - stopping.')
            break

        while time.time() < loop_start + 1.0 / loop_fps:
            time.sleep(0.001)

    if buf_imgs:
        out = save_file(args.folder, file_num, buf_imgs, buf_keys,
                        buf_clicks, buf_mouse, buf_ts)
        print('saved partial %s (%d frames)' % (out, len(buf_imgs)))
    print('done. total time: %.1f min' % ((time.time() - t_start) / 60))


if __name__ == '__main__':
    main()
