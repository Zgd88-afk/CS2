# CS2 adaptation of dm_run_agent.py (Counter-Strike Behavioural Cloning).
#
# Changes vs the original CSGO script (see docs/CS2迁移测试计划.md):
#   1. Window discovery: fuzzy-matches 'Counter-Strike 2' (also finds the
#      csgo_legacy window, so the same script works for A/B comparison runs).
#   2. Screen grabbing: cs2_config.grab_window_cs2, whose crop offsets come
#      from cs2_crop_offsets.json (produced by tools_screen_calibrate_cs2.py).
#      CS2's HUD/radar sit at different pixels than CSGO - recalibrate first.
#   3. MOUSE_SCALE (cs2_config): CS2 forces raw input ON while the model was
#      trained with raw input OFF, so the deg-per-delta mapping differs.
#      Calibrate with --calibrate and set the factor in cs2_config.
#   4. GSI reads are defensive (CS2 dropped some GSI payloads). GSI is OFF by
#      default; the primary scoring method is demo recording + tools_demo_stats.py.
#   5. CLI modes for the staged test plan:
#        --check-window  find and print the CS2 window
#        --smoke         hardcoded action pipeline test (no model needed)
#        --fps-test      measure screen-grab latency
#        --calibrate     constant mouse sweeps for deg/frame calibration
#        --demo          run agent with IS_DEMO vision overlay
#        --gsi           run agent with GSI stats (if appendix B validated)
#
# NOT YET VALIDATED ON A LIVE CS2 INSTALL - run the Phase 1 smoke checks first.
#
# If the game window refuses to react to injected keys (focus issue reported by
# some CSGO users), try adding at the very top:
#   import win32com.client
#   shell = win32com.client.Dispatch("WScript.Shell")
#   shell.SendKeys('%')

import os
import time
import pickle
import argparse

import mss
import cv2
import win32api as wapi  # noqa: F401  (kept for parity with original imports)
import win32gui

import numpy as np

from key_input import key_check, mouse_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import hold_left_click, release_left_click
from key_output import hold_right_click, release_right_click
from key_output import w_char, s_char, a_char, d_char, n_char  # noqa: F401
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
from key_output import m_char, u_char, under_char, g_char, esc_char
from key_output import i_char, v_char, o_char, k_char, seven_char

from config import (loop_fps, input_shape_lstm_pred, csgo_img_dimension,
                    aux_input_length, n_keys, mouse_x_possibles, mouse_y_possibles,
                    mouse_x_lim, mouse_y_lim, onehot_to_actions, actions_to_onehot,
                    tp_load_model, wait_for_loop_end)
from cs2_config import (find_cs2_window, grab_window_cs2, MOUSE_SCALE,
                        load_crop_offsets)

# set by run_agent() when --gsi is on; read by gsi_get()
server = None


# --------------------------------------------------------------------- helpers
def parse_args():
    ap = argparse.ArgumentParser(
        description='Run the BC agent on CS2 (see docs/CS2迁移测试计划.md)')
    ap.add_argument('--check-window', action='store_true',
                    help='find the CS2 window and exit')
    ap.add_argument('--smoke', action='store_true',
                    help='hardcoded action pipeline test, no model needed')
    ap.add_argument('--fps-test', action='store_true',
                    help='measure screen-grab latency (1000 frames)')
    ap.add_argument('--calibrate', action='store_true',
                    help='constant mouse sweeps for deg/frame calibration (use with cl_showpos 1)')
    ap.add_argument('--demo', action='store_true',
                    help='show agent vision overlay while running')
    ap.add_argument('--gsi', action='store_true',
                    help='enable GSI stats (validate fields first, appendix B)')
    ap.add_argument('--model', default='ak47_sub_55k_drop_d4_dmexpert_28',
                    help='model name (stateful) inside ./model/')
    ap.add_argument('--minutes', type=float, default=10,
                    help='minutes to run per iteration')
    ap.add_argument('--monitor', type=int, default=1,
                    help='mss monitor index for mouse mapping (1=primary)')
    return ap.parse_args()


def get_monitor_size(monitor_index=1):
    sct = mss.MSS() if hasattr(mss, 'MSS') else mss.mss()
    mon = sct.monitors[monitor_index]
    return mon["width"], mon["height"]


def grab_safe(hwin):
    """Grab a frame, waiting instead of crashing if the window disappears
    (minimized/accidentally closed) - resumes automatically when it's back."""
    while True:
        try:
            return grab_window_cs2(hwin)
        except RuntimeError as e:
            print('\n%s - waiting for the CS2 window...' % e)
            time.sleep(2)


def ensure_foreground(hwin):
    """Return True if the game window has keyboard/mouse focus.

    Windows blocks SetForegroundWindow from background processes; injecting
    while another window is focused would type into it (e.g. the editor).
    """
    win32gui.SetForegroundWindow(hwin)
    time.sleep(0.5)
    if win32gui.GetForegroundWindow() != hwin:
        print('REFUSING to inject: the CS2 window does NOT have focus '
              '(input would go to another window).')
        print('Click ONCE on the game window to bring it to the foreground, '
              'then run again.')
        return False
    return True


def gsi_get(path, default=None):
    # defensively read a dot-path like 'player.match_stats.kills' from the
    # latest GSI payload - CS2 dropped some fields the CSGO version provided
    try:
        node = server.data_all or {}
        for part in path.split('.'):
            node = node[part]
        return node
    except Exception:
        return default


def mp_restartgame():
    # types 'mp_restartgame 1' then 'give weapon_ak47' into the console
    # (commands verified to still work on CS2 local/offline servers).
    # NOTE: no trailing Esc presses - on CS2 they open the pause menu and
    # detach subsequent injected input from the game (found in live testing).
    for c in [cons_char, m_char, p_char, under_char, r_char, e_char, s_char,
              t_char, a_char, r_char, t_char, g_char, a_char, m_char, e_char,
              space_char, one_char, ret_char, cons_char]:
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)

    time.sleep(3)
    # give weapon_ak47
    for c in [cons_char, g_char, i_char, v_char, e_char, space_char,
              w_char, e_char, a_char, p_char, o_char, n_char, under_char,
              a_char, k_char, four_char, seven_char, ret_char, cons_char]:
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)
    return


def pause_game():
    for c in [cons_char, p_char, a_char, u_char, s_char, e_char, ret_char, cons_char]:
        time.sleep(0.1)
        HoldKey(c)
        ReleaseKey(c)
    return


# -------------------------------------------------------------- test modes
def mode_check_window():
    off = load_crop_offsets()
    hwnd, title = find_cs2_window()
    if hwnd is None:
        print('CS2 window NOT found. Is the game running windowed?')
        return
    print('found window: hwnd=%s  title=%r' % (hwnd, title))
    res = tuple(off['game_resolution'])
    w = res[0] - 2 * off['offset_sides']
    h = res[1] - off['offset_top'] - off['offset_bottom']
    offsets_from_json = os.path.isfile(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cs2_crop_offsets.json'))
    print('crop offsets from: %s' % ('cs2_crop_offsets.json' if offsets_from_json
                                     else 'defaults (run tools_screen_calibrate_cs2.py)'))
    print('crop region -> %dx%d, model input -> %s' % (w, h, tuple(csgo_img_dimension)))


def mode_fps_test(hwin, n_frames=1000):
    print('grabbing %d frames ...' % n_frames)
    times = []
    for i in range(n_frames):
        t0 = time.time()
        grab_window_cs2(hwin)
        times.append((time.time() - t0) * 1000)
        if i < 5:
            time.sleep(0.05)  # let the first DC allocations settle
    times = np.array(times[10:])  # drop warm-up frames
    print('grab latency ms: mean %.1f  p50 %.1f  p95 %.1f  max %.1f'
          % (times.mean(), np.percentile(times, 50),
             np.percentile(times, 95), times.max()))
    print('per-frame budget at %d fps is %.1f ms; grab should stay well under ~40 ms'
          % (loop_fps, 1000.0 / loop_fps))
    if np.percentile(times, 95) > 40:
        print('WARNING: too slow. Check: windowed 1024x768, lowest graphics, '
              'nothing else using the GPU.')


def mode_smoke(hwin, Wd, Hd, mid_x, mid_y):
    # hardcoded actions to validate the injection pipeline without the model.
    # if the mouse sweep does NOT turn the view, injected mouse input is not
    # reaching the game (focus/admin/AV) - do not proceed to model runs.
    if not ensure_foreground(hwin):
        return

    print('SMOKE 1/5  mouse sweep right (mouse_x=60), 5 s - view should turn right')
    t_end = time.time() + 5
    while time.time() < t_end:
        loop_start = time.time()
        set_pos(mid_x + 60 / 2 * MOUSE_SCALE, mid_y, Wd, Hd)
        time.sleep(0.5 / loop_fps)
        set_pos(mid_x + 60 / 2 * MOUSE_SCALE, mid_y, Wd, Hd)
        while time.time() < loop_start + 1 / loop_fps:
            time.sleep(0.001)

    print('SMOKE 2/5  W forward, 5 s - should walk forward')
    t_end = time.time() + 5
    HoldKey(w_char)
    while time.time() < t_end:
        time.sleep(0.05)
    ReleaseKey(w_char)

    print('SMOKE 3/5  left click fire, 3 s - should shoot')
    t_end = time.time() + 3
    hold_left_click()
    while time.time() < t_end:
        time.sleep(0.05)
    release_left_click()

    print('SMOKE 4/5  space jump x3')
    for _ in range(3):
        HoldKey(space_char)
        time.sleep(0.1)
        ReleaseKey(space_char)
        time.sleep(0.6)

    print('SMOKE 5/5  R reload')
    HoldKey(r_char)
    time.sleep(0.1)
    ReleaseKey(r_char)

    print('smoke test done. If any action had no effect, fix that before Phase 2.')


def mode_calibrate(hwin, Wd, Hd, mid_x, mid_y):
    # measure deg-per-frame for fixed mouse_x values using cl_showpos 1.
    # training reference (raw-input math): deg/frame = x * m_yaw(0.022) * sens(2.5)
    # i.e. x=30 -> ~1.65 deg/frame. CSGO raw-off training values may differ, so
    # the practical target is: make CS2 turns match what the same model output
    # produced on CSGO (compare vs a legacy run if you do Phase 3 E2).
    if not ensure_foreground(hwin):
        return

    print('make sure the console command  cl_showpos 1  is active - you will')
    print('read the yaw angle (ang: pitch yaw roll - 2nd number) from the screen.')
    print('phases run in fixed order: mouse_x = 10, 30, 60. Watch for prompts.\n')
    for c in [10.0, 30.0, 60.0]:
        print('[%s] PHASE mouse_x=%g : note your START yaw NOW - sweep in 10 s'
              % (time.strftime('%H:%M:%S'), c))
        time.sleep(10)
        print('[%s] sweeping mouse_x=%g for 5 s ...'
              % (time.strftime('%H:%M:%S'), c))
        t_start = time.time()
        k = 0
        while time.time() - t_start < 5.0:
            loop_start = time.time()
            set_pos(mid_x + c / 2 * MOUSE_SCALE, mid_y, Wd, Hd)
            time.sleep(0.5 / loop_fps)
            set_pos(mid_x + c / 2 * MOUSE_SCALE, mid_y, Wd, Hd)
            while time.time() < loop_start + 1 / loop_fps:
                time.sleep(0.001)
            k += 1
        print('[%s] phase done (%d frames) - note your END yaw NOW.  '
              'deg/frame = (end-start)/%d, reference ~%.3f'
              % (time.strftime('%H:%M:%S'), k, k, c * 0.022 * 2.5))
        time.sleep(8)
    print('all sweeps finished - report the six yaw readings (start/end per phase).')
    print('if yaw wrapped (e.g. 170 -> -170) just report the raw numbers.')


def _demo_overlay(img_small, mouse_x_smooth, mouse_y_smooth, keys_pressed,
                  Lclicks, clicks_pred, n_loops):
    """IS_DEMO vision overlay (ported from dm_run_agent.py).

    Returns True if the user asked to quit during the overlay.
    """
    img_show = np.clip(img_small, 0, 255).astype('uint8')

    target_width = 800
    scale = target_width / img_show.shape[1]
    dim = (target_width, int(img_show.shape[0] * scale))
    resized = cv2.resize(img_show, dim, interpolation=cv2.INTER_AREA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    lineType = 2
    fontColor_1 = (20, 255, 0)
    fontColor_2 = (0, 0, 255)
    fontColor_3 = (200, 200, 100)

    text_show_1 = 'mouse_x ' + str(int(mouse_x_smooth)) + \
        (5 - len(str(int(mouse_x_smooth)))) * ' '
    text_show_2 = 'fire ' + str(round(float(clicks_pred[0]), 3))
    text_show_3 = 'keys ' + ' '.join(k for k in keys_pressed
                                     if k not in ['shift', '1', '2', '3', 'ctrl'])

    # arrow showing mouse movement (origin = screen centre of the model input)
    (x1, y1) = (int(target_width / 2),
                int(target_width / csgo_img_dimension[1] * csgo_img_dimension[0] / 2))
    (x2, y2) = (x1 + int(mouse_x_smooth / 2), y1 + int(mouse_y_smooth / 2))
    if Lclicks > 0:
        cv2.arrowedLine(resized, (x1, y1), (x2, y2), (50, 0, 255), thickness=3)
    else:
        cv2.arrowedLine(resized, (x1, y1), (x2, y2), (20, 200, 10), thickness=3)

    # bar showing firing probability
    cv2.rectangle(resized, (220, 70), (int(clicks_pred[0] * 100) + 220, 90),
                  fontColor_2, -1)
    cv2.rectangle(resized, (220, 70), (320, 90), fontColor_2, 2)

    # boxes showing key pushes
    for (cx1, cy1, cx2, cy2, key) in [(70, 170, 96, 196, 'w'),
                                      (40, 200, 66, 226, 'a'),
                                      (70, 200, 96, 226, 's'),
                                      (100, 200, 126, 226, 'd')]:
        if key in keys_pressed:
            cv2.rectangle(resized, (cx1, cy1), (cx2, cy2), (200, 200, 100), 4)
        else:
            cv2.rectangle(resized, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)

    cv2.putText(resized, text_show_1, (50, 50), font, fontScale, fontColor_1, lineType)
    cv2.putText(resized, text_show_2, (50, 90), font, fontScale, fontColor_2, lineType)
    cv2.putText(resized, text_show_3, (50, 130), font, fontScale, fontColor_3, lineType)

    cv2.imshow('resized', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return True
    keys_pressed_tp = key_check()
    if 'Q' in keys_pressed_tp:
        cv2.destroyAllWindows()
        return True
    if n_loops == 1:
        print('\npausing to align windows (15 s)')
        time.sleep(15)
    return False


# --------------------------------------------------------------- agent mode
def run_agent(args, hwin):
    # tensorflow import happens here so the lightweight test modes start fast;
    # tp_load_model (config.py) loads keras lazily as well
    import tensorflow as tf
    # allocate VRAM on demand instead of grabbing all of it - the game needs
    # VRAM too, and a second TF process must still be able to run for diagnostics
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
    print('tensorflow %s' % tf.__version__)

    global server
    if args.gsi:
        from meta_utils import server as gsi_server
        server = gsi_server
        print('GSI enabled - expecting pushes on localhost:3000')

    Wd, Hd = get_monitor_size(args.monitor)
    print('monitor %d: %dx%d' % (args.monitor, Wd, Hd))

    time.sleep(0.2)
    print('capturing mouse position as centre ...')
    mouse_x_mid, mouse_y_mid = mouse_check()

    mins_per_iter = args.minutes
    model_name = args.model
    model_save_dir = os.path.join(os.getcwd(), 'model')
    print('model: %s (from %s)' % (model_name, model_save_dir))

    # which actions the agent may control (same defaults as dm_run_agent.py)
    IS_CLICKS = 1
    IS_RCLICK = 0
    IS_MOUSEMOVE = 1
    IS_WASD = 1
    IS_JUMP = 1
    IS_RELOAD = 1
    IS_KEYS = 0

    IS_SPLIT_MOUSE = True
    IS_PROBABILISTIC_ACTIONS = True
    N_FILES_RESTART = 500  # noqa: F841  (kept for parity with original)
    SAVE_TRAIN_DATA = False
    IS_DEMO = args.demo
    IS_GSI = args.gsi

    pickle_reward_path = os.path.join(os.getcwd(), 'rewards_.p')
    try:
        pickle.dump([], open(pickle_reward_path, 'wb'))
        print('saved pickled rewards', pickle_reward_path)
    except OSError as e:
        print('could not init rewards pickle:', e)

    print('\n preparing buffer...')
    recent_imgs = []
    recent_actions = []
    recent_mouses = []
    recent_health = []
    recent_ammo = []
    recent_team = []
    recent_val = []

    if not ensure_foreground(hwin):
        return
    mp_restartgame()
    time.sleep(0.5)

    for i in range(0, 16):
        loop_start_time = time.time()
        img_small = grab_safe(hwin)
        x_img = np.expand_dims(img_small, 0)
        x_img = x_img.astype('float16')
        recent_imgs.append(x_img)

        dummy_action = np.zeros(int(aux_input_length))
        dummy_action[0] = 1  # encourage to start moving
        recent_actions.append(dummy_action)
        recent_mouses.append([0., 0.])
        recent_val.append(0)

        if IS_GSI:
            server.handle_request()
        recent_health.append(100)
        recent_ammo.append(30)
        recent_team.append('T')

        while time.time() < loop_start_time + 1 / loop_fps:
            time.sleep(0.01)

    print('\n starting loop, press Q to quit...')
    try:
        model_run = tp_load_model(model_save_dir, model_name + '_stateful')
    except (OSError, IOError):
        print('\nERROR: could not load %s_stateful from %s' % (model_name, model_save_dir))
        print('download the *_stateful.json/.h5 pair into ./model/ (新手入门指南.md 路线A)')
        return

    n_loops = 0
    keys_pressed = []
    Lclicks = 0
    Rclicks = 0
    count_inaction = 0
    training_data = []
    hdf5_num = 1
    iteration_deaths = 0
    iteration_kills = 0
    prev_vars = {'gsi_kills': -99, 'gsi_deaths': -99}
    time_for_pass = 0.1

    # wall-stuck rescue: a movement key held while the screen barely changes =
    # grinding a wall (running normally always changes the frame). After 2 s
    # of that, override the model's (usually zero) mouse with a forced 1 s turn.
    prev_img = None
    move_held = 0
    rescue_frames = 0
    rescue_dx = 0.0
    STUCK_STATIC_THR = 1.5  # mean abs frame diff below this counts as static

    while n_loops < 1000 * (mins_per_iter + 0.02):
        if IS_GSI:
            data_all = server.data_all or {}
            if 'map' not in data_all.keys() or 'player' not in data_all.keys():
                print('not running, map or player not in GSI keys:', list(data_all.keys()))
                time.sleep(5)
                continue

        loop_start_time = time.time()
        n_loops += 1
        keys_pressed_prev = keys_pressed.copy()
        Lclicks_prev = Lclicks
        Rclicks_prev = Rclicks

        del recent_imgs[0]
        del recent_actions[0]
        del recent_mouses[0]
        del recent_health[0]
        del recent_ammo[0]
        del recent_team[0]
        del recent_val[0]

        img_small = grab_safe(hwin)
        x_img = np.expand_dims(img_small, 0)
        x_img = x_img.astype('float16')
        recent_imgs.append(x_img)

        x_input_main = np.zeros(input_shape_lstm_pred)
        x_input_main[0] = recent_imgs[-1]
        x_input_main = np.expand_dims(x_input_main, 0)

        time_before_pass = time.time()
        y_preds = model_run.predict_on_batch(x_input_main)
        if n_loops <= 1:
            time_for_pass = 0.1
        else:
            time_for_pass = 0.5 * time_for_pass + 0.5 * (time.time() - time_before_pass)

        [keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks, val_pred] = \
            onehot_to_actions(y_preds)

        y_preds = y_preds.squeeze()
        keys_pred = y_preds[0:n_keys]
        clicks_pred = y_preds[n_keys:n_keys + 2]
        mouse_x_pred = y_preds[n_keys + 2:n_keys + 2 + len(mouse_x_possibles)]
        mouse_y_pred = y_preds[n_keys + 2 + len(mouse_x_possibles):
                               n_keys + 2 + len(mouse_x_possibles) + len(mouse_y_possibles)]

        # stuck-detection (same logic as original)
        if IS_WASD:
            if np.array([x in keys_pressed for x in ['w', 's', 'a', 'd']]).sum() == 0 \
                    and mouse_x == 0 and mouse_y == 0:
                count_inaction += 1
            else:
                count_inaction = 0
        else:
            if mouse_x == 0 and mouse_y == 0:
                count_inaction += 1
            else:
                count_inaction = 0

        if IS_PROBABILISTIC_ACTIONS or count_inaction > 8:
            Lclick_prob = clicks_pred[0]
            Lclicks = 1 if (np.random.rand() < Lclick_prob and IS_CLICKS) else 0

            Rclick_prob = clicks_pred[1]
            Rclicks = 1 if (np.random.rand() < Rclick_prob and IS_RCLICK) else 0

            if IS_JUMP:
                jump_prob = keys_pred[4:5][0]
                if 'expert' in model_name:
                    jump_prob *= 5  # manual boost, as in original
                if 'space' in keys_pressed:
                    keys_pressed.remove('space')
                if np.random.rand() <= jump_prob:
                    keys_pressed.append('space')

            if IS_RELOAD:
                reload_prob = keys_pred[n_keys - 1:n_keys][0]
                if 'r' in keys_pressed:
                    keys_pressed.remove('r')
                if np.random.rand() <= reload_prob:
                    keys_pressed.append('r')

            if IS_PROBABILISTIC_ACTIONS and IS_MOUSEMOVE:
                # sample the mouse bucket from the softmax EVERY frame.
                # argmax collapses to the centre bucket, and the agent's own
                # straight-line history then self-reinforces (covariate shift):
                # it never sees itself turning, so it never turns.
                mouse_x = np.random.choice(mouse_x_possibles, size=1, p=mouse_x_pred)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1, p=mouse_y_pred)[0]
            elif count_inaction > 8 and IS_MOUSEMOVE:
                mouse_x = np.random.choice(mouse_x_possibles, size=1, p=mouse_x_pred)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1, p=mouse_y_pred)[0]

            if count_inaction > 96 and IS_MOUSEMOVE:
                print('\n\n choosing random mouse \n')
                mouse_x = np.random.choice(mouse_x_possibles, size=1)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1)[0]
                model_run.reset_states()
                print('\n\n reset states')

            if np.random.rand() > 0.999:
                model_run.reset_states()
                print('\n\n reset states')

            if count_inaction > 96 and IS_WASD:
                if np.random.rand() < np.maximum(keys_pred[0], 0.05):
                    keys_pressed.append('w')
                elif np.random.rand() < np.maximum(keys_pred[1], 0.05):
                    keys_pressed.append('a')
                elif np.random.rand() < np.maximum(keys_pred[2], 0.05):
                    keys_pressed.append('s')
                elif np.random.rand() < np.maximum(keys_pred[3], 0.05):
                    keys_pressed.append('d')

        # MOUSE_SCALE compensates for the raw-input domain gap (see cs2_config)
        mouse_x_smooth = np.clip(mouse_x * MOUSE_SCALE, -300, 300)
        mouse_y_smooth = mouse_y * MOUSE_SCALE

        # wall-stuck detection (after the model's own action choice)
        if any(k in keys_pressed for k in ('w', 'a', 's', 'd')):
            move_held += 1
        else:
            move_held = 0
        if prev_img is None:
            static = False
        else:
            static = float(np.mean(cv2.absdiff(img_small, prev_img))) < STUCK_STATIC_THR
        prev_img = img_small
        if move_held > 2 * loop_fps and static and rescue_frames == 0 and IS_MOUSEMOVE:
            rescue_frames = loop_fps
            rescue_dx = float(np.random.choice([-300., -200., -100., 100., 200., 300.]))
            print('\n[stuck rescue] movement + static screen -> forcing turn %.0f\n' % rescue_dx)
        if rescue_frames > 0:
            mouse_x_smooth = np.clip(rescue_dx * MOUSE_SCALE, -300, 300)
            mouse_y_smooth = 0.0
            rescue_frames -= 1

        if IS_MOUSEMOVE:
            if IS_SPLIT_MOUSE:
                set_pos(mouse_x_mid + mouse_x_smooth / 2,
                        mouse_y_mid + mouse_y_smooth / 2, Wd, Hd)
            else:
                set_pos(mouse_x_mid + mouse_x_smooth,
                        mouse_y_mid + mouse_y_smooth, Wd, Hd)

        if n_loops > 2:
            model_name_print = model_name[-15:] if len(model_name) > 15 else model_name
            print(model_name_print, ', n', n_loops,
                  ', ds', round(iteration_deaths * 1000 / n_loops, 3),
                  ', ks', round(iteration_kills * 1000 / n_loops, 3),
                  ', fwd', round(time_for_pass, 3),
                  ', ms', round(time.time() - loop_start_time, 3),
                  ', inaction', count_inaction, end='\r')
            if n_loops % 1000 == 0:
                print('')

        keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot = \
            actions_to_onehot(keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks)

        recent_actions.append(np.concatenate([keys_pressed_onehot, Lclicks_onehot,
                                              Rclicks_onehot, mouse_x_onehot,
                                              mouse_y_onehot]))
        recent_mouses.append([mouse_x / mouse_x_lim[1], mouse_y / mouse_y_lim[1]])
        recent_val.append(val_pred)
        if IS_GSI:
            recent_health.append(gsi_get('player.state.health', -99))
            recent_team.append(gsi_get('player.team', 'No GSI'))
            gsi_weapons = gsi_get('player.weapons')  # noqa: F841
        else:
            recent_health.append(-99)
            recent_team.append('No GSI')
            gsi_weapons = None  # noqa: F841
        recent_ammo.append(None)

        helper_i = np.zeros(6)
        helper_i[2] = val_pred
        curr_vars = {}
        curr_vars['gsi_kills'] = gsi_get('player.match_stats.kills', -99) if IS_GSI else -99
        curr_vars['gsi_deaths'] = gsi_get('player.match_stats.deaths', -99) if IS_GSI else -99
        if n_loops >= 2:
            if curr_vars['gsi_kills'] != -99 and \
                    curr_vars['gsi_kills'] == prev_vars['gsi_kills'] + 1:
                helper_i[0] = 1
                iteration_kills += 1
            if curr_vars['gsi_deaths'] != -99 and \
                    curr_vars['gsi_deaths'] == prev_vars['gsi_deaths'] + 1:
                helper_i[1] = 1
                iteration_deaths += 1
        prev_vars = curr_vars.copy()

        # apply actions (release old, press new) - same as original
        if IS_WASD:
            for key, ch in [('w', w_char), ('a', a_char), ('s', s_char), ('d', d_char)]:
                if key in keys_pressed_prev and key not in keys_pressed:
                    ReleaseKey(ch)
                if key in keys_pressed:
                    HoldKey(ch)
        if IS_JUMP:
            if 'space' in keys_pressed_prev and 'space' not in keys_pressed:
                ReleaseKey(space_char)
            if 'space' in keys_pressed:
                HoldKey(space_char)
        if IS_RELOAD:
            if 'r' in keys_pressed_prev and 'r' not in keys_pressed:
                ReleaseKey(r_char)
            if 'r' in keys_pressed:
                HoldKey(r_char)
        if IS_KEYS:
            for key, ch in [('shift', shift_char), ('ctrl', ctrl_char),
                            ('1', one_char), ('2', two_char), ('3', three_char)]:
                if key in keys_pressed_prev and key not in keys_pressed:
                    ReleaseKey(ch)
                if key in keys_pressed:
                    HoldKey(ch)
        if IS_CLICKS:
            if Lclicks == 0 and Lclicks_prev == 1:
                release_left_click()
            if Lclicks == 1:
                hold_left_click()
        if IS_RCLICK:
            if Rclicks == 0 and Rclicks_prev == 1:
                release_right_click()
            if Rclicks == 1:
                hold_right_click()

        if IS_GSI:
            server.handle_request()

        keys_pressed_tp = key_check()
        if 'Q' in keys_pressed_tp:
            print('exiting...')
            if IS_GSI:
                server.server_close()
            break

        if IS_SPLIT_MOUSE and IS_MOUSEMOVE:
            while time.time() < loop_start_time + 0.5 / loop_fps:
                time.sleep(0.001)
            set_pos(mouse_x_mid + mouse_x_smooth / 2,
                    mouse_y_mid + mouse_y_smooth / 2, Wd, Hd)

        if IS_DEMO:
            if _demo_overlay(img_small, mouse_x_smooth, mouse_y_smooth, keys_pressed,
                             Lclicks, clicks_pred, n_loops):
                print('exiting...')
                if IS_GSI:
                    server.server_close()
                break

        if SAVE_TRAIN_DATA:
            training_data.append([img_small, curr_vars])
            if len(training_data) >= 1000:
                np.save(os.path.join(os.getcwd(), 'cs2_agent_data_%d' % hdf5_num),
                        training_data)
                training_data = []
                hdf5_num += 1

        wait_for_loop_end(loop_start_time, loop_fps, n_loops, is_clear_decals=True)

    # shutdown: release everything we might be holding
    print('\n\n -- finished, releasing keys --')
    for ch in [w_char, a_char, s_char, d_char, r_char, space_char]:
        ReleaseKey(ch)
    release_left_click()
    release_right_click()
    time.sleep(0.1)
    pause_game()

    if n_loops > 0 and os.path.isfile(pickle_reward_path):
        try:
            reward_list = pickle.load(open(pickle_reward_path, 'rb'))
        except (OSError, EOFError):
            reward_list = []
        reward_list.append([model_name, 0, n_loops,
                            iteration_kills * 1000 / n_loops,
                            iteration_deaths * 1000 / n_loops,
                            iteration_kills, iteration_deaths,
                            iteration_kills / max(iteration_deaths, 1)])
        pickle.dump(reward_list, open(pickle_reward_path, 'wb'))
        print('saved pickled rewards', pickle_reward_path)


# ---------------------------------------------------------------- entry point
def main():
    args = parse_args()

    if args.check_window:
        mode_check_window()
        return

    hwnd, title = find_cs2_window()
    wait_started = False
    while hwnd is None or win32gui.IsIconic(hwnd):
        if not wait_started:
            print('CS2 window not found/minimized - waiting for the game '
                  'window (start CS2 and enter a match)...')
            wait_started = True
        time.sleep(3)
        hwnd, title = find_cs2_window()
    if wait_started:
        print('window acquired: %r (hwnd=%s)' % (title, hwnd))
    # the game can be mid-restart between discovery and this call; retry
    for _attempt in range(10):
        try:
            win32gui.SetForegroundWindow(hwnd)
            break
        except win32gui.error as e:
            print('SetForegroundWindow failed (%s) - re-finding window...' % e)
            time.sleep(3)
            hwnd, title = find_cs2_window()
            if hwnd is None:
                continue
    time.sleep(1)

    if args.fps_test:
        mode_fps_test(hwnd)
        return

    Wd, Hd = get_monitor_size(args.monitor)
    time.sleep(0.2)
    mid_x, mid_y = mouse_check()

    if args.smoke:
        mode_smoke(hwnd, Wd, Hd, mid_x, mid_y)
        return
    if args.calibrate:
        mode_calibrate(hwnd, Wd, Hd, mid_x, mid_y)
        return

    run_agent(args, hwnd)


if __name__ == '__main__':
    main()
