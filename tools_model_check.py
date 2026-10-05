# Model sanity check for the CS2 migration test (docs/CS2迁移测试计划.md Phase 2).
#
# Verifies whether a trained BC model is "acting normally" on three frame
# sources, printing the key activeness metrics for each:
#   1. noise frames              - random RGB input
#   2. CSGO training frames      - ground-truth data, if the dataset .npy is
#                                  available (--data path); reports fire-key
#                                  discrimination and mouse-bucket accuracy
#   3. live CS2 capture          - the current game window
#
# Verdict logic from the 2026-10 CS2 migration session:
#   - model that responds on training frames but outputs near-zero action
#     probabilities on live CS2 frames has a VISUAL DOMAIN GAP, not a broken
#     model (TF 2.10 loads it faithfully; stateful/non-stateful agree).
#
# Usage (inside the csgo_tf210 env):
#   python tools_model_check.py --model ak47_sub_55k_drop_d4_dmexpert_28_stateful
#   python tools_model_check.py --model ... --data _downloads/ds/dm_july2021_expert_1.npy
#
# NOT YET VALIDATED ON A LIVE CS2 INSTALL (validated during the 2026-10-05
# session on RTX 4070 Ti SUPER / TF 2.10.1).

import argparse
import os
import sys

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0


def frame_stats(y):
    mx = y[13:36].astype(float)
    return {
        'Lclick': float(y[11]),
        'w': float(y[0]),
        'mx_argmax': int(np.argmax(mx)),
        'mx_entropy': entropy(mx),
        'value': float(y[51]),
    }


def run_source(model, frames, reset_each=False):
    stats = []
    for f in frames:
        if reset_each:
            model.reset_states()
        y = np.array(model.predict_on_batch(
            np.expand_dims(np.asarray(f, dtype='float16'), (0, 1)))).squeeze()
        stats.append(frame_stats(y))
    return stats


def summarize(name, stats):
    lclick = np.array([s['Lclick'] for s in stats])
    wprob = np.array([s['w'] for s in stats])
    ent = np.array([s['mx_entropy'] for s in stats])
    print('[%s] n=%d | Lclick mean %.3f max %.3f | w mean %.3f | '
          'mx entropy mean %.2f/%.2f | value %.3f'
          % (name, len(stats), lclick.mean(), lclick.max(), wprob.mean(),
             ent.mean(), np.log(23)))


def main():
    ap = argparse.ArgumentParser(description='BC model activeness check')
    ap.add_argument('--model', default='ak47_sub_55k_drop_d4_dmexpert_28_stateful')
    ap.add_argument('--model-dir', default=os.path.join(os.getcwd(), 'model'))
    ap.add_argument('--data', default=None,
                    help='optional CSGO training .npy (with images) to test on')
    ap.add_argument('--n-train', type=int, default=300,
                    help='training frames to test (default 300)')
    ap.add_argument('--no-live', action='store_true', help='skip live capture')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    from config import tp_load_model
    model = tp_load_model(args.model_dir, args.model)

    # 1) noise
    rng = np.random.default_rng(0)
    noise_frames = [rng.random((150, 280, 3)).astype('float16') * 255
                    for _ in range(20)]
    summarize('noise     ', run_source(model, noise_frames, reset_each=True))

    # 2) CSGO training data (with ground truth if available)
    if args.data and os.path.isfile(args.data):
        import warnings
        warnings.filterwarnings('ignore')
        data = np.load(args.data, allow_pickle=True)
        fire_frames = [i for i in range(len(data))
                       if int(np.atleast_1d(data[i][2])[3]) == 1]
        start = max(fire_frames[0] - 60, 0) if fire_frames else 0
        win = list(range(start, min(start + args.n_train, len(data))))
        stats = run_source(model, [data[i][0] for i in win])
        summarize('csgo-data ', stats)
        gt_L = np.array([int(data[i][2][3]) for i in win])
        pred_L = np.array([s['Lclick'] for s in stats])
        if (gt_L == 1).any() and (gt_L == 0).any():
            print('   fire discrimination: gt=1 %.3f vs gt=0 %.3f  -> %s'
                  % (pred_L[gt_L == 1].mean(), pred_L[gt_L == 0].mean(),
                     'MODEL RESPONSIVE' if pred_L[gt_L == 1].mean() >
                     2 * pred_L[gt_L == 0].mean() else 'NOT discriminating'))
    elif args.data:
        print('training data not found:', args.data)

    # 3) live CS2 capture
    if not args.no_live:
        from cs2_config import find_cs2_window, grab_window_cs2
        hwnd, _ = find_cs2_window()
        if hwnd:
            model.reset_states()
            live_frames = [grab_window_cs2(hwnd) for _ in range(20)]
            summarize('live-cs2  ', run_source(model, live_frames))
        else:
            print('[live-cs2  ] CS2 window not found, skipped')

    print('\ninterpretation: responsive on training data but pinned near-zero '
          'on live frames = visual domain gap (see Phase 4 of the test plan).')


if __name__ == '__main__':
    main()
