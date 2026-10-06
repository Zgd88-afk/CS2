# Incremental recording validator: checks every cs2_ft_*.npz with index
# >= --since and prints one PASS/FAIL summary line per run.
#
# Checks per new file: shape integrity, blank frames, mouse both-axis activity,
# key/click presence. Overall FAIL if: any new file missing 'mouse' (v1), any
# file with >5% blank frames, mouse nonzero < 30% on a file with >300 frames,
# or zero new files found (recording stalled).
#
# Usage: python tools_validate_recording.py --since 8
# Prints: "SINCE <n>: <k> new files, <frames> frames - PASS" (or FAIL + reason)

import argparse
import glob
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', type=int, required=True,
                    help='only validate files numbered above this')
    ap.add_argument('--folder', default=r'D:\csgo_data\cs2_ft')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, 'cs2_ft_*.npz')),
                   key=lambda p: int(os.path.basename(p)[7:-4]))
    new = [p for p in files if int(os.path.basename(p)[7:-4]) > args.since]
    if not new:
        print('SINCE %d: 0 new files - FAIL (recording stalled?)' % args.since)
        sys.exit(1)

    total, blank, mnz, fires, mov = 0, 0, 0, 0, 0
    problems = []
    for p in new:
        z = np.load(p)
        if 'mouse' not in z:
            problems.append('%s: v1 file (no mouse)' % os.path.basename(p))
            continue
        imgs, keys, clicks, mouse = z['imgs'], z['keys'], z['clicks'], z['mouse']
        if imgs.shape[1:] != (150, 280, 3):
            problems.append('%s: bad img shape %s' % (os.path.basename(p), imgs.shape))
            continue
        b = int((imgs.reshape(len(imgs), -1).std(axis=1) < 2).sum())
        m = int(((np.abs(mouse[:, 0]) + np.abs(mouse[:, 1])) > 0).sum())
        total += len(imgs)
        blank += b
        mnz += m
        fires += int(clicks[:, 0].sum())
        mov += int(keys[:, :4].any(axis=1).sum())
        if len(imgs) > 300 and b / len(imgs) > 0.05:
            problems.append('%s: %.0f%% blank frames' % (os.path.basename(p), 100 * b / len(imgs)))
        if len(imgs) > 300 and m / len(imgs) < 0.30:
            problems.append('%s: mouse active only %.0f%%' % (os.path.basename(p), 100 * m / len(imgs)))

    verdict = 'PASS' if not problems else 'FAIL: ' + '; '.join(problems)
    print('SINCE %d: %d new files, %d frames (~%.1f min) | mouse %d%% | '
          'move %d%% | fire %d - %s'
          % (args.since, len(new), total, total / 16 / 60,
             100 * mnz / max(total, 1), 100 * mov / max(total, 1), fires,
             verdict))
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
