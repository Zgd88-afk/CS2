# Score statistics from CS2 demo files, for evaluating the BC agent
# (primary scoring method of docs/CS2迁移测试计划.md Phase 2/3 - no GSI needed).
#
# Usage:
#   pip install demoparser2 pandas          # any Python >= 3.8, separate env is fine
#   python tools_demo_stats.py match1.dem match2.dem
#   python tools_demo_stats.py path/to/demo_folder/
#   python tools_demo_stats.py --player "YourSteamName" *.dem   # tag the agent row
#   python tools_demo_stats.py --list-columns match1.dem        # debug column names
#
# Output:
#   - per-demo, per-player kills/deaths/K-D/K/min table printed to console
#   - all rows appended to demo_stats_summary.csv (override with --out)
#
# The agent is the account you logged in with; in an offline bot match it is
# the only human player, so its row is easy to spot. Column names differ
# slightly between demoparser2 versions - common variants are resolved
# automatically, and --list-columns shows what your version provides.
#
# NOT YET VALIDATED ON A LIVE CS2 INSTALL.

import argparse
import glob
import os
import sys

import pandas as pd

from demoparser2 import DemoParser

ATTACKER_COL_CANDIDATES = ["attacker_name", "attacker_steamid", "attacker"]
VICTIM_COL_CANDIDATES = ["user_name", "user_steamid", "user"]


def pick_col(df, candidates):
    """Return the first column name present in df among candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_demo(path, player_filter=None):
    parser = DemoParser(path)

    header = {}
    try:
        header = parser.parse_header()
    except Exception as e:
        print('  ! could not parse header: %s' % e)

    map_name = header.get('map_name', '?')
    duration_s = None
    if header.get('playback_time'):
        duration_s = float(header['playback_time'])
    elif header.get('playback_ticks'):
        # CS2 demos are recorded at 64 ticks/s (source 2 matchmaking demos);
        # POVs can differ slightly - good enough for K/min
        duration_s = float(header['playback_ticks']) / 64.0

    kills_df = parser.parse_event("player_death")

    attacker_col = pick_col(kills_df, ATTACKER_COL_CANDIDATES)
    victim_col = pick_col(kills_df, VICTIM_COL_CANDIDATES)
    if attacker_col is None or victim_col is None:
        raise KeyError(
            'could not find name columns; available columns: %s'
            % list(kills_df.columns))

    kill_counts = kills_df.groupby(attacker_col).size()
    death_counts = kills_df.groupby(victim_col).size()

    players = sorted(set(kill_counts.index) | set(death_counts.index))
    rows = []
    for name in players:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            continue
        kills = int(kill_counts.get(name, 0))
        deaths = int(death_counts.get(name, 0))
        kd = kills / deaths if deaths > 0 else float(kills)
        kpm = kills / (duration_s / 60.0) if duration_s else float('nan')
        rows.append({
            'demo': os.path.basename(path),
            'map': map_name,
            'duration_s': round(duration_s, 1) if duration_s else None,
            'player': name,
            'kills': kills,
            'deaths': deaths,
            'kd': round(kd, 3),
            'k_per_min': round(kpm, 3) if duration_s else None,
            'is_agent': (player_filter is not None and name == player_filter),
        })
    df = pd.DataFrame(rows).sort_values('kills', ascending=False)
    return df, map_name, duration_s


def collect_demo_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, '*.dem'))))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            print('skipping (not found): %s' % item)
    return paths


def main():
    ap = argparse.ArgumentParser(
        description='Kill/death statistics from CS2 demo files (demoparser2)')
    ap.add_argument('demos', nargs='+',
                    help='.dem files and/or folders containing them')
    ap.add_argument('--out', default='demo_stats_summary.csv',
                    help='output CSV path (default: demo_stats_summary.csv)')
    ap.add_argument('--player', default=None,
                    help='Steam name of the agent-controlled account to tag')
    ap.add_argument('--list-columns', action='store_true',
                    help='print player_death columns of the first demo and exit')
    args = ap.parse_args()

    paths = collect_demo_paths(args.demos)
    if not paths:
        print('no .dem files found')
        sys.exit(1)

    if args.list_columns:
        df = DemoParser(paths[0]).parse_event("player_death")
        print('columns of %s:' % paths[0])
        print(df.columns.tolist())
        return

    all_rows = []
    for path in paths:
        print('\n=== %s ===' % os.path.basename(path))
        try:
            df, map_name, duration_s = parse_demo(path, args.player)
        except Exception as e:
            print('  ! parse failed: %s' % e)
            continue
        print('  map: %s   duration: %s' % (
            map_name,
            '%.1f min' % (duration_s / 60.0) if duration_s else '?'))
        print(df.to_string(index=False))
        all_rows.append(df)

    if not all_rows:
        print('\nnothing parsed')
        sys.exit(1)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(args.out, index=False)
    print('\nsaved %d rows -> %s' % (len(summary), args.out))

    if args.player:
        agent_rows = summary[summary['is_agent']]
        if len(agent_rows):
            print('\nagent summary (%s):' % args.player)
            print(agent_rows[['demo', 'kills', 'deaths', 'kd', 'k_per_min']].to_string(index=False))
            print('\nmean K/D: %.3f   mean kills/min: %.3f'
                  % (agent_rows['kd'].mean(), agent_rows['k_per_min'].mean()))
        else:
            print('\n(!) player %r not found in any demo - check the exact in-game name'
                  % args.player)


if __name__ == '__main__':
    main()
