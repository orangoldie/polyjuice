"""Spot-check 2 random joined Polymarket-OE rows: print both sides side-by-side."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('40_pmoe_spot_check.txt')
import random
import pandas as pd

random.seed(42)

joined = pd.read_csv('data/polymarket/markets_oe_joined.csv')
joined = joined[joined['status'] == 'ok'].reset_index(drop=True)
print(f'ok joined rows: {len(joined)}')

samples = joined.sample(n=2, random_state=7).to_dict('records')

# Load OE once across years
oe_frames = []
for yr in [2022, 2023, 2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        low_memory=False,
    )
    oe_frames.append(df)
oe_all = pd.concat(oe_frames, ignore_index=True)

trades_iter_path = 'data/polymarket/trades.csv'

for i, s in enumerate(samples, 1):
    print(f'\n{"="*80}')
    print(f'SAMPLE {i}')
    print('=' * 80)

    print('\n--- POLYMARKET SIDE ---')
    print(f'  condition_id : {s["condition_id"]}')
    print(f'  event_slug   : {s["event_slug"]}')
    print(f'  match_date   : {s["match_date"]}')
    print(f'  game_number  : {s["game_number"]}')
    print(f'  outcome_0    : {s["pm_outcome_0"]!r}')
    print(f'  outcome_1    : {s["pm_outcome_1"]!r}')

    # First/last 3 trades for this market — stream-filter so we don't load whole trades.csv
    cid = s['condition_id']
    rows = []
    for chunk in pd.read_csv(trades_iter_path, chunksize=200_000,
                             dtype={'condition_id': str, 'asset': str,
                                    'transaction_hash': str, 'proxy_wallet': str}):
        sub = chunk[chunk['condition_id'] == cid]
        if len(sub):
            rows.append(sub)
    if rows:
        tr = pd.concat(rows, ignore_index=True).sort_values('timestamp_unix')
        tr['ts_utc'] = pd.to_datetime(tr['timestamp_unix'], unit='s', utc=True)
        print(f'  n_trades     : {len(tr)}')
        print(f'  first trade  : {tr.iloc[0]["ts_utc"]}  price={tr.iloc[0]["price"]}  side={tr.iloc[0]["side"]}  outcome_idx={tr.iloc[0]["outcome_index"]}')
        print(f'  last trade   : {tr.iloc[-1]["ts_utc"]} price={tr.iloc[-1]["price"]}  side={tr.iloc[-1]["side"]}  outcome_idx={tr.iloc[-1]["outcome_index"]}')
        print(f'  price range  : [{tr["price"].min():.3f}, {tr["price"].max():.3f}]')
    else:
        print(f'  (no trades found in trades.csv for this cid?!)')

    print('\n--- OE SIDE ---')
    gid = s['oe_gameid']
    g = oe_all[oe_all['gameid'] == gid]
    teams = g[g['position'] == 'team']
    print(f'  oe_gameid    : {gid}')
    print(f'  league       : {teams.iloc[0]["league"]} ({teams.iloc[0]["year"]} {teams.iloc[0]["split"]})')
    print(f'  oe_date      : {teams.iloc[0]["date"]} (UTC)')
    print(f'  patch        : {teams.iloc[0]["patch"]}')
    print(f'  gamelength   : {teams.iloc[0]["gamelength"]}s ({teams.iloc[0]["gamelength"]/60:.1f} min)')
    for _, t in teams.iterrows():
        result = 'WON' if t['result'] == 1 else 'lost'
        print(f'  {t["side"]:5s} {t["teamname"]:35s} {result}')

    # Verify alignment
    print('\n--- ALIGNMENT CHECK ---')
    pm_set = {s['oe_team_a'], s['oe_team_b']}
    oe_set = set(teams['teamname'].tolist())
    print(f'  pm teams (resolved): {pm_set}')
    print(f'  oe teams           : {oe_set}')
    print(f'  match              : {pm_set == oe_set}')
    print(f'  pm match_date      : {s["match_date"]}')
    print(f'  oe date_day        : {pd.to_datetime(teams.iloc[0]["date"]).date()}')
    print(f'  pm game_number     : {s["game_number"]}')
    print(f'  oe game            : {teams.iloc[0]["game"]}')
