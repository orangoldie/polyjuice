"""Test whether OE 'date' is the actual game-start timestamp.

Hypothesis: if OE date is true kickoff time, then `oe_date + gamelength` should
match the moment the Polymarket price pins to its resolution (0.999 / 0.001).
Compute that offset on several high-volume joined markets.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('42_pmoe_date_anchor_check.txt')
import pandas as pd

joined = pd.read_csv('data/polymarket/markets_oe_joined.csv')
joined = joined[joined['status'] == 'ok']

# Load OE for date + gamelength
oe_frames = []
for yr in [2022, 2023, 2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['gameid', 'date', 'gamelength', 'position', 'result', 'teamname'],
        low_memory=False,
    )
    df = df[df['position'] == 'team']
    oe_frames.append(df)
oe = pd.concat(oe_frames, ignore_index=True)
oe['date'] = pd.to_datetime(oe['date'], utc=True, errors='coerce')

oe_meta = oe.groupby('gameid').agg(
    date=('date', 'first'),
    gamelength=('gamelength', 'first'),
).reset_index()

# Pick high-volume markets — load trades.csv and count per cid first
print('counting trades per joined cid...')
counts = {}
for chunk in pd.read_csv('data/polymarket/trades.csv', chunksize=500_000,
                         dtype={'condition_id': str},
                         usecols=['condition_id']):
    vc = chunk['condition_id'].value_counts()
    for cid, n in vc.items():
        counts[cid] = counts.get(cid, 0) + int(n)

joined['n_trades'] = joined['condition_id'].map(counts).fillna(0).astype(int)
top = joined.sort_values('n_trades', ascending=False).head(8)

# Pull all trades for those cids
top_cids = set(top['condition_id'])
print(f'pulling fills for {len(top_cids)} top-volume markets...')
parts = []
for chunk in pd.read_csv('data/polymarket/trades.csv', chunksize=500_000,
                         dtype={'condition_id': str, 'asset': str}):
    sub = chunk[chunk['condition_id'].isin(top_cids)]
    if len(sub):
        parts.append(sub)
trades = pd.concat(parts, ignore_index=True)
trades['ts'] = pd.to_datetime(trades['timestamp_unix'], unit='s', utc=True)

print(f'\n{"event_slug":40s} {"G":>2s}  {"n_fills":>7s}  {"oe_date":>20s}  {"oe_end":>20s}  {"first_pin":>20s}  {"offset_s":>9s}  {"final":>5s}')
for _, r in top.iterrows():
    cid = r['condition_id']
    meta = oe_meta[oe_meta['gameid'] == r['oe_gameid']].iloc[0]
    oe_date = meta['date']
    glen = meta['gamelength']
    if pd.isna(glen):
        continue
    oe_end = oe_date + pd.Timedelta(seconds=int(glen))

    sub = trades[trades['condition_id'] == cid].sort_values('ts').copy()
    # Unified P(outcome_0): outcome_idx=0 -> price; idx=1 -> 1 - price
    sub['p0'] = sub.apply(
        lambda x: x['price'] if int(x['outcome_index']) == 0 else 1 - x['price'],
        axis=1,
    )
    # Find first time after oe_date when price stays pinned (>=0.99 or <=0.01) and never moves out
    # Simpler: walk from end backwards, find earliest contiguous pin
    final_p = sub.iloc[-1]['p0']
    pinned_thresh = 0.99 if final_p > 0.5 else 0.01
    if final_p > 0.5:
        cond = sub['p0'] >= pinned_thresh
    else:
        cond = sub['p0'] <= pinned_thresh
    # earliest fill where cond holds and remains true through end
    pinned_after = sub[cond & (sub['ts'] >= oe_date)]
    if len(pinned_after) == 0:
        first_pin = pd.NaT
        offset_s = float('nan')
    else:
        # sweep forward: find first index from which cond is true for ALL subsequent rows
        idx = sub[sub['ts'] >= oe_date].index
        first_pin = pd.NaT
        for i in idx:
            tail = sub.loc[i:]
            if (cond.loc[tail.index]).all():
                first_pin = sub.loc[i, 'ts']
                break
        offset_s = (first_pin - oe_end).total_seconds() if pd.notna(first_pin) else float('nan')

    print(f'{r["event_slug"]:40s} {int(r["game_number"]):>2d}  {int(r["n_trades"]):>7d}  '
          f'{oe_date.strftime("%Y-%m-%d %H:%M:%S"):>20s}  '
          f'{oe_end.strftime("%Y-%m-%d %H:%M:%S"):>20s}  '
          f'{first_pin.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(first_pin) else "NaT":>20s}  '
          f'{offset_s:>9.1f}  {final_p:>5.3f}')
