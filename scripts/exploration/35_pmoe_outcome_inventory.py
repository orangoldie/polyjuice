"""Inspect outcome_0 / outcome_1 columns from markets.csv — these may be full team names."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('35_pmoe_outcome_inventory.txt')
import pandas as pd

mk = pd.read_csv(
    'data/polymarket/markets.csv',
    usecols=['outcome_0', 'outcome_1', 'question', 'game_number', 'fetch_status'],
)
mk = mk[mk['fetch_status'] == 'ok']
print('sample rows:')
print(mk.head(15).to_string())
print()

uniq = set(mk['outcome_0'].dropna()).union(set(mk['outcome_1'].dropna()))
print(f'unique outcome strings: {len(uniq)}')

# Cross with OE
oe_teams = set()
for yr in [2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['teamname', 'position'],
        low_memory=False,
    )
    df = df[df['position'] == 'team']
    oe_teams.update(df['teamname'].dropna().unique())

def norm(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())

oe_norm = {norm(t): t for t in oe_teams}

exact = 0
miss = []
for o in uniq:
    if norm(o) in oe_norm:
        exact += 1
    else:
        miss.append(o)
print(f'exact normalized match to OE: {exact} / {len(uniq)}')
print(f'misses: {len(miss)}')
print('sample misses (first 40):')
for m in sorted(miss)[:40]:
    print(f'  {m!r}')
