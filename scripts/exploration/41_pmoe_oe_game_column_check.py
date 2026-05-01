"""Sanity check: verify OE 'game' column numbers within a series 1..N as expected."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('41_pmoe_oe_game_column_check.txt')
import pandas as pd

df = pd.read_csv(
    'data/lol-historical-data/2025_LoL_esports_match_data_from_OraclesElixir.csv',
    usecols=['gameid', 'date', 'game', 'teamname', 'position', 'league'],
    low_memory=False,
)
df = df[df['position'] == 'team']
df['date'] = pd.to_datetime(df['date'])
df['date_day'] = df['date'].dt.date

# Group: (date_day, frozenset(teams), game) -> rows. Look at distribution of N per series.
grp = df.groupby('gameid').agg(
    date=('date_day', 'first'),
    game=('game', 'first'),
    league=('league', 'first'),
    teams=('teamname', lambda s: tuple(sorted(s.tolist()))),
).reset_index()
grp['series_key'] = list(zip(grp['date'], grp['teams']))

series_lens = grp.groupby('series_key')['game'].max()
print('series-length distribution (max game in a series):')
print(series_lens.value_counts().sort_index().to_string())

# Show a couple of LCK BO5s
lck = grp[grp['league'].isin(['LCK', 'WLDs', 'MSI'])].copy()
big_series = lck.groupby('series_key').filter(lambda g: len(g) >= 4)
if len(big_series):
    keys = big_series['series_key'].drop_duplicates().head(3).tolist()
    for k in keys:
        rows = big_series[big_series['series_key'] == k].sort_values('game')
        print(f'\n--- {rows.iloc[0]["league"]} {k[0]} {k[1]} ---')
        for _, r in rows.iterrows():
            print(f'  game={r["game"]}  gameid={r["gameid"]}')
