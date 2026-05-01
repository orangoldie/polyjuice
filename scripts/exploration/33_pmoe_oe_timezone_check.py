"""Check OE date column timezone by looking at hour-of-day distribution per league.

LCK regular schedule: 17:00 KST = 08:00 UTC
LEC regular schedule: 18:00-21:00 CET/CEST = 17:00-20:00 UTC (winter) / 16:00-19:00 UTC (summer)
LCS regular schedule: 15:00-18:00 PT = 22:00-01:00 UTC
LPL regular schedule: 17:00 CST = 09:00 UTC
"""
import pandas as pd

from _tee import tee_stdout
tee_stdout('33_pmoe_oe_timezone_check.txt')

df = pd.read_csv(
    'data/lol-historical-data/2025_LoL_esports_match_data_from_OraclesElixir.csv',
    usecols=['league', 'date', 'position'],
    low_memory=False,
)
df = df[df['position'] == 'team']
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour

print('league   n      hour_p10  hour_p50  hour_p90  modes')
print('-' * 70)
for lg in sorted(df['league'].unique()):
    sub = df[df['league'] == lg]
    if len(sub) < 50:
        continue
    p = sub['hour'].quantile([.1, .5, .9]).tolist()
    modes = sub['hour'].mode().tolist()[:3]
    print(f'{lg:8s} {len(sub):5d}  {int(p[0]):>8d}  {int(p[1]):>8d}  {int(p[2]):>8d}  {modes}')
