"""Diagnose why ~1k joined gameids are missing from processed/checkpoint_T_gpr.csv.

Cross our joined ok gameids against:
  - data/processed/checkpoint_15_gpr.csv  (the modeling-ready dataset)
  - data/processed/03_features.csv         (pre-checkpoint-split features)
  - data/processed/02_one_row.csv          (one row per game, post one-row consolidation)
  - data/processed/01_filtered.csv         (post initial league/tier filter)

For each missing gameid, report league/tier so we can decide whether to broaden
the processed pipeline.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('44_pmoe_no_model_prob_diag.txt')
import pandas as pd

joined = pd.read_csv('data/polymarket/markets_oe_joined.csv')
joined = joined[joined['status'] == 'ok']
joined_gids = set(joined['oe_gameid'])
print(f'joined ok gameids: {len(joined_gids)}')

# OE meta (league, tier proxy via known league ranks) — load from raw CSVs
oe_frames = []
for yr in [2022, 2023, 2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['gameid', 'league', 'datacompleteness', 'position', 'year'],
        low_memory=False,
    )
    df = df[df['position'] == 'team'].drop_duplicates('gameid')
    oe_frames.append(df)
oe_meta = pd.concat(oe_frames, ignore_index=True)
oe_meta = oe_meta[oe_meta['gameid'].isin(joined_gids)].copy()
print(f'OE meta rows for joined gameids: {len(oe_meta)}')

# Stage by stage
def stage_set(path, col='gameid'):
    df = pd.read_csv(path, usecols=[col], low_memory=False)
    return set(df[col].unique())

stages = [
    ('01_filtered',         'data/processed/01_filtered.csv'),
    ('02_one_row',          'data/processed/02_one_row.csv'),
    ('03_features',         'data/processed/03_features.csv'),
    ('checkpoint_15_gpr',   'data/processed/checkpoint_15_gpr.csv'),
]
present = {}
for name, path in stages:
    s = stage_set(path)
    n_joined_in = len(joined_gids & s)
    print(f'  {name:25s} {len(s):>7d} total  /  {n_joined_in:>5d} of {len(joined_gids)} joined')
    present[name] = s

missing_at_final = joined_gids - present['checkpoint_15_gpr']
print(f'\nmissing at checkpoint_15_gpr stage: {len(missing_at_final)}')

# Where do they drop? Find first stage that drops them
print('\nfirst stage at which each missing gameid drops:')
drop_stages = []
for gid in missing_at_final:
    for name, _ in stages:
        if gid not in present[name]:
            drop_stages.append(name)
            break
    else:
        drop_stages.append('checkpoint_15_gpr_only')
print(pd.Series(drop_stages).value_counts().to_string())

# League breakdown for the missing
print('\nleague breakdown of missing gameids:')
miss_meta = oe_meta[oe_meta['gameid'].isin(missing_at_final)]
print(miss_meta['league'].value_counts().head(25).to_string())

print('\ndatacompleteness breakdown of missing gameids:')
print(miss_meta['datacompleteness'].value_counts().to_string())

# Compare to the present-in-final ones for reference
print('\nleague breakdown of PRESENT-in-checkpoint_15_gpr joined gameids (top 15):')
pres_meta = oe_meta[oe_meta['gameid'].isin(present['checkpoint_15_gpr'])]
print(pres_meta['league'].value_counts().head(15).to_string())
