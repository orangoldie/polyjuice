"""Quick OE teamname lookup for resolving Polymarket alias misses."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('37_pmoe_oe_team_lookup.txt')
import pandas as pd

teams = set()
for yr in [2023, 2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['teamname', 'position'], low_memory=False,
    )
    df = df[df['position'] == 'team']
    teams.update(df['teamname'].dropna().unique())

needles = [
    'furia', 'nrg', 'big', 'berlin', 'brion', 'freecs', 'kwangdong',
    'levia', 't1 acad', 'okbri', 'ok br', 'dn ', 'lØs', 'giantx',
    'orbit', 'anonymo', 'otter', 'devils', 'kmt', 'tq durpa',
    'periquitos', 'skillcamp', 'secret', 'senshi', 'ucam', 'pcific',
    'one man',
]
for n in needles:
    hits = sorted(t for t in teams if n.lower() in t.lower())
    print(f'{n!r:25s} -> {hits[:8]}')
