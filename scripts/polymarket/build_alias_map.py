"""Write the curated Polymarket -> OE team alias map.

Combines:
  - All exact normalized matches from markets.csv outcomes (198 names).
  - 16 manually-resolved fuzzy mappings.
  - 4 names dropped intentionally (uncertain — left out of the map so they
    will simply fail to join downstream and surface as unmatched).

Output: data/polymarket/team_aliases.csv with columns (pm_name, oe_teamname).
This is the authoritative file the join script consumes.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import csv
import pandas as pd

OUT = 'data/polymarket/team_aliases.csv'

MANUAL_OVERRIDES = {
    'T1 Academy': 'T1 Esports Academy',
    'FURIA Esports': 'FURIA',
    'Leviatan Esports': 'Leviatan',
    'LOS': 'LØS',
    'PCIFIC ': 'PCIFIC Esports',
    'The Secret Club Esport': 'The Secret Club',
    'NRG Esports': 'NRG',
    'Senshi Esports Club': 'Senshi eSports',
    'UCAM Esports Club': 'UCAM Esports',
    'The Otter Side': 'Otter Side',
    'Orbit Anonymo': 'Orbit Anonymo Esports',
    'BIG': 'Berlin International Gaming',
    'TQ DURPA KASHLQ': 'TQ DURPA KASLQ',
    'A One Man Army Esports Prime': 'A One Man Army Prime',
    'Periquitos Zangados': 'Periquitos Zangados Gaming',
    'Skillcamp Esport': 'Skillcamp',
}

DROPPED = {'DN Freecs', 'OKSavingsBank BRION', 'GIANTX PRIDE', 'devils.one x KMT'}


def norm(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())


def main():
    mk = pd.read_csv(
        'data/polymarket/markets.csv',
        usecols=['outcome_0', 'outcome_1', 'fetch_status'],
    )
    mk = mk[mk['fetch_status'] == 'ok']
    pm_names = set(mk['outcome_0'].dropna()).union(set(mk['outcome_1'].dropna()))

    oe_teams = set()
    for yr in [2023, 2024, 2025, 2026]:
        df = pd.read_csv(
            f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
            usecols=['teamname', 'position'], low_memory=False,
        )
        df = df[df['position'] == 'team']
        oe_teams.update(df['teamname'].dropna().unique())
    oe_norm_to_name = {}
    for t in oe_teams:
        oe_norm_to_name.setdefault(norm(t), t)

    rows = []
    n_exact = n_manual = n_dropped = n_unresolved = 0
    for pm in sorted(pm_names):
        if pm in DROPPED:
            n_dropped += 1
            continue
        if pm in MANUAL_OVERRIDES:
            rows.append({'pm_name': pm, 'oe_teamname': MANUAL_OVERRIDES[pm], 'source': 'manual'})
            n_manual += 1
            continue
        oe = oe_norm_to_name.get(norm(pm))
        if oe is not None:
            rows.append({'pm_name': pm, 'oe_teamname': oe, 'source': 'exact'})
            n_exact += 1
        else:
            n_unresolved += 1
            print(f'WARN unresolved: {pm!r}')

    rows.sort(key=lambda r: r['pm_name'])
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['pm_name', 'oe_teamname', 'source'])
        w.writeheader()
        w.writerows(rows)

    print(f'\nwrote {OUT}')
    print(f'  exact:        {n_exact}')
    print(f'  manual:       {n_manual}')
    print(f'  dropped:      {n_dropped}  (intentionally unmapped)')
    print(f'  unresolved:   {n_unresolved}  (should be 0)')
    print(f'  total mapped: {len(rows)} / {len(pm_names)} pm names')


if __name__ == '__main__':
    main()
