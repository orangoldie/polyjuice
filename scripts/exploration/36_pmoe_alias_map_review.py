"""Build a Polymarket-team -> OE-teamname alias map for review.

Strategy:
  1. Pull unique team names from markets.csv outcome_0/outcome_1.
  2. Pull OE teamnames from 2024-2026 match data.
  3. Exact normalized match (lowercase + alnum only) -> auto.
  4. For misses, propose top-3 fuzzy candidates by SequenceMatcher ratio,
     restricted to OE teams that actually played within the same date range
     as that Polymarket name's events (cuts down on look-alike academy teams).
  5. Write TSV to data/exploration/36_polymarket_oe_alias_map.tsv for review.

Reviewer fills in `oe_teamname_final` column (or sets to empty if no match).
Downstream join script reads that column.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import csv
from difflib import SequenceMatcher

import pandas as pd

OUT = 'data/exploration/36_polymarket_oe_alias_map.tsv'


def norm(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())


def main():
    mk = pd.read_csv(
        'data/polymarket/markets.csv',
        usecols=['outcome_0', 'outcome_1', 'event_id', 'fetch_status'],
    )
    mk = mk[mk['fetch_status'] == 'ok']
    ev = pd.read_csv(
        'data/polymarket/events.csv',
        usecols=['event_id', 'match_date'],
    )
    mk = mk.merge(ev, on='event_id', how='left')
    mk['match_date'] = pd.to_datetime(mk['match_date'], errors='coerce')

    # Long format: one row per (pm_name, event)
    a = mk[['outcome_0', 'event_id', 'match_date']].rename(columns={'outcome_0': 'pm_name'})
    b = mk[['outcome_1', 'event_id', 'match_date']].rename(columns={'outcome_1': 'pm_name'})
    long = pd.concat([a, b], ignore_index=True).dropna(subset=['pm_name'])

    pm_stats = long.groupby('pm_name').agg(
        n_events=('event_id', 'nunique'),
        first_date=('match_date', 'min'),
        last_date=('match_date', 'max'),
    ).reset_index()
    print(f'unique Polymarket team names: {len(pm_stats)}')

    # OE teams + their active dates per teamname
    oe_frames = []
    for yr in [2023, 2024, 2025, 2026]:
        df = pd.read_csv(
            f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
            usecols=['teamname', 'date', 'position'],
            low_memory=False,
        )
        df = df[df['position'] == 'team']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        oe_frames.append(df[['teamname', 'date']])
    oe = pd.concat(oe_frames, ignore_index=True).dropna()
    oe_teamnames = oe['teamname'].unique().tolist()
    print(f'OE distinct teamnames (2023-2026): {len(oe_teamnames)}')

    oe_norm_to_name = {}
    for t in oe_teamnames:
        oe_norm_to_name.setdefault(norm(t), t)

    # OE active-window per teamname
    oe_window = oe.groupby('teamname').agg(
        oe_first=('date', 'min'), oe_last=('date', 'max'),
    ).to_dict('index')

    rows = []
    for _, r in pm_stats.iterrows():
        pm_name = r['pm_name']
        n = norm(pm_name)
        exact = oe_norm_to_name.get(n)
        proposed, alts, conf = '', '', ''
        if exact:
            proposed = exact
            conf = 'exact'
        else:
            # Restrict candidates to OE teams active in overlapping window
            pm_first = r['first_date']
            pm_last = r['last_date']
            cands = []
            for t in oe_teamnames:
                w = oe_window.get(t)
                if w is None:
                    continue
                if pd.notna(pm_first) and pd.notna(pm_last):
                    if w['oe_last'] < pm_first or w['oe_first'] > pm_last:
                        continue
                ratio = SequenceMatcher(None, n, norm(t)).ratio()
                cands.append((ratio, t))
            cands.sort(reverse=True)
            top = cands[:5]
            if top:
                proposed = top[0][1]
                conf = f'fuzzy_{top[0][0]:.2f}'
                alts = ' | '.join(f'{t} ({r:.2f})' for r, t in top[1:5])
            else:
                conf = 'no_candidates'

        rows.append({
            'pm_name': pm_name,
            'n_events': int(r['n_events']),
            'first_date': r['first_date'].strftime('%Y-%m-%d') if pd.notna(r['first_date']) else '',
            'last_date': r['last_date'].strftime('%Y-%m-%d') if pd.notna(r['last_date']) else '',
            'oe_proposed': proposed,
            'confidence': conf,
            'oe_teamname_final': proposed if conf == 'exact' else '',
            'alt_candidates': alts,
        })

    rows.sort(key=lambda x: (x['confidence'] != 'exact', -x['n_events'], x['pm_name']))

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(rows)

    n_exact = sum(1 for r in rows if r['confidence'] == 'exact')
    n_fuzzy = sum(1 for r in rows if r['confidence'].startswith('fuzzy'))
    n_none = sum(1 for r in rows if r['confidence'] == 'no_candidates')
    print(f'wrote {OUT}')
    print(f'  exact:   {n_exact}')
    print(f'  fuzzy:   {n_fuzzy}  (need review)')
    print(f'  no cand: {n_none}')

    # Spot-print the fuzzy ones for quick review
    print('\nfuzzy proposals (review these):')
    print(f'{"pm_name":35s} {"n":>4s}  {"oe_proposed":35s} {"conf":12s} alts')
    for r in rows:
        if r['confidence'].startswith('fuzzy') or r['confidence'] == 'no_candidates':
            print(f'{r["pm_name"]:35s} {r["n_events"]:>4d}  {r["oe_proposed"]:35s} {r["confidence"]:12s} {r["alt_candidates"]}')


if __name__ == '__main__':
    main()
