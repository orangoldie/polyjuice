"""Inventory unique Polymarket team shortcodes and cross-reference with OE teamnames.

Goal: see how many distinct shortcodes exist and how many resolve unambiguously
to OE teams without manual work.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from _tee import tee_stdout
tee_stdout('34_pmoe_shortcode_inventory_obsolete.txt')
import pandas as pd
from collections import Counter

mk = pd.read_csv(
    'data/polymarket/markets.csv',
    usecols=['event_id', 'fetch_status'],
)
ev = pd.read_csv(
    'data/polymarket/events.csv',
    usecols=['event_id', 'team_a_short', 'team_b_short', 'match_date'],
)
ok_event_ids = set(mk[mk['fetch_status'] == 'ok']['event_id'])
ev_ok = ev[ev['event_id'].isin(ok_event_ids)].copy()
print(f'events with >=1 ok market: {len(ev_ok)} / {len(ev)}')

shorts = Counter()
for _, r in ev_ok.iterrows():
    if isinstance(r['team_a_short'], str):
        shorts[r['team_a_short']] += 1
    if isinstance(r['team_b_short'], str):
        shorts[r['team_b_short']] += 1

print(f'unique shortcodes: {len(shorts)}')
print(f'total team-event slots: {sum(shorts.values())}')
print()
print('top 30 by event-frequency:')
for code, n in shorts.most_common(30):
    print(f'  {code:12s} {n:4d}')

print()
print('shortcodes appearing in only 1 event (long tail):')
singletons = [c for c, n in shorts.items() if n == 1]
print(f'  count: {len(singletons)}')
print(f'  sample: {singletons[:30]}')

# OE side: distinct team names from recent years
oe_teams = set()
for yr in [2024, 2025, 2026]:
    df = pd.read_csv(
        f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
        usecols=['teamname', 'position'],
        low_memory=False,
    )
    df = df[df['position'] == 'team']
    oe_teams.update(df['teamname'].dropna().unique())
print(f'\nOE distinct teamnames (2024-2026): {len(oe_teams)}')

# Naive auto-match: shortcode (lowercased) is substring of OE teamname's normalized form
def norm(s):
    return ''.join(c for c in s.lower() if c.isalnum())

oe_norm = {norm(t): t for t in oe_teams}

unambiguous = {}
ambiguous = {}
unmatched = []
for code in shorts:
    nc = norm(code)
    # exact match first
    if nc in oe_norm:
        unambiguous[code] = oe_norm[nc]
        continue
    # substring matches
    cands = [oe_norm[k] for k in oe_norm if nc in k or k in nc]
    if len(cands) == 1:
        unambiguous[code] = cands[0]
    elif len(cands) > 1:
        ambiguous[code] = cands
    else:
        unmatched.append(code)

print(f'\nauto-match results across {len(shorts)} shortcodes:')
print(f'  unambiguous: {len(unambiguous)}')
print(f'  ambiguous:   {len(ambiguous)}')
print(f'  unmatched:   {len(unmatched)}')

# Coverage of events: fraction where BOTH shortcodes auto-resolved
both_ok = 0
for _, r in ev_ok.iterrows():
    if isinstance(r['team_a_short'], str) and isinstance(r['team_b_short'], str):
        if r['team_a_short'] in unambiguous and r['team_b_short'] in unambiguous:
            both_ok += 1
print(f'\nevents where BOTH shortcodes auto-resolved: {both_ok} / {len(ev_ok)} '
      f'({100*both_ok/len(ev_ok):.1f}%)')

print('\nsample ambiguous (first 15):')
for code, cands in list(ambiguous.items())[:15]:
    print(f'  {code:10s} -> {cands[:5]}')

print('\nsample unmatched (first 30):')
print(' ', unmatched[:30])
