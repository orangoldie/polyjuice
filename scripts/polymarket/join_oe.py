"""Join Polymarket Game-N-Winner markets to OE gameids.

Per market (fetch_status=ok):
  1. Resolve outcome_0/outcome_1 via data/polymarket/team_aliases.csv -> OE teamname.
  2. Look up (date, frozenset({team_a, team_b}), game_number) in OE -> gameid.
  3. Bucket misses by reason.

Output:
  - data/polymarket/markets_oe_joined.csv  (one row per ok market, with gameid or miss_reason)
  - stdout summary of coverage and miss bucketing.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import Counter, defaultdict

import pandas as pd

OUT = 'data/polymarket/markets_oe_joined.csv'
OE_YEARS = [2022, 2023, 2024, 2025, 2026]


def main():
    aliases = pd.read_csv('data/polymarket/team_aliases.csv')
    pm_to_oe = dict(zip(aliases['pm_name'], aliases['oe_teamname']))
    print(f'alias map: {len(pm_to_oe)} pm names -> oe teamnames')

    mk = pd.read_csv(
        'data/polymarket/markets.csv',
        usecols=['condition_id', 'market_id', 'event_id', 'question',
                 'game_number', 'outcome_0', 'outcome_1', 'fetch_status'],
    )
    mk = mk[mk['fetch_status'] == 'ok'].copy()
    ev = pd.read_csv(
        'data/polymarket/events.csv',
        usecols=['event_id', 'event_slug', 'match_date'],
    )
    mk = mk.merge(ev, on='event_id', how='left')
    mk['match_date'] = pd.to_datetime(mk['match_date'], errors='coerce').dt.date
    print(f'ok markets: {len(mk)}')

    # Build OE index: (date_day, frozenset({team_a, team_b}), game_num) -> gameid
    oe_frames = []
    for yr in OE_YEARS:
        df = pd.read_csv(
            f'data/lol-historical-data/{yr}_LoL_esports_match_data_from_OraclesElixir.csv',
            usecols=['gameid', 'date', 'game', 'teamname', 'position', 'league'],
            low_memory=False,
        )
        df = df[df['position'] == 'team']
        oe_frames.append(df)
    oe = pd.concat(oe_frames, ignore_index=True)
    oe['date'] = pd.to_datetime(oe['date'], errors='coerce')
    oe['date_day'] = oe['date'].dt.date
    oe = oe.dropna(subset=['gameid', 'date_day', 'teamname'])
    print(f'OE team rows {OE_YEARS[0]}-{OE_YEARS[-1]}: {len(oe)}')

    # Each gameid should have 2 team rows. Group to (date_day, game_num, gameid, league, teams)
    grp = oe.groupby('gameid').agg(
        date_day=('date_day', 'first'),
        game_num=('game', 'first'),
        league=('league', 'first'),
        teams=('teamname', lambda s: frozenset(s.tolist())),
        n_teams=('teamname', 'nunique'),
    ).reset_index()
    grp = grp[grp['n_teams'] == 2].copy()
    print(f'OE games (2-team): {len(grp)}')

    oe_index = defaultdict(list)
    for _, r in grp.iterrows():
        try:
            gn = int(r['game_num'])
        except (ValueError, TypeError):
            continue
        oe_index[(r['date_day'], r['teams'], gn)].append((r['gameid'], r['league']))

    # Also build a date+pair index ignoring game_num for diagnostics
    oe_index_pair_only = defaultdict(list)
    for _, r in grp.iterrows():
        oe_index_pair_only[(r['date_day'], r['teams'])].append(
            (r['gameid'], int(r['game_num']) if pd.notna(r['game_num']) else -1, r['league'])
        )

    # Index by date+single team for diagnostics ("team-not-playing-that-day" miss)
    oe_team_dates = set()
    for _, r in oe.iterrows():
        oe_team_dates.add((r['date_day'], r['teamname']))

    rows_out = []
    miss_reasons = Counter()
    league_hits = Counter()

    for _, r in mk.iterrows():
        oa = pm_to_oe.get(r['outcome_0'])
        ob = pm_to_oe.get(r['outcome_1'])
        match_date = r['match_date']
        gn = int(r['game_number']) if pd.notna(r['game_number']) else None
        status, gameid, league, reason = 'ok', '', '', ''

        if oa is None or ob is None:
            status = 'miss'
            reason = 'team_unmapped'
            unm = []
            if oa is None:
                unm.append(r['outcome_0'])
            if ob is None:
                unm.append(r['outcome_1'])
            reason += ':' + '|'.join(str(x) for x in unm)
        elif match_date is None or gn is None:
            status = 'miss'
            reason = 'pm_missing_date_or_gamenum'
        else:
            key = (match_date, frozenset({oa, ob}), gn)
            hits = oe_index.get(key, [])
            if len(hits) == 1:
                gameid, league = hits[0]
            elif len(hits) > 1:
                status = 'miss'
                reason = f'oe_ambiguous_{len(hits)}'
            else:
                # Diagnose why no hit
                pair_hits = oe_index_pair_only.get((match_date, frozenset({oa, ob})), [])
                if pair_hits:
                    gnums = sorted({h[1] for h in pair_hits})
                    reason = f'oe_pair_present_but_game_num_off_have={gnums}_want={gn}'
                    status = 'miss'
                else:
                    a_played = (match_date, oa) in oe_team_dates
                    b_played = (match_date, ob) in oe_team_dates
                    if not a_played and not b_played:
                        reason = 'oe_neither_team_on_date'
                    elif not a_played:
                        reason = f'oe_missing_team:{oa}'
                    elif not b_played:
                        reason = f'oe_missing_team:{ob}'
                    else:
                        reason = 'oe_both_teams_played_but_not_each_other'
                    status = 'miss'

        if status == 'ok':
            league_hits[league] += 1
        else:
            miss_reasons[reason.split(':')[0]] += 1

        rows_out.append({
            'condition_id': r['condition_id'],
            'event_id': r['event_id'],
            'event_slug': r['event_slug'],
            'match_date': str(match_date) if match_date else '',
            'game_number': gn if gn is not None else '',
            'pm_outcome_0': r['outcome_0'],
            'pm_outcome_1': r['outcome_1'],
            'oe_team_a': oa or '',
            'oe_team_b': ob or '',
            'oe_gameid': gameid,
            'oe_league': league,
            'status': status,
            'miss_reason': reason,
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT, index=False)

    n_ok = (out_df['status'] == 'ok').sum()
    n_miss = (out_df['status'] == 'miss').sum()
    print(f'\nwrote {OUT}')
    print(f'  ok:   {n_ok} / {len(out_df)} ({100*n_ok/len(out_df):.1f}%)')
    print(f'  miss: {n_miss}')
    print('\nmiss buckets:')
    for k, v in miss_reasons.most_common():
        print(f'  {k:50s} {v}')
    print('\ntop hit leagues:')
    for k, v in league_hits.most_common(15):
        print(f'  {k:10s} {v}')

    # Event-level coverage: an event is "covered" if all its ok markets joined
    ev_status = out_df.groupby('event_id')['status'].agg(
        lambda s: 'ok' if (s == 'ok').all() else ('partial' if (s == 'ok').any() else 'none')
    )
    print('\nevent-level coverage (across markets within event):')
    print(ev_status.value_counts().to_string())


if __name__ == '__main__':
    main()
