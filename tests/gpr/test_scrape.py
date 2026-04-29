"""Parser tests for scripts/gpr/scrape.

Pins the format-fragile path: extracting Apollo SSR transport pushes from
HTML, substituting bare `undefined` literals, and flattening teamGPR
records to snapshot rows. The lolesports SSR-cache embedding format can
change at any time and the first signal of breakage is a silent empty
`snapshots.csv`; these tests fail loudly instead.

Fixture: tests/fixtures/gpr/sample_page.html — see header there for the
edge cases it exercises.
"""
from pathlib import Path

import pytest

from scripts.gpr.scrape import (
    dedupe,
    extract_apollo_pushes,
    flatten_to_snapshots,
    iter_team_records,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gpr" / "sample_page.html"


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def payloads(html):
    return list(extract_apollo_pushes(html))


def test_extracts_both_pushes(payloads):
    assert len(payloads) == 2


def test_undefined_literal_substituted(payloads):
    # T_gamma's record carries `"nextFetchPolicy": undefined` in the source.
    # If the substitution failed, json.loads would have errored and the push
    # would be missing from `payloads` (the parser logs and skips). So the
    # presence of T_gamma in the parsed output is the assertion.
    teams = list(iter_team_records(payloads))
    ids = {t["id"] for t in teams}
    assert "tgpr_gamma" in ids


def test_both_payload_shapes_yield_records(payloads):
    # Push 1 uses the `rehydrate[<key>].data` shape; push 2 uses
    # `events[*].value.data`. iter_team_records must drain both.
    teams_per_push = [list(iter_team_records([p])) for p in payloads]
    assert all(len(t) > 0 for t in teams_per_push), (
        f"Each push must yield records; got per-push counts "
        f"{[len(t) for t in teams_per_push]}"
    )


def test_escaped_quote_in_team_name_preserved(payloads):
    teams = list(iter_team_records(payloads))
    beta = next(t for t in teams if t["id"] == "tgpr_beta")
    assert beta["team"]["name"] == 'Team "Beta" Gaming'


def test_flatten_emits_history_plus_current(payloads):
    teams = list(iter_team_records(payloads))
    alpha_only = [t for t in teams if t["id"] == "tgpr_alpha"]
    # Alpha appears in both pushes. From push 1 it has 2 history rows + 1
    # current; from push 2, 1 history row + 1 current. Pre-dedupe we expect
    # 5 raw rows for alpha across the whole fixture.
    rows = [r for r in flatten_to_snapshots(alpha_only)]
    assert len(rows) == 5
    # Every row carries the locked schema.
    expected_keys = {
        "team_id", "team_code", "team_name", "home_league_slug",
        "snapshot_date", "power_score", "team_elo", "rank",
    }
    for r in rows:
        assert set(r) == expected_keys


def test_snapshot_field_mapping(payloads):
    teams = list(iter_team_records(payloads))
    rows = list(flatten_to_snapshots(teams))
    feb10_alpha = next(
        r for r in rows
        if r["team_id"] == "team_alpha" and r["snapshot_date"] == "2024-02-10T04:00:00Z"
    )
    # power_score ← gprScore, team_elo ← elo, home_league_slug ← team.homeLeague.slug.
    # If Riot renames any of these on the wire, this is the test that fails first.
    assert feb10_alpha["power_score"] == 1500
    assert feb10_alpha["team_elo"] == 1480
    assert feb10_alpha["rank"] == 1
    assert feb10_alpha["team_code"] == "ALP"
    assert feb10_alpha["team_name"] == "Team Alpha"
    assert feb10_alpha["home_league_slug"] == "lck"


def test_dedupe_collapses_cross_push_duplicates(payloads):
    teams = list(iter_team_records(payloads))
    rows = list(flatten_to_snapshots(teams))
    deduped = dedupe(rows)
    keys = [(r["team_id"], r["snapshot_date"]) for r in deduped]
    assert len(keys) == len(set(keys)), "dedupe must produce unique (team_id, snapshot_date)"
    # Alpha's 2024-02-20 snapshot appears in both pushes with identical values
    # — must collapse to one row, not two.
    alpha_feb20 = [r for r in deduped if r["team_id"] == "team_alpha" and r["snapshot_date"] == "2024-02-20T04:00:00Z"]
    assert len(alpha_feb20) == 1


def test_dedupe_drops_rows_missing_keys():
    rows = [
        {"team_id": None, "snapshot_date": "2024-01-01", "power_score": 1, "team_elo": 1, "rank": 1},
        {"team_id": "x", "snapshot_date": None, "power_score": 1, "team_elo": 1, "rank": 1},
        {"team_id": "x", "snapshot_date": "2024-01-01", "power_score": 1, "team_elo": 1, "rank": 1},
    ]
    out = dedupe(rows)
    assert len(out) == 1
    assert out[0]["team_id"] == "x"


def test_dedupe_flags_value_conflict(capsys):
    rows = [
        {"team_id": "x", "snapshot_date": "2024-01-01", "power_score": 1500, "team_elo": 1480, "rank": 1},
        {"team_id": "x", "snapshot_date": "2024-01-01", "power_score": 9999, "team_elo": 1480, "rank": 1},
    ]
    dedupe(rows)
    captured = capsys.readouterr()
    assert "conflict" in captured.err
