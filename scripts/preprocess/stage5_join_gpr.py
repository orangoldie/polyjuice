"""
Stage 5: per-game cutoff join of GPR snapshots onto each per-checkpoint frame.

For each row in `checkpoint_{T}.csv` (one row per (game, team-perspective)),
attach the most recent GPR snapshot for that team and its opponent with
`snapshot_date < game_date`. Strict less-than enforces leakage prevention by
construction — a published rating dated D cannot encode the outcome of any
game played at or after D.

Emits four new columns on top of the stage-4 schema:
  - gpr_team:      power_score for `teamname`     (0 when has_gpr=0)
  - gpr_opp_team:  power_score for `opp_teamname` (0 when has_gpr=0)
  - gpr_diff:      gpr_team - gpr_opp_team        (0 when has_gpr=0)
  - has_gpr:       1 iff both teams have a qualifying snapshot, else 0

Both teams must have a qualifying snapshot for has_gpr=1; otherwise the
GPR columns are zeroed (so the row stays in training but its has_gpr=0
coefficient absorbs the population shift, per the spec).

Date handling. OE `date` is a naive UTC datetime ("YYYY-MM-DD HH:MM:SS")
per OE convention; Riot `dateCalculated` is ISO-8601 with explicit zone
(typically Z / UTC). Both are normalized to tz-aware UTC `pd.Timestamp`
before the strict-`<` comparison. A naive-vs-aware compare would raise; a
both-naive compare with mismatched implicit zones could silently introduce
~1 day of phantom leakage near regional broadcast windows.

Input:  data/processed/checkpoint_{T}.csv
        data/gpr/snapshots.csv
        data/gpr/team_aliases.csv
Output: data/processed/checkpoint_{T}_gpr.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "gpr" / "snapshots.csv"
DEFAULT_ALIASES = REPO_ROOT / "data" / "gpr" / "team_aliases.csv"

CHECKPOINTS = (10, 15, 20, 25)

NEW_COLUMNS = ["gpr_team", "gpr_opp_team", "gpr_diff", "has_gpr"]


def load_aliases(path: Path) -> dict[str, str]:
    """Return oe_teamname -> gpr_team_name. Excludes uncovered (empty) rows.

    The OE-side name is the lookup key. Any OE team without an alias entry
    or with an UNCOVERED entry will fail the lookup and yield has_gpr=0.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        oe = r["oe_teamname"]
        gpr = r["gpr_team_name"]
        if not oe or not gpr:
            continue
        out[oe] = gpr
    return out


def load_snapshots(path: Path) -> pd.DataFrame:
    """Load snapshots and normalize snapshot_date to tz-aware UTC."""
    df = pd.read_csv(path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)
    if df["snapshot_date"].isna().any():
        raise ValueError("snapshots.csv has unparseable snapshot_date values")
    return df.sort_values("snapshot_date").reset_index(drop=True)


def _merge_side(
    df: pd.DataFrame,
    snapshots: pd.DataFrame,
    side: str,
    aliases: dict[str, str],
) -> pd.Series:
    """Attach `power_score` for one side's team via per-game cutoff merge.

    `side` is 'team' or 'opp_team'. The OE column is `teamname` /
    `opp_teamname`; the output is a power_score Series aligned to df's index,
    NaN where no qualifying snapshot exists.
    """
    oe_col = "teamname" if side == "team" else "opp_teamname"
    gpr_key = df[oe_col].map(aliases)

    # Short-circuit on empty snapshots — merge_asof's by= dtype check
    # mishandles all-NaN object columns, and an empty join is trivially NaN.
    if len(snapshots) == 0:
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Build a left frame for merge_asof: index, date, gpr_key.
    # Don't use .values on tz-aware Series — that strips the tz to naive
    # ns and merge_asof then refuses to compare with the tz-aware right.
    # Force _gpr_key to object dtype — when no OE name resolves (e.g. empty
    # alias table), df.map() returns all-NaN float64, which merge_asof's by=
    # then refuses to compare against the right's string-typed team_name.
    left = pd.DataFrame({
        "_orig_idx": df.index.to_numpy(),
        "_date": df["_date_utc"].reset_index(drop=True),
        "_gpr_key": gpr_key.reset_index(drop=True).astype(object),
    })
    # merge_asof requires the left key sorted; preserve mapping back via _orig_idx.
    left = left.sort_values("_date", kind="mergesort").reset_index(drop=True)

    right = snapshots[["team_name", "snapshot_date", "power_score"]].rename(
        columns={"team_name": "_gpr_key", "snapshot_date": "_date"}
    )
    # Coerce both _gpr_key dtypes to object — pandas may infer StringDtype
    # for one side and object for the other depending on how each frame was
    # constructed, and merge_asof's by= refuses to mix them.
    right = right.assign(_gpr_key=right["_gpr_key"].astype(object))

    merged = pd.merge_asof(
        left,
        right,
        on="_date",
        by="_gpr_key",
        direction="backward",
        allow_exact_matches=False,  # strict <
    )

    # Restore original df row order.
    out = pd.Series(index=df.index, dtype=float)
    out.loc[merged["_orig_idx"].values] = merged["power_score"].values
    return out


def run(
    checkpoint_df: pd.DataFrame,
    snapshots: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """Apply the per-game cutoff join and append the four GPR columns."""
    df = checkpoint_df.copy()

    # Normalize game date to tz-aware UTC. OE writes naive UTC.
    parsed = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("stage4 checkpoint frame has unparseable `date` values")
    df["_date_utc"] = parsed

    ps_team = _merge_side(df, snapshots, "team", aliases)
    ps_opp = _merge_side(df, snapshots, "opp_team", aliases)

    has_gpr = (~ps_team.isna() & ~ps_opp.isna()).astype(np.int8)
    gpr_team = ps_team.where(has_gpr == 1, 0.0).astype(float)
    gpr_opp = ps_opp.where(has_gpr == 1, 0.0).astype(float)
    gpr_diff = gpr_team - gpr_opp

    df["gpr_team"] = gpr_team
    df["gpr_opp_team"] = gpr_opp
    df["gpr_diff"] = gpr_diff
    df["has_gpr"] = has_gpr

    df = df.drop(columns=["_date_utc"])
    return df


def validate(df: pd.DataFrame) -> None:
    """Schema + invariants on the stage-5 output."""
    for c in NEW_COLUMNS:
        assert c in df.columns, f"missing {c}"
        assert df[c].notna().all(), f"null {c} present"

    assert df["has_gpr"].isin([0, 1]).all(), "has_gpr outside {0, 1}"

    # Mirror stage 3's *diff invariant: gpr_diff == gpr_team - gpr_opp_team.
    # Use exact equality — both sides are computed from the same float
    # operands in run(), so no rounding noise should accumulate.
    diff_mismatch = (df["gpr_diff"] != (df["gpr_team"] - df["gpr_opp_team"])).sum()
    assert diff_mismatch == 0, f"{diff_mismatch} rows violate gpr_diff invariant"

    # Unrated rows must be zeroed across all three numeric GPR columns.
    unrated = df[df["has_gpr"] == 0]
    for c in ("gpr_team", "gpr_opp_team", "gpr_diff"):
        bad = (unrated[c] != 0).sum()
        assert bad == 0, f"{bad} has_gpr=0 rows have nonzero {c}"


def coverage_summary(df: pd.DataFrame) -> dict:
    """Diagnostic-only — print how many rows ended up rated."""
    n = len(df)
    rated = int(df["has_gpr"].sum())
    by_year = df.groupby("year")["has_gpr"].agg(["sum", "count"])
    by_year["rate"] = by_year["sum"] / by_year["count"]
    return {
        "rows": n,
        "rated": rated,
        "rated_pct": 100.0 * rated / n if n else 0.0,
        "by_year": by_year,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    p.add_argument("--aliases", default=str(DEFAULT_ALIASES))
    args = p.parse_args()

    print("Loading GPR snapshots + aliases...")
    snapshots = load_snapshots(Path(args.snapshots))
    aliases = load_aliases(Path(args.aliases))
    print(f"  snapshots: {len(snapshots):,} rows, {snapshots['team_name'].nunique()} teams")
    print(f"  aliases:   {len(aliases)} OE-to-GPR mappings (uncovered rows excluded)")

    os.makedirs(args.output_dir, exist_ok=True)
    for T in CHECKPOINTS:
        in_path = os.path.join(args.input_dir, f"checkpoint_{T}.csv")
        out_path = os.path.join(args.output_dir, f"checkpoint_{T}_gpr.csv")
        print(f"Checkpoint T={T}...")
        df = pd.read_csv(in_path, low_memory=False)
        joined = run(df, snapshots, aliases)
        validate(joined)
        summary = coverage_summary(joined)
        print(f"  rows={summary['rows']:,}  rated={summary['rated']:,} ({summary['rated_pct']:.1f}%)")
        joined.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
