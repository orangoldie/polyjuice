"""
Orchestrator: run preprocessing stages 1 → 5 in order.

Each stage loads the prior stage's CSV output, transforms it, validates,
and writes its own CSV. Runs on real data by default:

  stage 1: data/lol-historical-data/*.csv  →  data/processed/01_filtered.csv
  stage 2: 01_filtered.csv                 →  data/processed/02_one_row.csv
  stage 3: 02_one_row.csv                  →  data/processed/03_features.csv
  stage 4: 03_features.csv                 →  data/processed/checkpoint_{10,15,20,25}.csv
  stage 5: checkpoint_{T}.csv + data/gpr/  →  data/processed/checkpoint_{T}_gpr.csv

Stage 5 reads `data/gpr/snapshots.csv` and `data/gpr/team_aliases.csv`. The
scrape that produces snapshots.csv is a separate, manually-invoked step
(`python -m scripts.gpr.scrape`) and is intentionally NOT part of this
orchestrator — the page-scrape mirrors how raw OE CSVs are also refreshed
out-of-band, and keeps build_dataset.py purely offline + deterministic.
"""
import os
import subprocess
import sys
from pathlib import Path

STAGES = [
    "scripts.preprocess.stage1_load_and_filter",
    "scripts.preprocess.stage2_one_row_per_game",
    "scripts.preprocess.stage3_features",
    "scripts.preprocess.stage4_split_checkpoints",
    "scripts.preprocess.stage5_join_gpr",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    for stage in STAGES:
        print(f"\n=== {stage} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", stage],
            cwd=repo_root,
            env=env,
        )
        if result.returncode != 0:
            sys.exit(f"{stage} failed with exit code {result.returncode}")
    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
