"""
Orchestrator: run preprocessing stages 1 → 4 in order.

Each stage loads the prior stage's CSV output, transforms it, validates,
and writes its own CSV. Runs on real data by default:

  stage 1: data/lol-historical-data/*.csv  →  data/processed/01_filtered.csv
  stage 2: 01_filtered.csv                 →  data/processed/02_one_row.csv
  stage 3: 02_one_row.csv                  →  data/processed/03_features.csv
  stage 4: 03_features.csv                 →  data/processed/checkpoint_{10,15,20,25}.csv
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
