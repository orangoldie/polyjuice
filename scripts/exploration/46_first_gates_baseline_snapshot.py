"""Snapshot model metrics across all 4 models × 4 checkpoints into one JSON.

Run before the FIRST_GATES change (writes _baseline.json) and again after
retrain (writes _after.json) so we can diff. Reads what's on disk — does
not retrain.

For each (model, checkpoint), captures:
  - the training-time validation metrics from `checkpoint_{T}/metrics.json`
  - the held-out 2026 production metrics from `production/checkpoint_{T}/metrics_2026.json`
    (only present for baseline_logistic and baseline_logistic_gpr)

Usage:
  python scripts/exploration/46_first_gates_baseline_snapshot.py            # writes _baseline.json
  python scripts/exploration/46_first_gates_baseline_snapshot.py after      # writes _after.json
"""
import json
import sys
from pathlib import Path

from _tee import tee_stdout

MODELS = ['baseline_logistic', 'baseline_logistic_gpr', 'xgboost_v1', 'xgboost_v1_gpr']
CHECKPOINTS = [10, 15, 20, 25]
ROOT = Path('data/models')
OUT_DIR = Path('data/exploration')

# Fields we care about for diffing. Order = display order.
KEY_FIELDS = ['log_loss', 'brier', 'auc', 'accuracy', 'ece', 'n']


def _flatten_train_metrics(d: dict) -> dict:
    """Pull the 'best' set of headline metrics from a training-time metrics.json.

    Different models nest things differently:
      - baseline_logistic:     top-level {log_loss, brier, auc, accuracy}
      - baseline_logistic_gpr: top-level {log_loss, brier, auc, accuracy}
      - xgboost_v1 / _gpr:     {raw: {...}, cal: {...}}; prefer 'cal' if present.
    """
    if 'cal' in d and isinstance(d['cal'], dict):
        return {k: d['cal'].get(k) for k in KEY_FIELDS}
    if 'raw' in d and isinstance(d['raw'], dict):
        return {k: d['raw'].get(k) for k in KEY_FIELDS}
    return {k: d.get(k) for k in KEY_FIELDS}


def _flatten_prod_metrics(d: dict) -> dict:
    """Production metrics_2026.json schema varies by model:
      - baseline_logistic_gpr nests under `overall.{raw,calibrated}` (with rated/unrated splits).
      - baseline_logistic     puts `raw` / `calibrated` at the top level.
    Prefer 'calibrated' (production-served) over 'raw'."""
    container = d.get('overall') if isinstance(d.get('overall'), dict) else d
    for key in ('calibrated', 'raw'):
        sub = container.get(key)
        if isinstance(sub, dict):
            out = {k: sub.get(k) for k in KEY_FIELDS}
            if out.get('n') is None:
                out['n'] = d.get('test_n')  # n lives at top level for non-gpr baseline
            return out
    return {k: d.get(k) for k in KEY_FIELDS}


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    out_path = OUT_DIR / f'46_first_gates_change_{suffix}.json'
    tee_stdout(f'46_first_gates_baseline_snapshot_{suffix}.txt')

    snapshot: dict = {}
    for model in MODELS:
        snapshot[model] = {}
        for T in CHECKPOINTS:
            entry: dict = {'train': None, 'production': None}

            train_path = ROOT / model / f'checkpoint_{T}' / 'metrics.json'
            if train_path.exists():
                with train_path.open('r', encoding='utf-8') as f:
                    entry['train'] = _flatten_train_metrics(json.load(f))
            else:
                print(f'WARN missing {train_path}')

            prod_path = ROOT / model / 'production' / f'checkpoint_{T}' / 'metrics_2026.json'
            if prod_path.exists():
                with prod_path.open('r', encoding='utf-8') as f:
                    entry['production'] = _flatten_prod_metrics(json.load(f))

            snapshot[model][f'T{T}'] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)
    print(f'wrote {out_path}')

    # Print compact human-readable summary
    print()
    for view in ('train', 'production'):
        print(f'\n=== {view.upper()} metrics (Brier / log_loss / acc) ===')
        header = f'  {"model":26s} ' + ''.join(f'  {f"T={T}":>22s}' for T in CHECKPOINTS)
        print(header)
        for model in MODELS:
            cells = []
            for T in CHECKPOINTS:
                m = snapshot[model][f'T{T}'].get(view)
                if m is None or m.get('brier') is None:
                    cells.append(f'  {"--":>22s}')
                else:
                    cells.append(f'  {m["brier"]:>6.4f}/{m["log_loss"]:>6.4f}/{m.get("accuracy", 0):>5.3f}')
            print(f'  {model:26s} ' + ''.join(cells))


if __name__ == '__main__':
    main()
