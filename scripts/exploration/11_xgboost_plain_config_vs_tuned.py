"""Sanity check: does Optuna tuning add anything over a plain XGB config?

For each checkpoint, fit a couple of fixed-config XGBClassifiers on the same
train + early_stop_val slice the tuned model uses for its post-Optuna refit
(2020..2024-12-31). Score raw predictions on:

- optuna_val (2025-H1) — the slice the tuned model is optimized for
- acceptance_val (2025-H2) — the honest gate

Compare against:

- baseline_logistic H2 (the acceptance threshold, from BASELINE_H2)
- tuned xgboost_v1 H2 (read from checkpoint_{T}/metrics.json)

Interpretation — if plain configs match the tuned model on H1 (Optuna's own
slice), tuning isn't buying nonlinear structure that a simple config misses.
Evidence for "features are mostly linear; XGB can't pull ahead of logistic."
"""
import json
import os
import pickle

import pandas as pd
import xgboost as xgb

from scripts.models.xgboost_v1 import (
    BASELINE_H2,
    CHECKPOINTS,
    INPUT_DIR,
    OUTPUT_DIR,
    _xgb_base_params,
    evaluate,
    prepare_Xy,
    split_temporal,
)


PLAIN_CONFIGS = {
    "shallow": dict(max_depth=4, learning_rate=0.1,  n_estimators=300),
    "medium":  dict(max_depth=6, learning_rate=0.05, n_estimators=500),
    "deep":    dict(max_depth=8, learning_rate=0.03, n_estimators=800),
}


def score_config(T: int, config: dict, splits: dict) -> tuple[dict, dict]:
    train = pd.concat([splits["train"], splits["early_stop_val"]], ignore_index=True)
    X_tr, y_tr, _ = prepare_Xy(train, T)
    X_h1, y_h1, _ = prepare_Xy(splits["optuna_val"], T)
    X_h2, y_h2, _ = prepare_Xy(splits["acceptance_val"], T)
    model = xgb.XGBClassifier(**_xgb_base_params(), **config)
    model.fit(X_tr, y_tr, verbose=False)
    h1 = evaluate(y_h1.values, model.predict_proba(X_h1)[:, 1])
    h2 = evaluate(y_h2.values, model.predict_proba(X_h2)[:, 1])
    return h1, h2


def _read_tuned(T: int) -> dict | None:
    path = os.path.join(OUTPUT_DIR, f"checkpoint_{T}", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    rows = []
    for T in CHECKPOINTS:
        df = pd.read_csv(os.path.join(INPUT_DIR, f"checkpoint_{T}.csv"), low_memory=False)
        splits = split_temporal(df)

        for name, cfg in PLAIN_CONFIGS.items():
            h1, h2 = score_config(T, cfg, splits)
            rows.append({
                "T": T, "source": f"plain/{name}",
                "H1_logloss": h1["log_loss"], "H2_logloss": h2["log_loss"],
                "H2_brier": h2["brier"],
            })

        tuned = _read_tuned(T)
        if tuned is not None:
            study_path = os.path.join(OUTPUT_DIR, f"checkpoint_{T}", "study.pkl")
            with open(study_path, "rb") as f:
                study = pickle.load(f)
            rows.append({
                "T": T, "source": "tuned/xgboost_v1",
                "H1_logloss": float(study.best_value),
                "H2_logloss": tuned["raw"]["log_loss"],
                "H2_brier":   tuned["raw"]["brier"],
            })
        rows.append({
            "T": T, "source": "baseline_logistic",
            "H1_logloss": float("nan"),
            "H2_logloss": BASELINE_H2[T]["logloss"],
            "H2_brier":   BASELINE_H2[T]["brier"],
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False,
                        formatters={"H1_logloss": "{:.4f}".format,
                                    "H2_logloss": "{:.4f}".format,
                                    "H2_brier":   "{:.4f}".format}))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = os.path.join(repo_root, "data", "exploration",
                            "11_xgboost_plain_config_vs_tuned.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
