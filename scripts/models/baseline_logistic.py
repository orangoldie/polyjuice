"""
Baseline logistic regression per checkpoint.

Diagnostic lower bound — gives a floor that XGBoost must clearly beat. If it
doesn't, investigate bug / leakage / feature set before further model work.

Trained on 2020–2024, scored on 2025 val. 2026 is NOT touched here (held out
for the final XGBoost evaluation). See docs/baseline-logistic.md.

Inputs:  data/processed/checkpoint_{10,15,20,25}.csv
Outputs: data/models/baseline_logistic/checkpoint_{T}/{model.pkl, metrics.json,
         coefficients.csv, reliability.png}
         data/experiments/runs.csv (appended)
"""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "models", "baseline_logistic")
RUNS_CSV = os.path.join(REPO_ROOT, "data", "experiments", "runs.csv")

CHECKPOINTS = (10, 15, 20, 25)
C_GRID = (0.01, 0.1, 1, 10, 100)
CV_SEED = 42

TIER_CATEGORIES = ["S", "A", "B"]  # S = baseline; dummies for A and B

FIRST_GATES = {
    10: ["firstblood"],
    15: ["firstblood", "firstdragon", "firsttower"],
    20: ["firstblood", "firstdragon", "firsttower"],
    25: ["firstblood", "firstdragon", "firsttower", "firstbaron"],
}

# Anything in X that is NOT in this set is numeric and gets standardized.
_NON_NUMERIC_FEATURE_NAMES = frozenset({
    "side_is_blue",
    "firstblood", "firstdragon", "firsttower", "firstbaron",
    "tier_A", "tier_B",
})


def NUMERIC_FEATURES(T: int) -> list[str]:
    return [
        f"goldat{T}", f"opp_goldat{T}",
        f"csat{T}", f"opp_csat{T}",
        f"killsat{T}", f"opp_killsat{T}",
        f"assistsat{T}", f"opp_assistsat{T}",
    ]


def BINARY_FEATURES(T: int) -> list[str]:
    return ["side_is_blue"] + FIRST_GATES[T]


def prepare_Xy(df: pd.DataFrame, T: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, feature_names) for checkpoint T.

    Drops the collinear *diffat{T} columns. tier is one-hot with S as baseline
    via explicit Categorical ordering (plain drop_first drops alphabetically
    first — A — which is not what we want).
    """
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES(T):
        X[col] = df[col].astype(float)

    X["side_is_blue"] = (df["side"] == "Blue").astype(int)
    for flag in FIRST_GATES[T]:
        X[flag] = df[flag].astype(int)

    tier = pd.Categorical(df["tier"], categories=TIER_CATEGORIES)
    tier_dummies = pd.get_dummies(tier, prefix="tier", drop_first=True).astype(int)
    tier_dummies.index = df.index
    for col in tier_dummies.columns:
        X[col] = tier_dummies[col]

    y = df["result"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, list(X.columns)


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """year ≤ 2024 → train; year == 2025 → val; year == 2026 → dropped."""
    train = df[df["year"] <= 2024].reset_index(drop=True)
    val = df[df["year"] == 2025].reset_index(drop=True)
    return train, val


def numeric_cols_in(X: pd.DataFrame) -> list[str]:
    """Single source of truth: numeric = any X column not in the non-numeric set.

    Used both at fit time (scaler + model training) and at predict time
    (re-applying the scaler), so the two pipelines can't drift.
    """
    return [c for c in X.columns if c not in _NON_NUMERIC_FEATURE_NAMES]


def fit(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[StandardScaler, LogisticRegressionCV]:
    """Fit StandardScaler on train numeric cols, then LogisticRegressionCV on the
    combined (scaled-numeric + passthrough-binary/tier) feature matrix.

    Internal 5-fold CV over C_GRID picks C; val is never touched.
    """
    numeric_cols = numeric_cols_in(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols].values)

    X_fit = X_train.copy()
    X_fit[numeric_cols] = scaler.transform(X_train[numeric_cols].values)

    model = LogisticRegressionCV(
        Cs=list(C_GRID),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED),
        penalty="l2",
        solver="lbfgs",
        scoring="neg_log_loss",
        max_iter=1000,
        refit=True,
    )
    model.fit(X_fit.values, y_train.values)
    return scaler, model


def predict_proba(scaler: StandardScaler, model: LogisticRegressionCV,
                  X: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    """Apply train-fit scaler to numeric cols, return P(result=1)."""
    X = X.copy()
    X[numeric_cols] = scaler.transform(X[numeric_cols].values)
    return model.predict_proba(X.values)[:, 1]


def evaluate(y_true, probs) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    return {
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probs)),
        "auc": auc,
        "accuracy": float((preds == y_true).mean()),
    }


def reliability_bins(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_pred": np.nan, "mean_actual": np.nan})
        else:
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                "mean_pred": float(probs[mask].mean()),
                "mean_actual": float(y_true[mask].mean()),
            })
    return pd.DataFrame(rows)


def _plot_reliability(bins: pd.DataFrame, T: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
    nonempty = bins[bins["n"] > 0]
    ax.plot(nonempty["mean_pred"], nonempty["mean_actual"], "o-", label=f"baseline T={T}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — baseline_logistic T={T}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_checkpoint(T: int,
                     input_dir: str = INPUT_DIR,
                     output_dir: str = OUTPUT_DIR,
                     runs_csv: str = RUNS_CSV,
                     run_id: str | None = None) -> dict:
    """Train baseline at checkpoint T, write all 4 artifacts, append runs.csv."""
    input_path = os.path.join(input_dir, f"checkpoint_{T}.csv")
    df = pd.read_csv(input_path, low_memory=False)

    train_df, val_df = split_train_val(df)
    X_train, y_train, feature_names = prepare_Xy(train_df, T)
    X_val, y_val, _ = prepare_Xy(val_df, T)

    scaler, model = fit(X_train, y_train)
    C_selected = float(model.C_[0])

    numeric_cols = numeric_cols_in(X_train)
    probs = predict_proba(scaler, model, X_val, numeric_cols)
    metrics = evaluate(y_val, probs)
    metrics["C_selected"] = C_selected
    metrics["train_n"] = int(len(X_train))
    metrics["val_n"] = int(len(X_val))

    ckpt_dir = os.path.join(output_dir, f"checkpoint_{T}")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "model.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "model": model,
                     "feature_names": feature_names,
                     "numeric_cols": numeric_cols}, f)

    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    coefs.to_csv(os.path.join(ckpt_dir, "coefficients.csv"), index=False)

    bins = reliability_bins(y_val, probs, n_bins=10)
    _plot_reliability(bins, T, os.path.join(ckpt_dir, "reliability.png"))

    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_baseline_logistic"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "baseline_logistic",
        "checkpoint": T,
        "hparams_json": json.dumps({
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "Cs": list(C_GRID),
            "cv": 5,
            "scoring": "neg_log_loss",
            "C_selected": C_selected,
        }),
        "train_n": metrics["train_n"],
        "val_n": metrics["val_n"],
        "val_logloss": metrics["log_loss"],
        "val_brier": metrics["brier"],
        "val_auc": metrics["auc"],
        "val_accuracy": metrics["accuracy"],
        "notes": "",
    }
    write_header = not os.path.exists(runs_csv)
    pd.DataFrame([row]).to_csv(runs_csv, mode="a", index=False, header=write_header)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ") + "_baseline_logistic"
    print(f"run_id = {run_id}")
    for T in CHECKPOINTS:
        print(f"\n=== checkpoint T={T} ===")
        m = train_checkpoint(T, args.input_dir, args.output_dir, args.runs_csv, run_id=run_id)
        print(f"  C_selected={m['C_selected']:g}  train_n={m['train_n']:,}  val_n={m['val_n']:,}")
        print(f"  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"auc={m['auc']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
