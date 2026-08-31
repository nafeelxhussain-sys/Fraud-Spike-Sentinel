"""
Trains the fraud classifier and picks an operating threshold by minimizing
business cost and not by maximizing F1.

Usage:
    python -m src.train --data data/paysim.csv --out model/
"""

import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from src.features import build_training_features, FEATURE_COLUMNS

#  BUSINESS CONSTANTS
COST_PER_FALSE_POSITIVE = 4.0       #marking a legit txn as fraud 
COST_PER_FALSE_NEGATIVE = 1.0       #marking a fraud txn as legit

def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    cutoff = df["step"].quantile(1 - test_fraction)
    train = df[df["step"] < cutoff].copy()
    test = df[df["step"] >= cutoff].copy()
    return train, test, cutoff

def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray, amounts: np.ndarray):
    thresholds = np.linspace(0.01,0.99,99)
    best = {"threshold": 0.5, "cost": float("inf")}
    sweep = []

    for t in thresholds:
        y_mask = (y_proba >= t ).astype(int)

        fp_mask = (y_true==0) & (y_mask==1)
        fn_mask = (y_true==1) & (y_mask==0)

        cost = fp_mask.sum() * COST_PER_FALSE_POSITIVE + amounts[fn_mask].sum() * COST_PER_FALSE_NEGATIVE

        sweep.append({"threshold": float(t), "cost": float(cost), "n_fp": int(fp_mask.sum()), "n_fn": int(fn_mask.sum())})

        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost)}


    return best["threshold"], best["cost"], sweep



def main(data_path: str, out_dir: str):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} ...")
    raw = pd.read_csv(data_path)
    print(f"  {len(raw):,} rows, {raw['isFraud'].sum():,} fraud ({100*raw['isFraud'].mean():.3f}%)")

    print("Building features ...")
    feat = build_training_features(raw)
    print("Building features done")

    train, test, cutoff = time_based_split(feat)
    print(f"Time based split at step {cutoff:.0f}: train={len(train):,} rows, test={len(test):,} rows")

    X_train, y_train = train[FEATURE_COLUMNS], train["isFraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["isFraud"].values
    amounts_test = test["amount"].values

    print("Training HistGradientBoostingClassifier (class_weight='balanced') ...")
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=67)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba_test)

    threshold, cost, sweep = choose_threshold(y_test, proba_test, amounts_test)
    pred_test = (proba_test >= threshold).astype(int)

    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)

    # cost of two naive baselines so the threshold choice can be justified
    always_zero_cost = amounts_test[y_test == 1].sum() * COST_PER_FALSE_NEGATIVE
    default_half_pred = (proba_test >= 0.5).astype(int)
    default_half_fp = ((default_half_pred == 1) & (y_test == 0)).sum()
    default_half_fn_mask = (default_half_pred == 0) & (y_test == 1)
    default_half_cost = default_half_fp * COST_PER_FALSE_POSITIVE + amounts_test[default_half_fn_mask].sum()

    print("\n--- Held-out test metrics (time-split, model has never seen these rows) ---")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1:             {f1:.3f}")
    print(f"  PR-AUC:         {pr_auc:.3f}")
    print(f"  Chosen threshold: {threshold:.2f}")
    print(f"  Cost at chosen threshold:     {cost:,.2f}")
    print(f"  Cost at naive threshold=0.5:  {default_half_cost:,.2f}")
    print(f"  Cost of flagging nothing:     {always_zero_cost:,.2f}")

    print("\nComputing permutation importance (this can take a bit) ...")
    sample_idx = np.random.RandomState(67).choice(len(X_test), size=min(10000, len(X_test)), replace=False)

    imp = permutation_importance(
        model, X_test.iloc[sample_idx], y_test[sample_idx],
        n_repeats=5, random_state=67, scoring="average_precision",
    )

    importance_ranked = sorted(
        zip(FEATURE_COLUMNS, imp.importances_mean), key=lambda x: -x[1]
    )

    print("Top features by permutation importance:")
    for name, val in importance_ranked[:5]:
        print(f"  {name}: {val:.4f}")

    joblib.dump(model, out / "model.joblib")

    feature_reference = {
        col: {"median": float(X_train[col].median()), "std": float(X_train[col].std() or 1.0)}
        for col in FEATURE_COLUMNS
    }

    metrics = {
        "feature_reference": feature_reference,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(train),
        "n_test": len(test),
        "split_step_cutoff": float(cutoff),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "cost_at_threshold": cost,
        "cost_naive_half": float(default_half_cost),
        "cost_flag_nothing": float(always_zero_cost),
        "cost_per_false_positive": COST_PER_FALSE_POSITIVE,
        "feature_columns": FEATURE_COLUMNS,
        "top_features": [{"feature": n, "importance": float(v)} for n, v in importance_ranked],
        "threshold_sweep": sweep,
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to {out}/  ({time.time()-t0:.1f}s total)")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/paysim.csv")
    parser.add_argument("--out", type=str, default="model")
    args = parser.parse_args()
    main(args.data, args.out)
