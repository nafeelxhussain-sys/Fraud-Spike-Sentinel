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
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from src.features import build_training_features, FEATURE_COLUMNS

#  BUSINESS CONSTANTS
MEDIAN_TRANSACTION = 74871.94                             #calculated on dataset
COST_PER_FALSE_POSITIVE = MEDIAN_TRANSACTION * 0.05       #marking a legit txn as fraud 
COST_PER_FALSE_NEGATIVE = 1.0                             #marking a fraud txn as legit

def time_based_split(df: pd.DataFrame,val_fraction:float=0.15, test_fraction: float = 0.15):
    val_cutoff = df["step"].quantile(1 - test_fraction-val_fraction)
    test_cutoff = df["step"].quantile(1 - test_fraction)

    train = df[df["step"] < val_cutoff].copy()
    val = df[(df["step"] >= val_cutoff) & (df["step"] < test_cutoff)].copy()
    test = df[df["step"] >= test_cutoff].copy()
    return train, val, test

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


    return best["threshold"], sweep

def cost_calculate(y_test, y_proba, amounts_test, threshold) -> float:
    y_pred = (y_proba >= threshold).astype(int)
    cost_fp = ((y_pred == 1) & (y_test == 0)).sum() * COST_PER_FALSE_POSITIVE
    cost_fn_mask = ((y_pred == 0) & (y_test == 1))
    cost_fn = amounts_test[cost_fn_mask].sum() * COST_PER_FALSE_NEGATIVE

    cost = cost_fp + cost_fn
    return cost



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
    
    train, val, test = time_based_split(feat)
    print(f"train={len(train):,} rows, validation={len(val):,} rows, test={len(test):,} rows")

    X_train, y_train = train[FEATURE_COLUMNS], train["isFraud"].values
    X_val, y_val = val[FEATURE_COLUMNS], val["isFraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["isFraud"].values
    amounts_val = val["amount"].values
    amounts_test = test["amount"].values

    print(X_train.info())

    fraud_count = sum(y_train == 1)
    normal_count = sum(y_train == 0)
    imbalance_ratio = normal_count / fraud_count
    
    print("Training XGboost  ...")
    model = xgb.XGBClassifier( 
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=imbalance_ratio, 
        random_state=67,
        use_label_encoder=False,
        eval_metric='aucpr' 
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_test, proba_test)

    threshold, sweep = choose_threshold(y_val, proba_val, amounts_val)
    pred_test = (proba_test >= threshold).astype(int)

    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)

    #cost at threshold
    threshold_cost = cost_calculate(y_test, proba_test, amounts_test, threshold)

    # cost of two naive baselines so the threshold choice can be justified
    always_zero_cost = cost_calculate(y_test, proba_test, amounts_test, threshold = 2)
    default_half_cost = cost_calculate(y_test, proba_test, amounts_test, threshold = 0.5)

    print("\n--- Held-out test metrics (time-split, model has never seen these rows) ---")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1:             {f1:.3f}")
    print(f"  PR-AUC:         {pr_auc:.3f}")
    print(f"  Chosen threshold: {threshold:.2f}")
    print(f"  Cost at chosen threshold:     {threshold_cost:,.2f}")
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "cost_at_threshold": threshold_cost,
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
