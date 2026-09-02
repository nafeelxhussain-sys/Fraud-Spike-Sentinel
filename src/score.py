"""
scores a single incoming transaction
"""

import json
import sqlite3
from pathlib import Path
import joblib
import pandas as pd

from src.features import (
    add_balance_error_features,
    add_time_features,
    add_type_dummies,
    FEATURE_COLUMNS,
    TXN_TYPES,
)
from src.store import get_recent_stats, record_transaction, log_score

# Bussiness Parameters
MONITOR_MULT = 0.4
REVIEW_MULT = 0.75

class FraudScorer:
    def __init__(self, model_dir: str = "model"):
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.model = joblib.load(out / "model.joblib")

        with open (out / "metrics.json") as f:
            self.metrics = json.load(f)

        self.threshold = self.metrics["threshold"]
        self.feature_reference = self.metrics["feature_reference"]
        self.importance_by_features =  {
            row["feature"]: row["importance"] for row in  self.metrics["top_features"]
        }

        
    def _build_feature_row(self, txn: dict, conn: sqlite3.Connection) -> pd.DataFrame:

        row = pd.DataFrame([txn])
        row = add_balance_error_features(row)
        row = add_time_features(row)
        row = add_type_dummies(row)

        acc_stats = get_recent_stats(conn, txn["nameOrig"], txn["step"], 24)

        row["txn_count_24h"] = acc_stats.txn_count_24h
        row["txn_amount_sum_24h"] = acc_stats.txn_amount_sum_24h
        
        
        for feat in FEATURE_COLUMNS:
            if feat not in row.columns:
                row[feat]=0.

        return row[FEATURE_COLUMNS]



    def _explain(self, feature_row: pd.DataFrame, top_n: int = 3) -> list:
        """
        contributin is used by llms to explain why a txn was flagged.
        """

        contribution = [] 

        for feat in FEATURE_COLUMNS:
            ref = self.feature_reference.get(feat, {"median": 0.0, "std": 1.0})
            median = ref["median"]
            std = ref["std"] or 1.0
            
            value = float(feature_row[feat].iloc[0])

            z_score = abs((value - median) / std)
            importance = self.importance_by_features.get(feat, 0.0)

            contribution.append({
                "feature": feat,
                "value": round(value, 2),
                "typical_value": round(median, 2),
                "deviation_z": round(z_score, 2),
                "importance": round(importance, 4),
                "contribution": round(z_score * importance, 4),
            })

        contribution.sort(key=lambda x : -x["contribution"])
        return [c for c in contribution[:top_n] if c["contribution"] > 0]
        
        
        

    def score(self, txn: dict, conn: sqlite3.Connection, record: bool = True) -> dict:

        row = self._build_feature_row(txn, conn)
        proba = float(self.model.predict_proba(row)[:, 1][0])

        if proba >= self.threshold:
            decision = "BLOCK"
        elif proba >= REVIEW_MULT * self.threshold:
            decision = "REVIEW"
        elif proba >= MONITOR_MULT * self.threshold:
            decision = "MONITOR"
        else:
            decision = "ALLOW"


        contri_feat = self._explain(row)
        contri_feat_json = json.dumps(contri_feat)

        log_score(
            conn, 
            txn["nameOrig"], 
            txn["step"], 
            txn["amount"],
            proba,
            self.threshold,
            decision,
            contri_feat_json
        )
        
        if record:
            record_transaction(conn,txn["nameOrig"], txn["step"], txn["amount"],txn["type"])
        
        return {
            "score":proba,
            "threshold": self.threshold,
            "decision" : decision,
            "contribution_features" : contri_feat,
        }