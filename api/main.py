"""
FastAPI service for the Fraud-Spike Sentinel.

Run from the project root:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /score                          
    GET  /account/{name_orig}/spike-status
    GET  /account/{name_orig}/history
"""

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.score import FraudScorer
from src.store import init_db, get_account_score_history
from src.spike_detector import detect_spike
from src.explainer import get_fraud_explanation
from dotenv import load_dotenv; 
load_dotenv()

MODEL_DIR = "model"
DB_PATH = "history.db"

app = FastAPI(title="Fraud-spike-sentinel")
scorer = FraudScorer(model_dir=MODEL_DIR)
conn = init_db(DB_PATH)

class TransactionIn(BaseModel):
    nameOrig: str
    step: int
    amount: float
    type: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class ScoreOut(BaseModel):
    score: float
    threshold: float
    decision: str
    contribution_features : List[dict]
    explanation : str = None


# Route 1: GET /health
@app.get("/health")
def health():
    return {"status": "ok", "threshold": scorer.threshold}


# Route 2: POST /score
@app.post("/score", response_model=ScoreOut)
def score_transaction(txn: TransactionIn):
    try:
        transc = txn.model_dump()
        result = scorer.score(transc,conn)
        result["explanation"] = get_fraud_explanation(
            transc,
            result["decision"],
            result.get("contribution_features", [])
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400,detail=f"could not score: {exc}")



# Route 3: GET /account/{name_orig}/spike-status
@app.get("/account/{name_orig}/spike-status")
def spike_status(name_orig: str):
    history = get_account_score_history(conn, name_orig)

    if len(history) == 0:
        raise HTTPException(status_code=404, detail=f"No history")

    proba_scores = [row[1] for row in history]
    result = detect_spike(proba_scores)

    return {
        "name_orig":name_orig,
        "n_scored_transactions": len(history),
        "is_spike":result.is_spike,
        "current_rate":round(result.current_rate,2),
        "baseline_mean":round(result.baseline_mean,2),
        "z_score":round(result.z_score,2),
    }

# ROUTE 4: GET /account/{name_orig}/history
@app.get("/account/{name_orig}/history")
def account_history(name_orig: str):
    """
    Raw score-by-score history for one account -- what the dashboard's
    spike-timeline chart actually plots. spike-status above returns the
    verdict; this returns the data behind it.
    """
    history = get_account_score_history(conn, name_orig)
    if not history:
        raise HTTPException(status_code=404, detail="No score history for this account yet")
    return [
        {"step": row[0], "score": row[1], "decision": row[2], "created_at": row[3]}
        for row in history
    ]
    