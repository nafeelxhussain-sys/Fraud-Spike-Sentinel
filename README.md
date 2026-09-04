# Fraud Spike Sentinel

**Track: AI Risk Manager**

A real-time transaction fraud scorer paired with an account-level monitor that catches abnormal *spikes* in fraud rate over time — not just one bad transaction, but an account suddenly getting hit by a wave of them. Every decision is threshold-gated by real business cost, routed through a four-state policy, explained in plain English by an LLM, and fully audit-logged.

## Results

Held out on a **time-based split** (train → validation → test, chronologically ordered by transaction step); the threshold was chosen on the validation slice and metrics are reported on the test slice.

| Metric | Value |
|---|---|
| Precision | **0.609** |
| Recall | **0.896** |
| F1 | **0.725** |
| PR-AUC | **0.903** |
| Decision threshold | **0.87** (cost-optimal, not F1-optimal) |
| Dataset | [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6,362,620 real transactions, 8,213 fraud (0.129%) |
| Model | XGBoost, `scale_pos_weight` = neg/pos ratio, 100 trees, depth 5, lr 0.1 |
| Scoring throughput | 92 transactions/sec |

**In plain terms:** at this threshold, roughly 1 in 3 blocked transactions is genuinely fraud, and the system catches ~90% of all real fraud in the held-out data. That's a deliberately conservative operating point for a hard BLOCK decision — the MONITOR/REVIEW states below exist specifically to catch the medium-confidence cases before they'd ever need to clear that bar alone.

## Architecture

```
raw transaction
      |
      v
 feature engineering  --  balance/time/type features + 24h account velocity
      |                          ^
      v                          |
 SQLite store (indexed on account+step) -- real transaction history, not a batch file
      |
      v
 XGBoost model  -->  risk probability
      |
      v
 cost-calibrated threshold  -->  ALLOW / MONITOR / REVIEW / BLOCK
      |                                          |
      v                                          v
 Groq LLM explanation                    audit log (every decision, permanent)
 ("why was this flagged")
      |
      v
 FastAPI (POST /score, GET /account/{id}/spike-status, GET /account/{id}/history)
      |
      v
 Streamlit dashboard -- live feed, model performance, account spike timeline
```

Separately, `spike_detector.py` watches each account's score history over time (control-chart style: current window vs. that account's own baseline mean/std) and flags a burst of medium-risk activity that no single transaction would trigger alone.

## How to run

```bash
pip install -r requirements.txt
./run_dev.sh          
```

Retrain explicitly on the real dataset:
```bash
python -m src.train --data data/paysim.csv --out model
```

## Why these design choices

- **Time-based 3-way split (train / validation / test), not a random split.** Random shuffling would leak future transactions into training on time-ordered data. Splitting by `step` is both more rigorous and more representative of how this would
actually be evaluated in production.
- **Cost-calibrated threshold, not F1-maximizing.** The threshold is chosen by minimizing `(false positives × cost_per_fp) + (false negatives × fraud amount missed)`, where `cost_per_fp` is scaled to the actual transaction amounts in this dataset rather than an arbitrary flat number; cost_per_fp (marking a legit transaction fraud) is set to 5% of median transaction amount.
- **Four-state policy (ALLOW / MONITOR / REVIEW / BLOCK)** A transaction that's mildly suspicious shouldn't cost a legitimate customer the same friction as one that's almost certainly fraud. BLOCK uses the cost-optimal threshold from training; MONITOR/REVIEW are softer bands below it.
- **SQLite, indexed on `(account, step)`** It stores the account history. It's what makes live per-account velocity lookups an indexed range scan instead of a full table scan.
- **LLM explanation** It uses LLM to explain the reasoning behind the decision provided by model. It uses groq API calls

## Data & limitations

Trained on PaySim, a synthetic mobile-money simulator, not real merchant payment-gateway data; accounts are treated as a stand-in for a merchant/account, and the spike-detection logic is a generalizable pattern rather than something tuned to actual transaction shape.


## Benchmarking

| Name | Throughput | 
| -------- | -------- | 
| API's /score | 92.32 | 

Two concrete things were done to make this fast, not just fast enough to demo:
- Live scoring computes features with direct arithmetic instead of routing through the batch pandas pipeline used for training — the batch pipeline's repeated `DataFrame.copy()` calls are correct for transforming millions of training rows at once, and pure overhead for one live transaction.
- SQLite runs in WAL mode with `synchronous=NORMAL`, and both writes per scored transaction (the audit log entry and the account-history record) are batched under a single commit instead of two.


## What I'd do with more time

- Destination-account velocity as a feature — money flowing *into* an account is a classic mule-account signal that sender-side features alone can't see
- SHAP values for per-transaction explanation instead of the current permutation-importance-based local approximation
- Docker for one-command deployment anywhere, not just this machine
