## What broke

1. while making the func "get_recent_stats()" in 'store.py' and testing it, it crashed because I accidently used fetchall() insted of fetchone().

2. Rolling velocity features ('txn_count_24h') were initially computed with
'shift(1)' on the previous rows rolling window value. That's wrong for
accounts with sparse activity; if an accounts last transaction was 10
days ago, shifting to that row gives a 10 days older number labeled as "last
24h activity". Fixed by computing the rolling window inclusive of the
current transaction, then subtracting the transactions own contribution.

3. Adding timestamp for velocity features was hectic and we needed to sort that as well to prevent future data leakage.

4. added check_same_thread=False in __init__ of database 

5. The Missing Key Crash (KeyError): _explain tried to pull errorBalanceOrig directly from the reference dictionary, crashing when a feature had 0 importance during training. I replaced direct bracket notation with .get(feat, {"median": 0.0, "std": 1.0}) for safe fallbacks.

6. The Pandas Series Pani Attempted to run round() on feature_row[feat], which returns a Pandas Series, causing Streamlit/Pydantic to break fixed it by extracted the pure float scalar using float(feature_row[feat].iloc[0]).

7. Passed the raw txn dictionary into _explain() instead of the engineered row DataFrame, so it couldn't find the engineered features. Simple fix was making self._explain(txn) to self._explain(row).

8. Accidently passed contri_feat (a Python list) directly into the SQLite log_score function, which expects a TEXT string. Then used json.dumps()

9. The first real-data training run hit 100% recall and 100% PR-AUC which was too good to be real. Permutation importance showed two balance-error features (errorbalanceOrig and errorbalanceDest) carrying almost the entire decision, traced back to a PaySim-specific simulator artifact rather than learned fraud behavior. Fixed by ablating those features and retraining the model. The model was trained on `XGboost` instead `histgradientBoosting`

10. The cost-sensitive threshold collapsed to blocking almost everyone (precision as low as 0.05 at one point) which was because of the false-positive cost constant was an arbitrary flat number completely disconnected from the scale of real fraud amounts in the data, so the optimizer treated blocking legitimate transactions as functionally free. Fixed by scaling the false-positive cost to the actual transaction-amount scale in the dataset.

11. The threshold was being chosen and evaluated on the same held-out test set which resulted in subtle form of leakage that flatters the reported numbers by however much the threshold search could exploit that specific slice. Fixed with a proper 3-way time split: threshold picked on validation, final metrics reported on test, which never influenced it, and the training was done on training set.

12. Adding catched data in stream made the scoring function work lightining fast.

13. I replaced the localhost URL with 127.0.0.1 in the API configurations to bypass a Windows IPv6 DNS resolution timeout that was delaying every request