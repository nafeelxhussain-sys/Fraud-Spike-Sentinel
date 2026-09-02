1. while making the func "get_recent_stats()" in 'store.py' and testing it, it crashed because I accidently used fetchall() insted of fetchone().

2. Rolling velocity features ('txn_count_24h') were initially computed with
'shift(1)' on the previous rows rolling window value. That's wrong for
accounts with sparse activity; if an accounts last transaction was 10
days ago, shifting to that row gives a 10 days older number labeled as "last
24h activity". Fixed by computing the rolling window inclusive of the
current transaction, then subtracting the transactions own contribution.

3. Adding timestamp for velocity features was hectic and we needed to sort that as well to prevent future data leakage.

4. added check_same_thread=False in __init__ of database because....

5. The Missing Key Crash (KeyError): _explain tried to pull errorBalanceOrig directly from the reference dictionary, crashing when a feature had 0 importance during training. I replaced direct bracket notation with .get(feat, {"median": 0.0, "std": 1.0}) for safe fallbacks.

6. The Pandas Series Pani Attempted to run round() on feature_row[feat], which returns a Pandas Series, causing Streamlit/Pydantic to break fixed it by extracted the pure float scalar using float(feature_row[feat].iloc[0]).

7. Passed the raw txn dictionary into _explain() instead of the engineered row DataFrame, so it couldn't find the engineered features. Simple fix was making self._explain(txn) to self._explain(row).

8. Accidently passed contri_feat (a Python list) directly into the SQLite log_score function, which expects a TEXT string. Then used json.dumps()



