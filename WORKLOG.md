1. while making the func "get_recent_stats()" in 'store.py' and testing it, it crashed because I accidently used fetchall() insted of fetchone().

2. Rolling velocity features ('txn_count_24h') were initially computed with
'shift(1)' on the previous rows rolling window value. That's wrong for
accounts with sparse activity; if an accounts last transaction was 10
days ago, shifting to that row gives a 10 days older number labeled as "last
24h activity". Fixed by computing the rolling window inclusive of the
current transaction, then subtracting the transactions own contribution.

3. Adding timestamp for velocity features was hectic and we needed to sort that as well to prevent future data leakage.

