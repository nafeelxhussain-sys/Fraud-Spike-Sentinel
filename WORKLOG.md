1. while making the func "get_recent_stats()" in 'store.py' and testing it, it crashed because I accidently used fetchall() insted of fetchone().

2. Adding timestamp for velocity features was hectic and we needed to sort that as well to prevent future data leakage.