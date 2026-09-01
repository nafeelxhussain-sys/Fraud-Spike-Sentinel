"""
A small SQLite-backed store for per account transaction history and audit
logging. This exists so the live API can know how much has this
account moved in the last 24h, with an indexed lookup instead of scanning
a growing dataframe on every request.

Two tables:
  transactions(name_orig, step, amount, type)  -- indexed on (name_orig, step)
  score_log(...)                                -- one row per scoring
                                                     decision, the audit trail
"""


import sqlite3 as sql
from contextlib import contextmanager
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name_orig TEXT NOT NULL,
    step INTEGER NOT NULL,
    amount REAL NOT NULL,
    type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS score_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name_orig TEXT NOT NULL,
    step INTEGER NOT NULL,
    amount REAL NOT NULL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    decision TEXT NOT NULL,
    top_features TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_txn_account_step
    ON transactions (name_orig, step);

CREATE INDEX IF NOT EXISTS idx_log_account
    ON score_log (name_orig);
"""


def init_db(path):
    conn = sql.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def get_conn(path):
    conn = init_db(path)
    try:
        yield conn
    finally:
        conn.close()


@dataclass
class AccountStats:
    txn_count_24h: float
    txn_amount_sum_24h: float

def record_transaction(conn, name_orig, step, amount, txn_type):
    conn.execute(
        "INSERT INTO transactions (name_orig, step, amount, type) VALUES (?,?,?,?)",
        (name_orig, step, amount, txn_type),
    )
    conn.commit()

def get_recent_stats(conn, name_orig, current_step, window_hours):
    cur = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(amount),0) 
        FROM transactions 
        WHERE name_orig = ? AND STEP >= ? AND STEP < ?
        """,
        (name_orig, current_step - window_hours, current_step),
    )

    count, amount_sum = cur.fetchone()
    return AccountStats(float(count), float(amount_sum))

def log_score(conn, name_orig, step, amount,score, threshold, decision, top_features):
    conn.execute(
        """
        INSERT INTO score_log (name_orig, step, amount, score, threshold, decision, top_features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (name_orig, step, amount, score, threshold, decision, top_features),
    )

    conn.commit()

def get_account_score_history(conn, name_orig, limit=200):
    cur = conn.execute(
        """
        SELECT step, score, decision, created_at FROM score_log
        WHERE name_orig = ? ORDER BY step ASC LIMIT ?
        """,
        (name_orig, limit),
    )
    return cur.fetchall()
