"""
Feature engineering, shared by training (batch, over a whole CSV) and
serving (online, one transaction at a time via the SQLite store).
"""

import pandas as pd

TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def add_balance_error_features(df):
    """
    A real transaction clearly has
    newbalanceOrig == oldbalanceOrg - amount,
    newbalanceDest == oldbalanceDest + amount 
    """

    df = df.copy()
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def add_time_features(df):
    df = df.copy()
    df["hour_of_day"] = df["step"]%24
    df["day"] = df["step"] // 24
    return df


def add_type_dummies(df):
    df = df.copy()
    for t in TXN_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)
        # one hot encoding
    return df


def add_velocity_features_batch(df, window_hours = 24):
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["step"], unit="h", origin="2026-01-01")
    
    df = df.sort_values(["nameOrig", "_ts"]).reset_index(drop=True)

    grouped = df.set_index("_ts").groupby("nameOrig")["amount"]

    count_incl = grouped.rolling(f"{window_hours}h").count().reset_index(drop=True)
    sum_incl = grouped.rolling(f"{window_hours}h").sum().reset_index(drop=True)

    df["txn_count_24h"] = (count_incl - 1).clip(lower=0).values
    df["txn_amount_sum_24h"] = (sum_incl - df["amount"]).clip(lower=0).values

    df = df.drop(columns=["_ts"])
    return df


FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "hour_of_day",
    "day",
    "txn_count_24h",
    "txn_amount_sum_24h",
    # "newbalanceOrig",
    # "newbalanceDest",
    # "errorBalanceOrig",
    # "errorBalanceDest",
] + [f"type_{t}" for t in TXN_TYPES]


def build_training_features(df):
    # df = add_balance_error_features(df)
    df = add_time_features(df)
    df = add_type_dummies(df)
    df = add_velocity_features_batch(df)
    return df