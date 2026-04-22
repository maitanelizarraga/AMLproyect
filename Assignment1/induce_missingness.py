"""
induce_missingness.py
=====================
Utility for inducing controlled, reproducible missingness into the
Digital Payment Fraud Detection dataset before an Advanced ML pipeline.

Two missing-data mechanisms are supported:
  - MCAR  Missing Completely At Random  (uniform random removal)
  - MAR   Missing At Random             (removal conditioned on another column)

Usage
-----
    from induce_missingness import induce_missingness, run_diagnostics

    df_missing = induce_missingness(df, seed=42)
    run_diagnostics(df_missing, df)
"""

import pandas as pd
import numpy as np


PROTECTED_COLS = {"transaction_id", "user_id", "fraud_label"}

DEFAULT_MCAR = [
    {"column": "ip_risk_score",    "rate": 0.10},
    {"column": "account_age_days", "rate": 0.05},
]

DEFAULT_MAR = [
    {
        "column":      "previous_failed_attempts",
        "cond_column": "device_type",
        "cond_value":  "Android",
        "rate":        0.15,
    },
    {
        "column":      "payment_mode",
        "cond_column": "device_location",
        "cond_value":  "Hyderabad",
        "rate":        0.20,
    },
]


def induce_missingness(df, seed=42, mcar_configs=None, mar_configs=None, verbose=True):
    # Introduce missing values into a DataFrame using MCAR and MAR mechanisms.
    # Returns a copy of the input — the original is never mutated.


    if mcar_configs is None:
        mcar_configs = DEFAULT_MCAR
    if mar_configs is None:
        mar_configs = DEFAULT_MAR

    # Check that no protected column is being corrupted
    all_cols = [c["column"] for c in mcar_configs] + [c["column"] for c in mar_configs]
    blocked  = [c for c in all_cols if c in PROTECTED_COLS]
    if blocked:
        raise ValueError(f"These columns are protected and cannot be corrupted: {blocked}")

    df_out = df.copy()
    rng    = np.random.default_rng(seed)

    # MCAR — purely random removal
    for cfg in mcar_configs:
        col, rate = cfg["column"], cfg["rate"]
        n         = int(rate * len(df_out))
        idx       = rng.choice(df_out.index, size=n, replace=False)
        df_out.loc[idx, col] = np.nan
        if verbose:
            print(f"[MCAR]  '{col}': {n} rows -> NaN  ({rate*100:.1f}% of total)")

    # MAR — removal conditioned on another observed column
    for cfg in mar_configs:
        col, cond_col, cond_value, rate = cfg["column"], cfg["cond_column"], cfg["cond_value"], cfg["rate"]
        cond_idx = df_out[df_out[cond_col] == cond_value].index
        n        = int(rate * len(cond_idx))
        idx      = rng.choice(cond_idx, size=n, replace=False)
        df_out.loc[idx, col] = np.nan
        if verbose:
            print(f"[MAR ]  '{col}': {n} rows -> NaN  "
                  f"(where {cond_col}=={cond_value!r}, rate={rate*100:.0f}%) "
                  f"-> {n/len(df_out)*100:.1f}% of total")

    return df_out


def run_diagnostics(df_missing, df_original):

    # Print and return a table of missingness rates broken down by fraud_label.
    # Returns pd.DataFrame  Summary table.
    cols = [c for c in df_missing.columns if df_missing[c].isnull().any()]

    rows = []
    for col in cols:
        overall  = df_missing[col].isnull().mean() * 100
        by_class = df_missing.groupby("fraud_label")[col].apply(lambda s: s.isnull().mean() * 100)
        fraud_0  = by_class.get(0, 0.0)
        fraud_1  = by_class.get(1, 0.0)
        delta    = abs(fraud_1 - fraud_0)
        rows.append({
            "column":            col,
            "overall_missing%":  round(overall, 2),
            "missing%_fraud=0":  round(fraud_0, 2),
            "missing%_fraud=1":  round(fraud_1, 2),
            "class_delta":       round(delta, 2),
            "warning":           "high class skew" if delta > 5 else "",
        })

    summary = pd.DataFrame(rows).set_index("column")

    print("\n" + "=" * 65)
    print("MISSINGNESS DIAGNOSTICS")
    print("=" * 65)
    print(summary.to_string())
    print("=" * 65)
    print("Note: class_delta > 5pp may bias imputation toward the majority class.\n")

    return summary

def main():
    df = pd.read_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset.csv")
    print(f"Original shape:  {df.shape}")
    print(f"Missing values:  {df.isnull().sum().sum()}\n")

    df_missing = induce_missingness(df, seed=42)
    run_diagnostics(df_missing, df)

    df_missing.to_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv", index=False)
    print("Saved to: datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv")


if __name__ == "__main__": 
    main()


