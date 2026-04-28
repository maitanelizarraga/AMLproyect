import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# 1. CONFIGURATION

TRAIN_PATH = "./datasets/train_balanced_adasyn.csv"   # ADASYN-balanced training set
VAL_PATH   = "./datasets/val.csv"                     # Validation set (threshold tuning)
TEST_PATH  = "./datasets/test.csv"                    # Final held-out test set

TARGET = "fraud_label"

# Numerical columns to scale (same subset used in baselinemodels.py)
COLS_TO_SCALE = [
    "account_age_days", "previous_failed_attempts", "ip_risk_score",
    "login_attempts_last_24h", "transaction_amount", "avg_transaction_amount",
]

# LightGBM hyper-parameters (same as baseline for a fair comparison)
LGBM_PARAMS = dict(n_estimators=1000, random_state=42, verbose=-1, n_jobs=-1)


# 2. HELPERS 
def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    #Keep only numeric columns (mirrors the filter used in imbalance.py).
    return df.select_dtypes(include=[np.number])


def minmax_scale(train: pd.DataFrame, *others: pd.DataFrame):
    """
    Fits Min-Max scaling on `train` and applies it to every DataFrame in
    `others`.  Returns scaled copies; original DataFrames are not modified.
    """
    train_scaled = train.copy()
    scaled_others = [df.copy() for df in others]

    for col in COLS_TO_SCALE:
        if col not in train.columns:
            continue
        lo, hi = train[col].min(), train[col].max()
        if hi == lo:                          # constant column – skip
            continue
        train_scaled[col] = (train[col] - lo) / (hi - lo)
        for df in scaled_others:
            if col in df.columns:
                df[col] = (df[col] - lo) / (hi - lo)

    return (train_scaled, *scaled_others)


def find_best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Sweeps probability thresholds on the VALIDATION set and returns the one
    that maximises the weighted F1-score.  This keeps the test set clean.
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(y_val, preds, average="weighted")
        if score > best_f1:
            best_f1, best_t = score, t
    print(f"  → Best threshold (val F1={best_f1:.4f}): {best_t:.2f}")
    return best_t


def plot_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# 3. MAIN 
def main():
    print("=" * 65)
    print("  FINAL RE-EVALUATION: ADASYN + LightGBM on TEST SET")
    print("=" * 65)

    # 3.1  Load datasets 
    print("\n[1/5] Loading datasets...")
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    print(f"  Train (ADASYN) : {train.shape}  |  Fraud ratio: {train[TARGET].mean():.2%}")
    print(f"  Validation     : {val.shape}   |  Fraud ratio: {val[TARGET].mean():.2%}")
    print(f"  Test           : {test.shape}   |  Fraud ratio: {test[TARGET].mean():.2%}")

    #  3.2  Separate features / target 
    X_train_raw = select_numeric_features(train.drop(TARGET, axis=1))
    y_train     = train[TARGET]

    X_val_raw   = select_numeric_features(val.drop(TARGET, axis=1))
    y_val       = val[TARGET]

    X_test_raw  = select_numeric_features(test.drop(TARGET, axis=1))
    y_test      = test[TARGET]

    #  3.3  Scale features (no data leakage) 
    print("\n[2/5] Scaling features (Min-Max fitted on ADASYN train)...")
    X_train, X_val, X_test = minmax_scale(X_train_raw, X_val_raw, X_test_raw)

    #  3.4  Train LightGBM 
    print("\n[3/5] Training LightGBM...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    print("  Training complete.")

    #  3.5  Threshold tuning on VALIDATION set 
    print("\n[4/5] Tuning decision threshold on VALIDATION set...")
    best_threshold = find_best_threshold(model, X_val, y_val)

    #  3.6  Final evaluation on TEST set 
    print("\n[5/5] Evaluating on TEST set (unseen data)...")
    probs_test  = model.predict_proba(X_test)[:, 1]
    y_pred_def  = model.predict(X_test)                            # default 0.5
    y_pred_tuned = (probs_test >= best_threshold).astype(int)      # tuned threshold

    roc_auc = roc_auc_score(y_test, probs_test)

    print("\n" + "─" * 65)
    print(f"  ROC-AUC (test): {roc_auc:.4f}")
    print("─" * 65)

    print("\n  ── Default threshold (0.50) ──")
    print(classification_report(y_test, y_pred_def,
                                target_names=["Non-Fraud", "Fraud"], zero_division=0))

    print(f"\n  ── Tuned threshold  ({best_threshold:.2f}) ──")
    print(classification_report(y_test, y_pred_tuned,
                                target_names=["Non-Fraud", "Fraud"], zero_division=0))

    #  3.7  Metrics comparison table 
    metrics_summary = pd.DataFrame([
        {
            "Threshold": "0.50 (default)",
            "Accuracy" : round((y_test == y_pred_def).mean(), 4),
            "Precision": round(f1_score(y_test, y_pred_def, average="weighted", zero_division=0), 4),
            "Recall"   : round(f1_score(y_test, y_pred_def, average="macro",    zero_division=0), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred_def, average="weighted", zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc, 4),
        },
        {
            "Threshold": f"{best_threshold:.2f} (tuned)",
            "Accuracy" : round((y_test == y_pred_tuned).mean(), 4),
            "Precision": round(f1_score(y_test, y_pred_tuned, average="weighted", zero_division=0), 4),
            "Recall"   : round(f1_score(y_test, y_pred_tuned, average="macro",    zero_division=0), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred_tuned, average="weighted", zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc, 4),
        },
    ])

    print("\n" + "=" * 65)
    print("  METRICS SUMMARY")
    print("=" * 65)
    print(metrics_summary.to_string(index=False))

    #  3.8  Confusion matrix (tuned threshold) 
    plot_confusion_matrix(y_test, y_pred_tuned,
                          f"LightGBM + ADASYN — Threshold {best_threshold:.2f}")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()