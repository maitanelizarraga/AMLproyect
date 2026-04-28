import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def process_all_features(X_train, X_val, X_test):
    # 1. Identify columns
    cat_cols = ['transaction_type', 'payment_mode', 'device_type']
    num_cols = [
        'account_age_days', 'previous_failed_attempts', 'is_international', 
        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
        'transaction_hour', 'avg_transaction_amount'
    ]
 
    # 2. Aply One-Hot Encoding
    X_train_final = pd.get_dummies(X_train[cat_cols + num_cols], columns=cat_cols, dtype=int)
    X_val_final = pd.get_dummies(X_val[cat_cols + num_cols], columns=cat_cols, dtype=int)
    X_test_final = pd.get_dummies(X_test[cat_cols + num_cols], columns=cat_cols, dtype=int)
    
    # Ensure that everyone has the same columns (in case any category does not appear in val/test)
    X_train_final, X_val_final = X_train_final.align(X_val_final, join='left', axis=1, fill_value=0)
    X_train_final, X_test_final = X_train_final.align(X_test_final, join='left', axis=1, fill_value=0)

    # We converted everything to float to avoid any further type problems
    X_train_final = X_train_final.astype(float)
    X_val_final = X_val_final.astype(float)
    X_test_final = X_test_final.astype(float)

    # 3. Scale all variables
    for col in X_train_final.columns:
        min_val = X_train_final[col].min()
        max_val = X_train_final[col].max()
        if max_val > min_val:
            X_train_final[col] = (X_train_final[col] - min_val) / (max_val - min_val)
            X_val_final[col] = (X_val_final[col] - min_val) / (max_val - min_val)
            X_test_final[col] = (X_test_final[col] - min_val) / (max_val - min_val)
            
    return X_train_final, X_val_final, X_test_final

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    #Trains on the Train set and evaluates on Validation to compare models.

    print("=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val) # Evaluate on VALIDATION
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='weighted'),
        'recall': recall_score(y_val, y_pred, average='weighted'),
        'f1': f1_score(y_val, y_pred, average='weighted')
    }
    
    for k, v in metrics.items(): 
        if k != 'model': print(f"{k.capitalize()}: {v:.4f}")
    return metrics

def compare_models(results):
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (ON VALIDATION)")
    print("=" * 60)
    
    # Sort by F1 Score (better metric for fraud detection)
    results.sort(key=lambda x: x['f1'], reverse=True) 
    
    for r in results:
        print(f"{r['model']:<20} | Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
    
    return results[0]

def main():
    train_df = pd.read_csv("./datasets/train.csv")
    val_df = pd.read_csv("./datasets/val.csv")
    test_df = pd.read_csv("./datasets/test.csv")
    
    X_train, y_train = train_df.drop("fraud_label", axis=1), train_df["fraud_label"]
    X_val, y_val = val_df.drop("fraud_label", axis=1), val_df["fraud_label"]
    X_test, y_test = test_df.drop("fraud_label", axis=1), test_df["fraud_label"]
    
    # 2. Process variables
    X_train_n, X_val_n, X_test_n = process_all_features(X_train, X_val, X_test)
    
    results = []
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    results.append(evaluate_model(LogisticRegression(max_iter=1000), X_train_n, y_train, X_val_n, y_val, "Logistic Regression"))
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    results.append(evaluate_model(RandomForestClassifier(n_estimators=1000, n_jobs=-1), X_train_n, y_train, X_val_n, y_val, "Random Forest"))
    
    # XGBoost
    from xgboost import XGBClassifier
    results.append(evaluate_model(XGBClassifier(n_estimators=1000, eval_metric='logloss'), X_train_n, y_train, X_val_n, y_val, "XGBoost"))
    
    # LightGBM
    from lightgbm import LGBMClassifier
    results.append(evaluate_model(LGBMClassifier(n_estimators=1000, verbose=-1), X_train_n, y_train, X_val_n, y_val, "LightGBM"))
    
    # 3. Compare and select the best model
    best_model = compare_models(results)
    print(f"\nThe best model is: {best_model['model']}")
    
    # 4. FINAL evaluation with the TEST set
    print("\n" + "!" * 60)
    print("FINAL EVALUATION ON TEST SET (UNSEEN DATA)")
    print("!" * 60)

    if best_model == "Logistic Regression":
        final_model = LogisticRegression(max_iter=1000)
    elif best_model == "Random Forest":
        final_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    elif best_model == "XGBoost":
        final_model = XGBClassifier(n_estimators=1000,eval_metric='logloss')
    else:
        final_model = LGBMClassifier(n_estimators=1000, verbose=-1)

    evaluate_model(final_model, X_train_n, y_train, X_test_n, y_test, f"Final test:{best_model['model']}")

if __name__ == "__main__":
    main()