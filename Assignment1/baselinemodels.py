import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def numerical_features(X_train, X_val, X_test):
    #Selects and scales numerical variables using Min-Max based on X_train.
    cols_to_use = [
        'account_age_days', 'previous_failed_attempts', 'is_international', 
        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
        'transaction_hour', 'avg_transaction_amount'
    ]
    
    # Filter columns
    X_train_num = X_train[cols_to_use].copy()
    X_val_num = X_val[cols_to_use].copy()
    X_test_num = X_test[cols_to_use].copy()
    
    print("\nScaling numerical variables...")
    cols_to_scale = [
        'account_age_days', 'previous_failed_attempts', 'ip_risk_score', 
        'login_attempts_last_24h', 'transaction_amount', 'avg_transaction_amount'
    ]
    
    for col in cols_to_scale:
        min_val = X_train_num[col].min()
        max_val = X_train_num[col].max()
        
        if max_val > min_val:
            # Apply transformation from Train set to all three sets
            X_train_num[col] = (X_train_num[col] - min_val) / (max_val - min_val)
            X_val_num[col] = (X_val_num[col] - min_val) / (max_val - min_val)
            X_test_num[col] = (X_test_num[col] - min_val) / (max_val - min_val)
    
    return X_train_num, X_val_num, X_test_num

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
    train_df = pd.read_csv("./Assignment1/datasets/train.csv")
    val_df = pd.read_csv("./Assignment1/datasets/val.csv")
    test_df = pd.read_csv("./Assignment1/datasets/test.csv")
    
    X_train, y_train = train_df.drop("fraud_label", axis=1), train_df["fraud_label"]
    X_val, y_val = val_df.drop("fraud_label", axis=1), val_df["fraud_label"]
    X_test, y_test = test_df.drop("fraud_label", axis=1), test_df["fraud_label"]
    
    # 2. Process variables
    X_train_n, X_val_n, X_test_n = numerical_features(X_train, X_val, X_test)
    
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
    best_model_meta = compare_models(results)
    print(f"\nThe best model is: {best_model_meta['model']}")

if __name__ == "__main__":
    main()