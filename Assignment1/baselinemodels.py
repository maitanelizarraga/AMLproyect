import numpy as np
import pandas as pd

def numerical_features(X_train, X_test):
    """
    Selects and scales numerical features using Min-Max normalization.
    """
    cols_to_use = [
        'account_age_days', 'previous_failed_attempts', 'is_international', 
        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
        'transaction_hour', 'avg_transaction_amount'
    ]
    
    X_train_num = X_train[cols_to_use].copy()
    X_test_num = X_test[cols_to_use].copy()
    
    print("\nScaling numerical features...")
    # List of columns to scale (excluding binary/categorical like transaction_hour)
    cols_to_scale = [
        'account_age_days', 'previous_failed_attempts', 'ip_risk_score', 
        'login_attempts_last_24h', 'transaction_amount', 'avg_transaction_amount'
    ]
    
    for col in cols_to_scale:
        min_val = X_train_num[col].min()
        max_val = X_train_num[col].max()
        if max_val > min_val:
            X_train_num[col] = (X_train_num[col] - min_val) / (max_val - min_val)
            # Use training min/max to scale test set to avoid data leakage
            X_test_num[col] = (X_test_num[col] - min_val) / (max_val - min_val)
    
    return X_train_num, X_test_num

def logistic_regression(X_train_num, y_train, X_test_num, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_num, y_train)
    y_pred = model.predict(X_test_num)
    
    metrics = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for k, v in metrics.items(): 
        if k != 'model': print(f"{k.capitalize()}: {v:.4f}")
    return metrics

def random_forest(X_train_num, y_train, X_test_num, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_num, y_train)
    y_pred = model.predict(X_test_num)
    
    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for k, v in metrics.items(): 
        if k != 'model': print(f"{k.capitalize()}: {v:.4f}")
    return metrics

def xgboost_model(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("XGBOOST MODEL")
    print("=" * 60)
    
    # Filter only numerical features to avoid string conversion errors
    X_train_proc = X_train.select_dtypes(include=['int64', 'float64']).copy()
    X_test_proc = X_test.select_dtypes(include=['int64', 'float64']).copy()
    
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    
    metrics = {
        'model': 'XGBoost',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for k, v in metrics.items(): 
        if k != 'model': print(f"{k.capitalize()}: {v:.4f}")
    return metrics

def lightgbm_model(X_train, y_train, X_test, y_test):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("LIGHTGBM MODEL")
    print("=" * 60)
    
    # Filter only numerical features to prevent pandas dtype issues
    X_train_proc = X_train.select_dtypes(include=['int64', 'float64']).copy()
    X_test_proc = X_test.select_dtypes(include=['int64', 'float64']).copy()
    
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    
    metrics = {
        'model': 'LightGBM',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for k, v in metrics.items(): 
        if k != 'model': print(f"{k.capitalize()}: {v:.4f}")
    return metrics

def compare_models(results):
    """
    Summarizes and sorts model performance by accuracy.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if r is not None]
    # Sort by accuracy in descending order
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for r in valid_results:
        print(f"{r['model']:<20} | Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
    
    return valid_results[0] if valid_results else None

def main(X_train, y_train, X_test, y_test):
    X_train_num, X_test_num = numerical_features(X_train, X_test)
    
    results = []
    results.append(logistic_regression(X_train_num, y_train, X_test_num, y_test))
    results.append(random_forest(X_train_num, y_train, X_test_num, y_test))
    results.append(xgboost_model(X_train_num, y_train, X_test_num, y_test))
    results.append(lightgbm_model(X_train_num, y_train, X_test_num, y_test))
    
    return compare_models(results)