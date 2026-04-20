import numpy as np
import pandas as pd

def get_numerical_features(X_train, X_test):  
    cols_to_use = ['account_age_days', 'previous_failed_attempts', 'is_international', 
                   'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
                   'transaction_hour', 'avg_transaction_amount']
    
    X_train_num = X_train[cols_to_use].copy()
    X_test_num = X_test[cols_to_use].copy()
    
    print("\nScaling numerical features...")
    for col in ['account_age_days', 'previous_failed_attempts', 'ip_risk_score', 
                'login_attempts_last_24h', 'transaction_amount', 'avg_transaction_amount']:
        min_val = X_train_num[col].min()
        max_val = X_train_num[col].max()
        if max_val > min_val:
            X_train_num[col] = (X_train_num[col] - min_val) / (max_val - min_val)
            X_test_num[col] = (X_test_num[col] - min_val) / (max_val - min_val)
    
    return X_train_num, X_test_num

def run_logistic_regression(X_train_num, y_train, X_test_num, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
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

def run_random_forest(X_train_num, y_train, X_test_num, y_test):
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

def run_xgboost(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("XGBOOST MODEL")
    print("=" * 60)
    
    # Filtrar solo numéricas para evitar error de strings
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

def run_lightgbm(X_train, y_train, X_test, y_test):
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("=" * 60)
        print("LIGHTGBM MODEL")
        print("=" * 60)
        
        # Filtrar solo numéricas para evitar el error de pandas dtypes
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
    except ImportError:
        print("LightGBM no está instalado.")
        return None

def compare_models(results):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for r in valid_results:
        print(f"{r['model']:<20} | Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
    return valid_results[0] if valid_results else None

def main(X_train, y_train, X_test, y_test):
    X_train_num, X_test_num = get_numerical_features(X_train, X_test)
    results = []
    results.append(run_logistic_regression(X_train_num, y_train, X_test_num, y_test))
    results.append(run_random_forest(X_train_num, y_train, X_test_num, y_test))
    results.append(run_xgboost(X_train, y_train, X_test, y_test))
    results.append(run_lightgbm(X_train, y_train, X_test, y_test))
    return compare_models(results)