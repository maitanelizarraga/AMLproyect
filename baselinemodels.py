def numericalfeatures(X_train, X_test):  # <-- Cambiar df por X_train, X_test
    # Seleccionamos solo valores numéricos
    cols_to_use = ['account_age_days', 'previous_failed_attempts', 'is_international', 
                   'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
                   'transaction_hour', 'avg_transaction_amount']
    
    X_train_num = X_train[cols_to_use].copy()
    X_test_num = X_test[cols_to_use].copy()
    
    # Escalado de variables
    print("\nEscalando variables numéricas...")
    for col in ['account_age_days', 'previous_failed_attempts', 'ip_risk_score', 
                'login_attempts_last_24h', 'transaction_amount', 'avg_transaction_amount']:
        min_val = X_train_num[col].min()
        max_val = X_train_num[col].max()
        
        if max_val > min_val:
            X_train_num[col] = (X_train_num[col] - min_val) / (max_val - min_val)
            X_test_num[col] = (X_test_num[col] - min_val) / (max_val - min_val)
    
    print(f"Dimensiones finales - X_train: {X_train_num.shape}, X_test: {X_test_num.shape}")
    print("-" * 60)
    return X_train_num, X_test_num


def logisticregression(X_train_num, y_train, X_test_num, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    import pandas as pd
    
    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    # train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_num, y_train)  # <-- Cambiar X_train_selected por X_train_num
    
    # make predictions
    y_pred = model.predict(X_test_num)  # <-- Cambiar X_test_selected por X_test_num
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Obtener nombres de columnas
    if hasattr(X_train_num, 'columns'):
        features_used = list(X_train_num.columns)
    else:
        features_used = ['account_age_days', 'previous_failed_attempts', 'is_international', 
                        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
                        'transaction_hour', 'avg_transaction_amount']
    
    return {
        'model': 'Logistic Regression',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'model_object': model,
        'features_used': features_used
    }


def randomforest(X_train_num, y_train, X_test_num, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    import pandas as pd
    
    print("=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)
    
    # train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_num, y_train)
    
    # make predictions
    y_pred = model.predict(X_test_num)
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Obtener nombres de columnas
    if hasattr(X_train_num, 'columns'):
        features_used = list(X_train_num.columns)
    else:
        features_used = ['account_age_days', 'previous_failed_attempts', 'is_international', 
                        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
                        'transaction_hour', 'avg_transaction_amount']
    
    # feature importance
    feature_importance = model.feature_importances_
    print(f"\nTop 5 Most Important Features:")
    indices = np.argsort(feature_importance)[-5:][::-1]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {features_used[idx]}: {feature_importance[idx]:.4f}")
    

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'model': 'Random Forest',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'feature_importance': feature_importance,
        'features_used': features_used,
        'model_object': model
    }


def XGBoost(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    import pandas as pd
    
    print("=" * 60)
    print("XGBOOST MODEL")
    print("=" * 60)
    
    # Para XGBoost, seleccionamos solo numéricas también (por simplicidad)
    # pero XGBoost podría usar categóricas nativamente
    
    # Identificar columnas numéricas
    if hasattr(X_train, 'select_dtypes'):
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_proc = X_train[numerical_cols].copy()
        X_test_proc = X_test[numerical_cols].copy()
    else:
        X_train_proc = X_train
        X_test_proc = X_test
    
    # Train model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_proc, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_proc)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        if hasattr(X_train_proc, 'columns'):
            print(f"\nTop 5 Most Important Features:")
            indices = np.argsort(feature_importance)[-5:][::-1]
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {X_train_proc.columns[idx]}: {feature_importance[idx]:.4f}")
    

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'model': 'XGBoost',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'model_object': model
    }


# def LightGBM(X_train, y_train, X_test, y_test):
#     """
#     LightGBM baseline model.
#     Gradient boosting algorithm optimized for speed and efficiency.
#     """
#     try:
#         from lightgbm import LGBMClassifier
#         from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#         import numpy as np
        
#         print("=" * 60)
#         print("LIGHTGBM MODEL")
#         print("=" * 60)
        
#         # Train model
#         model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
#         model.fit(X_train, y_train)
        
#         # Make predictions
#         y_pred = model.predict(X_test)
        
#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='weighted')
#         recall = recall_score(y_test, y_pred, average='weighted')
#         f1 = f1_score(y_test, y_pred, average='weighted')
        
#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")
        
#         # Feature importance
#         if hasattr(model, 'feature_importances_'):
#             feature_importance = model.feature_importances_
#             if hasattr(X_train, 'columns'):
#                 print(f"\nTop 5 Most Important Features:")
#                 indices = np.argsort(feature_importance)[-5:][::-1]
#                 for i, idx in enumerate(indices):
#                     print(f"  {i+1}. {X_train.columns[idx]}: {feature_importance[idx]:.4f}")
        
#         print("\nDetailed Classification Report:")
#         print(classification_report(y_test, y_pred))
#         print("Confusion Matrix:")
#         print(confusion_matrix(y_test, y_pred))
        
#         return {
#             'model': 'LightGBM',
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'y_pred': y_pred,
#             'model_object': model
#         }
        
#     except ImportError as e:
#         print(f"Error: Required library not installed - {e}")
#         print("Please install lightgbm: pip install lightgbm")
#         return None


# def compare_models(results):
#     """
#     Compare the performance of all trained models.
    
#     Parameters:
#     results (list): List of dictionaries containing model results
#     """
#     if not results:
#         print("No models to compare.")
#         return
    
#     print("\n" + "=" * 60)
#     print("MODEL COMPARISON SUMMARY")
#     print("=" * 60)
    
#     # Filter out None results
#     valid_results = [r for r in results if r is not None]
    
#     if not valid_results:
#         print("No valid models to compare.")
#         return
    
#     # Sort by accuracy
#     valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
#     print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
#     print("-" * 60)
    
#     for result in valid_results:
#         print(f"{result['model']:<20} {result['accuracy']:.4f}     {result['precision']:.4f}     {result['recall']:.4f}     {result['f1']:.4f}")
    
#     # Identify best model
#     best_model = valid_results[0]
#     print("\n" + "=" * 60)
#     print(f"BEST MODEL: {best_model['model']} with Accuracy: {best_model['accuracy']:.4f}")
#     print("=" * 60)
    
#     return best_model


# def main(X_train, y_train, X_test, y_test):
#     results = []
    
#     # Train and evaluate Logistic Regression
#     lr_result = logisticregression(X_train, y_train, X_test, y_test)
#     if lr_result is not None:
#         results.append(lr_result)
    
#     # Train and evaluate Random Forest
#     rf_result = randomforest(X_train, y_train, X_test, y_test)
#     if rf_result is not None:
#         results.append(rf_result)
    
#     # Train and evaluate XGBoost
#     xgb_result = XGBoost(X_train, y_train, X_test, y_test)
#     if xgb_result is not None:
#         results.append(xgb_result)
    
#     # Train and evaluate LightGBM
#     lgbm_result = LightGBM(X_train, y_train, X_test, y_test)
#     if lgbm_result is not None:
#         results.append(lgbm_result)
    
#     # Compare all models
#     best_model = compare_models(results)
    
#     return best_model

# if __name__ == "__main__": 
#     main()