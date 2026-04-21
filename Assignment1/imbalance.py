import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier

def oversampling_smote_data(X_train, y_train):
    # Solo aplicamos a columnas numéricas procesadas
    X_train_num = X_train.select_dtypes(include=[np.number])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_num, y_train)
    print(f"Post-SMOTE: Fraude={sum(y_res==1)}, No-Fraude={sum(y_res==0)}")
    return X_res, y_res

def evaluate_balanced_model(X_train, y_train, X_test, y_test):
    # Evaluamos usando LightGBM que suele ser el mejor en fraude
    X_test_num = X_test.select_dtypes(include=[np.number])
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_num)
    
    print("\nRESULTADOS MODELO BALANCEADO:")
    print(classification_report(y_test, y_pred))
    return f1_score(y_test, y_pred, average='weighted')

def balanced_rf_model(X_train, y_train, X_test, y_test):
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    brf.fit(X_train_num, y_train)
    y_pred = brf.predict(X_test_num)
    print("Balanced Random Forest F1-Score:", f1_score(y_test, y_pred, average='weighted'))