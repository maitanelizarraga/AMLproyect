import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier

def adjust_class_weights(y_train):
    from sklearn.utils import class_weight
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print("Class Weights:", class_weights)
    return class_weights

def automatic_weights_model(X_train, y_train, X_test, y_test):
    class_weights = adjust_class_weights(y_train)
    model = LGBMClassifier(n_estimators=100, random_state=42, class_weight=class_weights, verbose=-1)
    model.fit(X_train.select_dtypes(include=[np.number]), y_train)
    y_pred = model.predict(X_test.select_dtypes(include=[np.number]))
    
    print("\nRESULTADOS MODELO CON PESOS AUTOMÁTICOS:")
    print(classification_report(y_test, y_pred))
    return f1_score(y_test, y_pred, average='weighted')

def adasyn_oversampling(X_train, y_train):
    from imblearn.over_sampling import ADASYN
    X_train_num = X_train.select_dtypes(include=[np.number])
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_train_num, y_train)
    print(f"Post-ADASYN: Fraude={sum(y_res==1)}, No-Fraude={sum(y_res==0)}")
    return X_res, y_res

def oversampling_smote_data(X_train, y_train):
    # Solo aplicamos a columnas numéricas procesadas
    X_train_num = X_train.select_dtypes(include=[np.number])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_num, y_train)
    print(f"Post-SMOTE: Fraude={sum(y_res==1)}, No-Fraude={sum(y_res==0)}")
    return X_res, y_res

def borderline_smote_oversampling(X_train, y_train):
    from imblearn.over_sampling import BorderlineSMOTE
    X_train_num = X_train.select_dtypes(include=[np.number])
    bsmote = BorderlineSMOTE(random_state=42)
    X_res, y_res = bsmote.fit_resample(X_train_num, y_train)
    print(f"Post-Borderline SMOTE: Fraude={sum(y_res==1)}, No-Fraude={sum(y_res==0)}")
    return X_res, y_res

def undersampling_random(X_train, y_train):
    X_train_num = X_train.select_dtypes(include=[np.number])
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train_num, y_train)
    print(f"Post-UnderSampling: Fraude={sum(y_res==1)}, No-Fraude={sum(y_res==0)}")
    return X_res, y_res

def balanced_random_forest(X_train, y_train, X_test, y_test):
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    brf.fit(X_train_num, y_train)
    y_pred = brf.predict(X_test_num)
    
    print("\nRESULTADOS BALANCED RANDOM FOREST:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'model': 'Balanced Random Forest',
        'accuracy': np.mean(y_test == y_pred),
        'precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
        'recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'],
        'f1': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    }
    return metrics


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