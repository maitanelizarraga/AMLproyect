import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier

def load_partitioned_data():
    train = pd.read_csv("./Assignment1/datasets/train.csv")
    val = pd.read_csv("./Assignment1/datasets/val.csv")
    test = pd.read_csv("./Assignment1/datasets/test.csv")
    
    # split x and y 
    X_train, y_train = train.drop("fraud_label", axis=1), train["fraud_label"]
    X_val, y_val = val.drop("fraud_label", axis=1), val["fraud_label"]
    X_test, y_test = test.drop("fraud_label", axis=1), test["fraud_label"]
    
    # filter only numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_on_validation(X_train_res, y_train_res, X_val, y_val, method_name):
    #trains with balanced data and evaluates on the original VALIDATION set.
    print(f"\n" + "="*40)
    print(f"MÉTODO: {method_name}")
    print(f"="*40)
    
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_val)
    
    print(classification_report(y_val, y_pred))
    return f1_score(y_val, y_pred, average='weighted')

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_partitioned_data()
    
    results = {}

    # TECHNIQUE 1: Class Weights  ---
    from sklearn.utils import class_weight
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(zip(np.unique(y_train), weights))
    
    model_cw = LGBMClassifier(n_estimators=100, class_weight=cw_dict, random_state=42, verbose=-1)
    model_cw.fit(X_train, y_train)
    results['Class Weights'] = f1_score(y_val, model_cw.predict(X_val), average='weighted')
    print("\nResultados Class Weights guardados.")

    # --- TECHNIQUE 2: SMOTE ---
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    results['SMOTE'] = evaluate_on_validation(X_smote, y_smote, X_val, y_val, "SMOTE")

    # --- TECHNIQUE 3: ADASYN ---
    adasyn = ADASYN(random_state=42)
    X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
    results['ADASYN'] = evaluate_on_validation(X_ada, y_ada, X_val, y_val, "ADASYN")

    # --- TECHNIQUE 4: Random UnderSampling ---
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    results['UnderSampling'] = evaluate_on_validation(X_rus, y_rus, X_val, y_val, "UnderSampling")

    # --- TECHNIQUE 5: Balanced Random Forest ---
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, sampling_strategy='all', replacement=True)
    brf.fit(X_train, y_train)
    y_pred_brf = brf.predict(X_val)
    results['Balanced RF'] = f1_score(y_val, y_pred_brf, average='weighted')
    print("\nResultados Balanced RF guardados.")

    # --- FINAL COMPARATION ---
    print("\n" + "!"*30)
    print("SUMMARY DE F1-SCORE ")
    for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<20}: {score:.4f}")

if __name__ == "__main__":
    main()
    