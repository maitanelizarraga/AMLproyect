import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from lightgbm import LGBMClassifier

def load_partitioned_data():
    train = pd.read_csv("./datasets/train.csv")
    val = pd.read_csv("./datasets/val.csv")
    test = pd.read_csv("./datasets/test.csv")
    
    # Separate X and y
    X_train, y_train = train.drop("fraud_label", axis=1), train["fraud_label"]
    X_val, y_val = val.drop("fraud_label", axis=1), val["fraud_label"]
    X_test, y_test = test.drop("fraud_label", axis=1), test["fraud_label"]
    
    # Filter only numeric (necessary for ADASYN/SMOTE)
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_on_validation(X_train_res, y_train_res, X_val, y_val, method_name):
    print(f"\n" + "="*40)
    print(f"METHOD: {method_name}")
    print(f"="*40)
    
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_val)
    
    print(classification_report(y_val, y_pred))
    return f1_score(y_val, y_pred, average='weighted')

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_partitioned_data()
    results = {}

    # --- TECHNIQUE 1: Class Weights ---
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(zip(np.unique(y_train), weights))
    model_cw = LGBMClassifier(n_estimators=100, class_weight=cw_dict, random_state=42, verbose=-1)
    model_cw.fit(X_train, y_train)
    results['Class Weights'] = f1_score(y_val, model_cw.predict(X_val), average='weighted')

    # --- TECHNIQUE 2: SMOTE ---
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    results['SMOTE'] = evaluate_on_validation(X_smote, y_smote, X_val, y_val, "SMOTE")

    # --- TECHNIQUE 3: ADASYN (AND DATA SAVING) ---
    adasyn = ADASYN(random_state=42)
    X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
    results['ADASYN'] = evaluate_on_validation(X_ada, y_ada, X_val, y_val, "ADASYN")

    # We save only adasyn because it's the one that worked best, and it also created artificial data that helps improve model precision
    # RECONSTRUCTION AND SAVING OF ADASYN DATASET
    # Combine X and y into a single DataFrame
    df_adasyn = pd.concat([pd.DataFrame(X_ada), pd.Series(y_ada, name='fraud_label')], axis=1)
    output_path = "./datasets/train_balanced_adasyn.csv"
    df_adasyn.to_csv(output_path, index=False)
    print(f"\n[INFO] Balanced ADASYN training dataset saved to: {output_path}")

    # --- TECHNIQUE 4: Random UnderSampling ---
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    results['UnderSampling'] = evaluate_on_validation(X_rus, y_rus, X_val, y_val, "UnderSampling")

    # --- TECHNIQUE 5: Balanced Random Forest ---
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, sampling_strategy='all', replacement=True)
    brf.fit(X_train, y_train)
    results['Balanced RF'] = f1_score(y_val, brf.predict(X_val), average='weighted')

    # --- FINAL COMPARISON ---
    print("\n" + "!"*30)
    print("F1-SCORE SUMMARY (VALIDATION)")
    for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<20}: {score:.4f}")

if __name__ == "__main__":
    main()