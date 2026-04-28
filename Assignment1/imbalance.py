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
    
    # Define columns
    cat_cols = ['transaction_type', 'payment_mode', 'device_type']
    num_cols = [
        'account_age_days', 'previous_failed_attempts', 'is_international', 
        'ip_risk_score', 'login_attempts_last_24h', 'transaction_amount', 
        'transaction_hour', 'avg_transaction_amount'
    ] 
    # 1. Seplit X and y
    X_train, y_train = train.drop("fraud_label", axis=1), train["fraud_label"]
    X_val, y_val = val.drop("fraud_label", axis=1), val["fraud_label"]
    X_test, y_test = test.drop("fraud_label", axis=1), test["fraud_label"]

    # 2. One-Hot Encoding (OHE)
    X_train = pd.get_dummies(X_train[cat_cols + num_cols], columns=cat_cols, dtype=int)
    X_val = pd.get_dummies(X_val[cat_cols + num_cols], columns=cat_cols, dtype=int)
    X_test = pd.get_dummies(X_test[cat_cols + num_cols], columns=cat_cols, dtype=int)

    # 3. Align columns
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 4. Scale
    for col in X_train.columns:
        mx, mn = X_train[col].max(), X_train[col].min()
        if mx > mn:
            X_train[col] = (X_train[col] - mn) / (mx - mn)
            X_val[col] = (X_val[col] - mn) / (mx - mn)
            X_test[col] = (X_test[col] - mn) / (mx - mn)

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

def run_final_test(best_method_name, X_train, y_train, X_test, y_test, X_smote=None, y_smote=None, X_ada=None, y_ada=None, X_rus=None, y_rus=None, cw_dict=None):
    """
    Recibe el nombre del ganador y todos los posibles datasets para ejecutar el test final.
    """
    # 1. Internal mapping: we associate each name with its data
    data_mapping = {
        'SMOTE': (X_smote, y_smote),
        'ADASYN': (X_ada, y_ada),
        'UnderSampling': (X_rus, y_rus),
        'Class Weights': (X_train, y_train),
        'Balanced RF': (X_train, y_train)
    }

    # 2. Data selection
    X_f, y_f = data_mapping[best_method_name]
    
    print(f"\n" + "*"*60)
    print(f" RUNNING FINAL TEST WITH: {best_method_name}")
    print("*"*60)

    # 3. Setup and training
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    
    if best_method_name == 'Class Weights' and cw_dict:
        model.set_params(class_weight=cw_dict)

    model.fit(X_f, y_f)
    
    # 4. Prediction and Reporting
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

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

    # Take the best model
    best_method_name = sorted(results.items(), key=lambda x: x[1], reverse=True)[0][0]
    run_final_test(best_method_name, X_train, y_train, X_test, y_test,X_smote=X_smote, y_smote=y_smote, X_ada=X_ada, y_ada=y_ada, X_rus=X_rus, y_rus=y_rus, cw_dict=cw_dict)
   

if __name__ == "__main__":
    main()