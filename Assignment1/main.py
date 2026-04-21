import pandas as pd
import numpy as np
from eda import importcsv, datapartitioning, initialinspection, datacleaning
from induce_missingness import induce_missingness, run_diagnostics
import baselinemodels
import imbalance

def main():
    # 1. Load and Inspection
    print("--- STEP 1: DATA LOADING ---")
    df = importcsv()
    initialinspection(df)
    df = datacleaning(df)
    
    # 2. Missingness Induction (AML Requirement)
    print("\n--- STEP 2: INDUCING MISSING VALUES (MCAR/MAR) ---")
    df_missing = induce_missingness(df, seed=42)
    run_diagnostics(df_missing, df)
    
    # 3. Advanced Imputation (MICE)
    print("\n--- STEP 3: MULTIVARIATE IMPUTATION (MICE) ---")
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    # Select numerical columns for MICE (excluding target and IDs)
    cols_to_exclude = ["transaction_id", "user_id", "fraud_label"]
    numeric_cols = df_missing.select_dtypes(include=['int64', 'float64']).columns
    cols_to_impute = [c for c in numeric_cols if c not in cols_to_exclude]
    
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df_missing[cols_to_impute] = imputer.fit_transform(df_missing[cols_to_impute])
    print("Imputation complete. Remaining null values:", df_missing.isnull().sum().sum())

    # 4. Data Partitioning
    print("\n--- STEP 4: DATA PARTITIONING (80/20) ---")
    X_train, X_test, y_train, y_test = datapartitioning(df_missing)
    
    # 5. Baseline Model Execution
    print("\n--- STEP 5: BASELINE MODEL EVALUATION ---")
    results_base = baselinemodels.main(X_train, y_train, X_test, y_test)
    
    # 6. Imbalance Treatment
    print("\n--- STEP 6: IMBALANCE TREATMENT ---")
    # Example using SMOTE (Oversampling)
    print("\nTraining with SMOTE...")
    X_res, y_res = imbalance.oversampling_smote_data(X_train, y_train)
    
    # Re-evaluate with the best performing model (e.g., LightGBM)
    imbalance.evaluate_balanced_model(X_res, y_res, X_test, y_test)

if __name__ == "__main__":
    main()