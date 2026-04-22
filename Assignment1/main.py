import pandas as pd
import numpy as np
import eda
import induce_missingness
import imputation
import data_partition
import baselinemodels
import imbalance

def main():
    # First of all, we make a brief analysis to realize what we are working with
    print("--- STEP 1: DATA LOADING ---")
    eda.main()
    
    # Secondly, we induce missing values in the dataset to simulate real-world scenarios and evaluate imputation techniques.
    print("\n--- STEP 2: INDUCING MISSING VALUES (MCAR/MAR) ---")
    induce_missingness.main()
    
    # Thirdly, we imput the data of the missing values
    print("\n--- STEP 3: MULTIVARIATE IMPUTATION (MICE) ---")
    imputation.main()

    # 4. Data Partitioning
    print("\n--- STEP 4: DATA PARTITIONING (80/20) ---")
    data_partition.main()
    
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