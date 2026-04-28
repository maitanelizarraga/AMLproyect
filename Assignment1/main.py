import pandas as pd
import numpy as np
import eda
import induce_missingness
import imputation
import data_partition
import baselinemodels
import imbalance
import multi_class
import final_evaluation

def main():
    # First of all, we make a brief analysis to realize what we are working with
    print("--- STEP 1: DATA LOADING ---")
    #eda.main() #hecho
    
    # Secondly, we induce missing values in the dataset to simulate real-world scenarios and evaluate imputation techniques.
    print("\n--- STEP 2: INDUCING MISSING VALUES (MCAR/MAR/MNAR) ---")
    induce_missingness.main() #añadido MNAR
    
    # Thirdly, we imput the data of the missing values
    print("\n--- STEP 3: MULTIVARIATE IMPUTATION (MICE) ---")
    imputation.main() #hecho, no quedan missings

    # 4. Data Partitioning
    print("\n--- STEP 4: DATA PARTITIONING ---")
    data_partition.main() #hecho 
    
    # 5. Baseline Model Execution
    print("\n--- STEP 5: BASELINE MODEL EVALUATION ---")
    baselinemodels.main() #hecho
    
    # 6. Imbalance Treatment
    print("\n--- STEP 6: IMBALANCE TREATMENT ---")
    imbalance.main() #revisar bien
    
    #7. Multi-class Target Creation
    print("\n--- STEP 7: MULTI-CLASS TARGET CREATION ---")
    multi_class.main() #continuar haciendo

    # Re-evaluate with the best performing model (e.g., LightGBM)
    #imbalance.evaluate_balanced_model(X_res, y_res, X_test, y_test)
    final_evaluation.main()

if __name__ == "__main__":
    main()