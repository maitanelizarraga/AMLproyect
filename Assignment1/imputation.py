from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np

def imputation(df):
    df_temp = df.copy()
    cols_to_exclude = ["transaction_id", "user_id", "fraud_label"]
    
    # 1. Identificación automática
    cat_cols = [c for c in df_temp.select_dtypes(include=['object', 'string']).columns if c not in cols_to_exclude]
    num_cols = [c for c in df_temp.select_dtypes(include=['number']).columns if c not in cols_to_exclude]
    
    # 2. Encoding (igual que antes)
    encoders = {}
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(object)
        le = LabelEncoder()
        mask = df_temp[col].notnull()
        df_temp.loc[mask, col] = le.fit_transform(df_temp.loc[mask, col].astype(str))
        encoders[col] = le
        df_temp[col] = df_temp[col].astype(float)

    # 3. MICE con seguimiento de iteraciones
    all_cols_to_impute = cat_cols + num_cols
    
    """
    verbose=2 will print each iteration's progress,
    and imputation_order='ascending' will start with columns that have fewer missing values,
    which can help the imputation process converge faster.
    """
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
        max_iter=20,
        random_state=42,
        add_indicator=False,
        verbose=2,  # <--- Change this to 2 for detailed iteration logs
        imputation_order='ascending' # Imputa columns with fewer missings first, which can help convergence
    )
    
    print("\n" + "="*50)
    print("INDICATING IMPUTATION PROGRESS:")
    print("="*50)
    
    imputed_data = imputer.fit_transform(df_temp[all_cols_to_impute])
    df_temp[all_cols_to_impute] = imputed_data

    # 4. REVERSE ENCODING
    for col in cat_cols:
        le = encoders[col]
        df_temp[col] = df_temp[col].round().astype(int).clip(0, len(le.classes_) - 1)
        df_temp[col] = le.inverse_transform(df_temp[col])


    nulos_finales = df_temp.isnull().sum().sum()

    print("\n" + "="*50)
    print(f"Imputation successful. Remaining nulls: {nulos_finales}")
    print("="*50)
    return df_temp

def main():
    input_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv"
    output_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv"
    
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset loaded: {df.shape} with {df.isnull().sum().sum()} nulls.")
        
        df_imput = imputation(df)
        
        df_imput.to_csv(output_path, index=False)
        print(f"Success: File saved to {output_path}")
        
    except Exception as e:
        print(f"Error during the process: {e}")

if __name__ == "__main__": 
    main()