from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

#hay todavia missing values

def imputation(df):
    df_temp = df.copy()
    
    cols_to_exclude = ["transaction_id", "user_id", "fraud_label"]
    numeric_cols = df_temp.select_dtypes(include=['int64', 'float64']).columns
    cols_to_impute = [c for c in numeric_cols if c not in cols_to_exclude]
    
    imputer = IterativeImputer(max_iter=10, random_state=42)
    
    df_temp[cols_to_impute] = imputer.fit_transform(df_temp[cols_to_impute])
    
    print("Imputation complete. Remaining null values:", df_temp.isnull().sum().sum())
    
    return df_temp


def main():
    df = pd.read_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv")
    print(f"Original shape:  {df.shape}")
    print(f"Missing values:  {df.isnull().sum().sum()}\n")

    df_imput = imputation(df)

    df_imput.to_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv", index=False)
    print("Saved to: datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv")

if __name__ == "__main__": 
    main()