from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np

def imputation(df):
    #We make a copy for saving then the csv
    df_temp = df.copy()
    # We exclude the columns we do not want to be used for predicting
    cols_to_exclude = ["transaction_id", "user_id", "fraud_label"]
    
    # 1. Automatic identification
    #We separate the numeric columns and the categorical columns
    cat_cols = [c for c in df_temp.select_dtypes(include=['object', 'string']).columns if c not in cols_to_exclude]
    num_cols = [c for c in df_temp.select_dtypes(include=['number']).columns if c not in cols_to_exclude]
    
    # 2. Encoding 
    #We convert the categorical variables to numbers for the imputation.
    encoders = {}
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(object) # We ensure the column is treat as an object
        le = LabelEncoder() #We instance th encoder
        mask = df_temp[col].notnull() #For saving(True/false) only the rows that contain a value(not null)
        df_temp.loc[mask, col] = le.fit_transform(df_temp.loc[mask, col].astype(str)) #We encode the column
        encoders[col] = le # We save the encoder for later
        df_temp[col] = df_temp[col].astype(float) #We convert the columns to float

    # 3. MICE

    all_cols_to_impute = cat_cols + num_cols # We add the number of variables to imput

    # We define the imputer
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), #number of trees per null
        max_iter=20, #Number of iterations
        random_state=42,#seed
        add_indicator=False, #Fill the value(do not create a new column)
        verbose=2, # for prints
        imputation_order='ascending' # First impute the columns with the fewest nulls
    )
    
    print("\n" + "="*50)
    print("STARTING THE CHARGE PROCESS STEP BY STEP")
    print("="*50)
    
    #We train the models and impute the data, until we have no nulls
    imputed_data = imputer.fit_transform(df_temp[all_cols_to_impute])
    df_temp[all_cols_to_impute] = imputed_data #From matriz to data frame

    # 4. Revertir encoding
    for col in cat_cols:
        le = encoders[col] #We get the encoder we saved before
        df_temp[col] = df_temp[col].round().astype(int).clip(0, len(le.classes_) - 1) # We ensure we obtain the int numbers(without decimals)
        df_temp[col] = le.inverse_transform(df_temp[col]) # We transform it back to the category

    #We count the number of nulls, for seeing if there is any left
    nulos_finales = df_temp.isnull().sum().sum()

    print("\n" + "="*50)
    print(f"Imputation successfully completed. Remaining nulls:{nulos_finales}")
    print("="*50)
    return df_temp

def validate_imputation_focused(df_orig, df_missing, df_imputed):
    """
    Compare the original dataset to the imputed dataset, 
    but ONLY in the columns where missing values ​​existed
    """
    print("\n" + "="*85)
    print("VALIDATION REPORT: IMPACT OF IMPUTATION (ONLY COLUMNS WITH NULL VALUES)")
    print("="*85)
    
    # 1. We identified only the columns that had nulls before imputation.
    cols_with_nans = [c for c in df_missing.columns if df_missing[c].isnull().any()]
    
    results = []
    for col in cols_with_nans:
        # --- CASE FOR NUMERICAL VARIABLES ---
        if pd.api.types.is_numeric_dtype(df_orig[col]):
            m_orig = df_orig[col].mean()
            m_imp  = df_imputed[col].mean()
            
            # We calculate the Relative Percentage Error
            diff_mean = abs(m_orig - m_imp) / m_orig * 100
            
            results.append({
                "Column": col,
                "Type": "Numeric",
                "Metrics": "Mean",
                "Original": round(m_orig, 2),
                "Imputed": round(m_imp, 2),
                "Error": f"{round(diff_mean, 2)}%",
                "Quality": "OK" if diff_mean < 10 else "BIAS"
            })
            
        # --- CASE FOR CATEGORICAL VARIABLES (TEXT) ---
        else:
            #We identify the mode in the original dataset
            mode_orig = df_orig[col].mode()[0]

            #We calculate the frquency of that mode in the original and imputed dataset
            freq_orig = (df_orig[col] == mode_orig).mean() * 100
            freq_imp  = (df_imputed[col] == mode_orig).mean() * 100

            #We analyze if the mode has changed
            mode_imp  = df_imputed[col].mode()[0]
            change_alert = "" if mode_orig == mode_imp else f" -> NEW MODE: {mode_imp}"
                   
            diff_freq = abs(freq_orig - freq_imp)

            results.append({
                "Column": col,
                "Type": "Categoric",
                "Metrics": f"Mode:{mode_orig}{change_alert}",
                "Original": f"{round(freq_orig, 1)}% freq",
                "Imputed": f"{round(freq_imp, 1)}% freq",
                "Error": f"{round(diff_freq, 2)}%",
                "Quality": "OK" if diff_freq < 10 else "BIAS"
            })

    if not results:
        print("No columns with nulls were found for comparison.")
        return None

    summary_df = pd.DataFrame(results).set_index("Column")
    print(summary_df.to_string())
    print("="*85)
    
    return summary_df

def main():

    input_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv"
    output_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv"
    original_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset.csv"

    
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset loaded: {df.shape} with {df.isnull().sum().sum()} nulls.")
        
        df_imput = imputation(df)
        
        df_imput.to_csv(output_path, index=False)
        print(f"Success: File saved in {output_path}")

        df_original = pd.read_csv(original_path)
        df_inputed= pd.read_csv(output_path)
        validate_imputation_focused( df_original,df, df_inputed)
        
    except Exception as e:
        print(f"Error during the process: {e}")

if __name__ == "__main__": 
    main()