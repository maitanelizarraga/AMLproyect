import pandas as pd
import numpy as np

"""
Three missing-data mechanisms are supported:
  - MCAR  Missing Completely At Random  (uniform random removal)
  - MAR   Missing At Random             (removal conditioned on another column)
  - MNAR Missing Not At Random          (removal conditioned on the value itself)

"""

PROTECTED_COLS = {"transaction_id", "user_id", "fraud_label"}

# 1. MCAR: Random technical failures

DEFAULT_MCAR = [
    {"column": "ip_risk_score",    "rate": 0.10}, #10%
    {"column": "account_age_days", "rate": 0.05}, #5%
    {"column": "device_type",      "rate": 0.03}, #3%
]

# 2. MAR: Loss dependent on other observed variables
DEFAULT_MAR = [
    {
        # We remove valuess from previous_failed_attempts depending on device_type
        "column":      "previous_failed_attempts",
        "cond_column": "device_type",
        "cond_value":  "Android",
        "rate":        0.15, #15%
    },
    {
         # We remove valuess from payment_mode depending on device_location
        "column":      "payment_mode",
        "cond_column": "device_location",
        "cond_value":  "Hyderabad",
        "rate":        0.20, #20%
    },
    {
        # We remove valuess from login_attempts_last_24h depending on is_internacional
        "column":      "login_attempts_last_24h",
        "cond_column": "is_international",
        "cond_value":  1,
        "rate":        0.30, #30%
    }
]

# 3. MNAR: The value is lost due to the same value in the column.
DEFAULT_MNAR = [
    {
        "column": "transaction_amount", 
        "threshold_pct": 0.85, 
        "rate": 0.40 # 40%
    },
    {
        "column": "avg_transaction_amount",
        "threshold_pct": 0.10,
        "rate": 0.30 #30%
    }
]
 # Once we have defined the variables and the conditions, we remove them.
def induce_missingness(df, seed=42, mcar_configs=None, mar_configs=None, mnar_configs=None, verbose=True):
    #We make a copy for saving then the csv with the missings
    df_out = df.copy()
    rng = np.random.default_rng(seed) # We set the seed for the random generator

    #We introduce the variables we defined before
    if mcar_configs is None: mcar_configs = DEFAULT_MCAR
    if mar_configs is None: mar_configs = DEFAULT_MAR
    if mnar_configs is None: mnar_configs = DEFAULT_MNAR

    if verbose:
        print("\n--- Initiating induction of missing values ---")

    # --- MCAR ---
    for cfg in mcar_configs:
        col, rate = cfg["column"], cfg["rate"] 
        n = int(rate * len(df_out)) #Calculate the number of values to remove
        idx = rng.choice(df_out.index, size=n, replace=False) # Randomly select them
        df_out.loc[idx, col] = np.nan #Substitue/remove them
        if verbose:
            print(f"[MCAR] '{col}': {n} rows -> NaN (Completly random, rate={rate*100:.0f}%)")

    # --- MAR ---
    for cfg in mar_configs:
        col, cond_col, cond_value, rate = cfg["column"], cfg["cond_column"], cfg["cond_value"], cfg["rate"]
        cond_idx = df_out[df_out[cond_col] == cond_value].index # We get the indexes of the values that fullfill the condition
        n = int(rate * len(cond_idx)) # Calculate the number of values to remove depnding on the amount of indexes
        idx = rng.choice(cond_idx, size=n, replace=False) #Randomly select them
        df_out.loc[idx, col] = np.nan #Substitue/remove them
        if verbose:
            print(f"[MAR] '{col}': {n} rows -> NaN (Conditioned by {cond_col}=='{cond_value}')")

    # --- MNAR ---
    for cfg in mnar_configs:
        col, thresh_pct, rate = cfg["column"], cfg["threshold_pct"], cfg["rate"]
        threshold_val = df_out[col].quantile(thresh_pct) #We get the threshold of the column(The numeric value)
        
        if thresh_pct > 0.5: # If we want to remove the high values
            target_idx = df_out[df_out[col] >= threshold_val].index # We get the indexes of the values that are higher than the threshold
            cond_desc = f"values >= {int(threshold_val)}"
        else: #If we want to remove the low values
            target_idx = df_out[df_out[col] <= threshold_val].index # We get the indexes of the values that are lower than the threshold
            cond_desc = f"values <= {int(threshold_val)}"
            
        n = int(rate * len(target_idx)) # Calculate the number of values to remove
        idx = rng.choice(target_idx, size=n, replace=False)  # Select them randomly
        df_out.loc[idx, col] = np.nan #Substitue/remove them
        if verbose:
            print(f"[MNAR] '{col}': {n} rows -> NaN (based on {cond_desc})")

    return df_out

def run_diagnostics(df_missing, df_original):
    """
    Analyzes the quality of the dataset after inducing null values.
    Displays the percentage of data loss and detects biases between classes (Fraud vs. Non-Fraud).
    """
    # We identify columns that contain missing values
    cols = [c for c in df_missing.columns if df_missing[c].isnull().any()]

    rows = []
    for col in cols:
        overall = df_missing[col].isnull().mean() * 100 # We calculate the percentage of nulls
        
        #Mean of the variable with and without missings.
        if pd.api.types.is_numeric_dtype(df_original[col]): #We look if the column is numeric
            mean_orig = round(df_original[col].mean(),2)
            mean_miss = round(df_missing[col].mean(),2)
        else:
            mean_orig = "N/A (String)"
            mean_miss = "N/A(String)" 

        #We calculate the null rate for each type of fraud
        by_class = df_missing.groupby("fraud_label")[col].apply(lambda s: s.isnull().mean() * 100)
        fraud_0 = by_class.get(0, 0.0)
        fraud_1 = by_class.get(1, 0.0)
        
        #Delta tells us whether the data loss is "fair" or whether it affects fraud more.
        delta = abs(fraud_1 - fraud_0)
        
        # We identified the exact mechanism by comparing it to the configuration lists
        if col in [cfg["column"] for cfg in DEFAULT_MNAR]:
            mechanism = "MNAR"
        elif col in [cfg["column"] for cfg in DEFAULT_MAR]:
            mechanism = "MAR"
        elif col in [cfg["column"] for cfg in DEFAULT_MCAR]:
            mechanism = "MCAR"
        else:
            mechanism = "Unknown" 
            
        rows.append({
            "column":           col,
            "mechanism":        mechanism,
            "overall_missing %": round(overall, 2),
            "missing%_fraud=0": round(fraud_0, 2),
            "missing%_fraud=1": round(fraud_1, 2),
            "class_delta":      round(delta, 2),
            "warning":          "HIGH BIAS" if delta > 5 else "OK",
            "Original mean":    mean_orig,
            "Missings mean":    mean_miss
        })

    summary = pd.DataFrame(rows).set_index("column")

    print("\n" + "=" * 80)
    print("ADVANCED DIAGNOSIS OF MISSING VALUES (MCAR, MAR, MNAR)")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)
    print(f"Total number of nulls generated: {df_missing.isnull().sum().sum()}")
    print("Note: A high class_delta indicates that the model will have difficulty learning from that class.\n")

    return summary

def main():
    input_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset.csv"
    output_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv"
    
    try:
        df = pd.read_csv(input_path)
        print(f"Original dataset loaded: {df.shape}")

        # We applied induction with the 3 mechanisms
        df_missing = induce_missingness(df, seed=42)
        
        # We execute the diagnostics
        run_diagnostics(df_missing, df)

        # Save the csv
        df_missing.to_csv(output_path, index=False)
        print(f"Dataset with nulls saved in: {output_path}")

    except Exception as e:
        print(f"Error in the process: {e}")

if __name__ == "__main__": 
    main()


