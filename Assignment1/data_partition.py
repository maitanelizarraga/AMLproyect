import pandas as pd
from sklearn.model_selection import train_test_split

def datapartitioning(df): 
    #Splits the dataset into 70% Train, 15% Validation, and 15% Test while maintaining the proportion of the target variable (fraud_label).

    # 1. Define variables
    X = df.drop("fraud_label", axis=1)
    y = df["fraud_label"]
    
    # 2. First split: 70% for training, 30% for the rest (temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42, 
        stratify=y
    )
       
    # 3. Second split: The remaining 30% is split 50/50 (0.5 of 0.3 = 0.15 each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    df = pd.read_csv("./datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv")

    # Execute partitioning
    X_train, X_val, X_test, y_train, y_val, y_test = datapartitioning(df)

    # Display size report
    print("--- PARTITION SUMMARY ---")
    print(f"Total records: {len(df)}")
    print(f"Training   (70%): {len(X_train)} - Fraud cases: {y_train.sum()}")
    print(f"Validation (15%): {len(X_val)}  - Fraud cases: {y_val.sum()}")
    print(f"Test       (15%): {len(X_test)}  - Fraud cases: {y_test.sum()}")
    print("----------------------------\n")

    # Save to CSV
    pd.concat([X_train, y_train], axis=1).to_csv(f"./datasets/train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(f"./datasets/val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"./datasets/test.csv", index=False)
    
    print("Files saved in ./datasets/")

if __name__ == "__main__": 
    main()