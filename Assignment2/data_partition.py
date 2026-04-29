import pandas as pd
import os

def partition_by_store(df):
    """
    Splits the already grouped data into train, validation, and test sets
    preserving the chronological order for each store.
    """
    stores = df['Store ID'].unique()
    train_list, val_list, test_list = [], [], []

    for store_id in stores:
        store_df = df[df['Store ID'] == store_id].copy()
        
        # Calculate split points
        n = len(store_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        # Partition
        train_list.append(store_df.iloc[:train_end])
        val_list.append(store_df.iloc[train_end:val_end])
        test_list.append(store_df.iloc[val_end:])

    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

def main():
    # We load the output from data_agrupation.py
    input_path = "./datasets/retail_store_grouped.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run data_agrupation.py first.")
        return

    df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')
    train_df, val_df, test_df = partition_by_store(df)

    # Save final sets
    train_df.to_csv("./datasets/train.csv")
    val_df.to_csv("./datasets/val.csv")
    test_df.to_csv("./datasets/test.csv")
    
    print("--- PARTITION SUMMARY ---")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Final datasets saved: train.csv, val.csv, test.csv")

if __name__ == "__main__":
    main()