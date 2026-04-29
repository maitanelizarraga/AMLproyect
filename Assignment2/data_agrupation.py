import pandas as pd
import os

def group_and_clean_data(df):
    """
    Groups data by Store ID, aggregates multiple products per date to avoid duplicates,
    and fills temporal gaps with daily frequency.
    """
    stores = df['Store ID'].unique()
    grouped_list = []

    for store_id in stores:
        # 1. Filter by specific store
        store_df = df[df['Store ID'] == store_id].copy()
        store_df['Date'] = pd.to_datetime(store_df['Date'])
        
        # 2. Aggregate products per day (sum sales, average prices)
        store_df = store_df.groupby('Date').agg({
            'Units Sold': 'sum',
            'Inventory Level': 'sum',
            'Price': 'mean',
            'Discount': 'max',
            'Competitor Pricing': 'mean',
            'Store ID': 'first',
            'Region': 'first'
        })
        
        # 3. Apply daily frequency and fill gaps
        store_df = store_df.sort_index()
        store_df = store_df.asfreq('D')
        
        # Fill missing values: 0 for sales, forward fill for context
        store_df['Units Sold'] = store_df['Units Sold'].fillna(0)
        cols_to_ffill = ['Store ID', 'Price', 'Region']
        store_df[cols_to_ffill] = store_df[cols_to_ffill].ffill()
        store_df = store_df.fillna(0)
        
        grouped_list.append(store_df)

    return pd.concat(grouped_list)

def main():
    input_path = "./datasets/retail_store_inventory_cleaned.csv"
    output_path = "./datasets/retail_store_grouped.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    df_grouped = group_and_clean_data(df)
    
    df_grouped.to_csv(output_path)
    print(f"Agrupation complete. File saved as {output_path}")

if __name__ == "__main__":
    main()