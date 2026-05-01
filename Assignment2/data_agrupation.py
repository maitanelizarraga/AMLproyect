import pandas as pd
import os

def group_by_store(df):
    """
    Groups data by Store ID, aggregates multiple products per date to avoid duplicates,
    and fills temporal gaps with daily frequency.
    """
    stores = df['Store ID'].unique()
    grouped_list = []

    for store_id in stores:
        # 1. Filter by specific store
        store_df = df[df['Store ID'] == store_id].copy()   
        # 2. Aggregate products per day
        store_df = store_df.groupby('Date').agg({
            'Store ID': 'first',
            'Units Sold': 'sum',
            'Inventory Level': 'sum',
            'Units Ordered': 'sum',
            'Price': 'mean',
            'Discount': 'max',
            'Competitor Pricing': 'mean',
            'Region': 'first',
            'Weather Condition': 'first',
            'Seasonality': 'first'
        })
        
        # 3. Apply daily frequency and fill gaps
        store_df = store_df.sort_index()
        store_df = store_df.asfreq('D')
        
        # Fill missing values: 0 for sales, forward fill for context
        store_df['Units Sold'] = store_df['Units Sold'].fillna(0)
        cols_to_ffill = ['Store ID', 'Price', 'Region','Inventory Level','Weather Condition', 'Seasonality','Discount','Competitor Pricing','Units Ordered']
        store_df[cols_to_ffill] = store_df[cols_to_ffill].ffill().bfill()
        store_df = store_df.fillna(0)
        
        grouped_list.append(store_df)
    return pd.concat(grouped_list).round(2)

def group_by_region(df):
    """
    Groups data by Region, aggregates multiple products per date to avoid duplicates,
    and fills temporal gaps with daily frequency.
    """
    regions = df['Region'].unique()
    grouped_list = []

    for region in regions:
        # 1. Filter by specific region
        region_df = df[df['Region'] == region].copy()
        
        # 2. Aggregate products per day
        region_df = region_df.groupby('Date').agg({
            'Region': 'first',
            'Units Sold': 'sum',
            'Inventory Level': 'sum',
            'Units Ordered': 'sum',
            'Price': 'mean',
            'Discount': 'max',
            'Competitor Pricing': 'mean',
            'Seasonality': 'first'
        })
        
        # 3. Apply daily frequency and fill gaps
        region_df = region_df.sort_index().asfreq('D')
        
       # Fill missing values: 0 for sales, forward fill for context
        region_df['Units Sold'] = region_df['Units Sold'].fillna(0)
        cols_to_fill = ['Region', 'Price', 'Inventory Level', 'Seasonality', 'Discount','Units Ordered']
        region_df[cols_to_fill] = region_df[cols_to_fill].ffill().bfill()
        
        grouped_list.append(region_df)

    return pd.concat(grouped_list).round(2)

def group_by_category(df):
    """
    Groups data by Category, aggregates multiple products per date to avoid duplicates,
    and fills temporal gaps with daily frequency.
    """
    categories = df['Category'].unique()
    grouped_list = []

    for cat in categories:
        # 1. Filter by specific category
        cat_df = df[df['Category'] == cat].copy()

        # 2. Aggregate products per day
        cat_df = cat_df.groupby('Date').agg({
            'Category': 'first',
            'Units Sold': 'sum',
            'Inventory Level': 'sum',
            'Units Ordered': 'sum',
            'Price': 'mean',
            'Discount': 'max',
            'Seasonality': 'first'
        })
        
        # 3. Apply daily frequency and fill gaps
        cat_df = cat_df.sort_index().asfreq('D')
        
        # Fill missing values: 0 for sales, forward fill for context
        cat_df['Units Sold'] = cat_df['Units Sold'].fillna(0)
        cols_to_fill = ['Category', 'Price', 'Inventory Level', 'Seasonality', 'Discount','Units Ordered']
        cat_df[cols_to_fill] = cat_df[cols_to_fill].ffill().bfill()
        
        grouped_list.append(cat_df)

    return pd.concat(grouped_list).round(2)

def group_by_product(df):
    """
    Groups data by Product ID, aggregates multiple products per date to avoid duplicates,
    and fills temporal gaps with daily frequency.
    """
    products = df['Product ID'].unique()
    grouped_list = []

    for prod in products:
        # 1. Filter by specific product
        prod_df = df[df['Product ID'] == prod].copy()
        
        # 2. Aggregate products per day
        prod_df = prod_df.groupby('Date').agg({
            'Product ID': 'first',
            'Category': 'first',
            'Units Sold': 'sum',
            'Inventory Level': 'sum',
            'Units Ordered': 'sum',
            'Price': 'mean',
            'Discount': 'max',
            'Seasonality': 'first'
        })
        
        # 3. Apply daily frequency and fill gaps
        prod_df = prod_df.sort_index().asfreq('D')
        
        # Fill missing values: 0 for sales, forward fill for context
        prod_df['Units Sold'] = prod_df['Units Sold'].fillna(0)
        cols_to_fill = ['Product ID', 'Category', 'Price', 'Inventory Level', 'Seasonality', 'Discount','Units Ordered']
        prod_df[cols_to_fill] = prod_df[cols_to_fill].ffill().bfill()
        
        grouped_list.append(prod_df)

    return pd.concat(grouped_list).round(2)

def main():

    input_path = "./datasets/retail_store_inventory_cleaned.csv"
    output_path_store = "./datasets/retail_store_grouped_by_store.csv"
    output_path_region = "./datasets/retail_store_grouped_by_region.csv"
    output_path_category = "./datasets/retail_store_grouped_by_category.csv"
    output_path_product = "./datasets/retail_store_grouped_by_product.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path,parse_dates=['Date'])
    df_grouped_store = group_by_store(df)
    df_grouped_region = group_by_region(df)
    df_grouped_category = group_by_category(df)
    df_grouped_product = group_by_product(df)
       
    df_grouped_store.to_csv(output_path_store)
    df_grouped_region.to_csv(output_path_region)
    df_grouped_category.to_csv(output_path_category)
    df_grouped_product.to_csv(output_path_product)

    print(f"Agrupation complete. File saved as {output_path_store}")
    print(f"Agrupation complete. File saved as {output_path_region}")
    print(f"Agrupation complete. File saved as {output_path_category}")
    print(f"Agrupation complete. File saved as {output_path_product}")

if __name__ == "__main__":
    main()