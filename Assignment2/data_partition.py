import pandas as pd
import os

def partition_by_store(df):
    """
    Splits the already grouped data into train, validation, and test sets
    preserving the chronological order for each region.
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

def partition_by_region(df):
    """
    Splits the already grouped data into train, validation, and test sets
    preserving the chronological order for each store.
    """
    regions = df['Region'].unique()
    train_list, val_list, test_list = [], [], []

    for region in regions:
        region_df = df[df['Region'] == region].copy()
        
        # Calculate split points
        n = len(region_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        # Partition
        train_list.append(region_df.iloc[:train_end])
        val_list.append(region_df.iloc[train_end:val_end])
        test_list.append(region_df.iloc[val_end:])

    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

def partition_by_category(df):
    """
    Splits the already grouped data into train, validation, and test sets
    preserving the chronological order for each category.
    """
    categories = df['Category'].unique()
    train_list, val_list, test_list = [], [], []

    for category in categories:
        category_df = df[df['Category'] == category].copy()
        
        # Calculate split points
        n = len(category_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        # Partition
        train_list.append(category_df.iloc[:train_end])
        val_list.append(category_df.iloc[train_end:val_end])
        test_list.append(category_df.iloc[val_end:])

    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

def partition_by_product(df):
    """
    Splits the already grouped data into train, validation, and test sets
    preserving the chronological order for each product.
    """
    Products = df['Product ID'].unique()
    train_list, val_list, test_list = [], [], []

    for product in Products:
        product_df = df[df['Product ID'] == product].copy()
        
        # Calculate split points
        n = len(product_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        # Partition
        train_list.append(product_df.iloc[:train_end])
        val_list.append(product_df.iloc[train_end:val_end])
        test_list.append(product_df.iloc[val_end:])

    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)


def main():
    # We load the data
    input_path_store = "./datasets/retail_store_grouped_by_store.csv"
    input_path_region = "./datasets/retail_store_grouped_by_region.csv"
    input_path_category = "./datasets/retail_store_grouped_by_category.csv"
    input_path_product = "./datasets/retail_store_grouped_by_product.csv"
    
    if not os.path.exists(input_path_store):
        print(f"Error: {input_path_store} not found. Run data_agrupation.py first.")
        return
    if not os.path.exists(input_path_region):
        print(f"Error: {input_path_region} not found. Run data_agrupation.py first.")
        return
    if not os.path.exists(input_path_category):
        print(f"Error: {input_path_category} not found. Run data_agrupation.py first.")
        return
    if not os.path.exists(input_path_product):
        print(f"Error: {input_path_product} not found. Run data_agrupation.py first.")
        return

    df_store = pd.read_csv(input_path_store, parse_dates=['Date'], index_col='Date')
    train_store, val_store, test_store = partition_by_store(df_store)
    df_region = pd.read_csv(input_path_region, parse_dates=['Date'], index_col='Date')
    train_region, val_region, test_region = partition_by_region(df_region)
    df_category = pd.read_csv(input_path_category, parse_dates=['Date'], index_col='Date')
    train_category, val_category, test_category = partition_by_category(df_category)
    df_product = pd.read_csv(input_path_product, parse_dates=['Date'], index_col='Date')
    train_product, val_product, test_product = partition_by_product(df_product)

    # Save final sets for store
    train_store.to_csv("./datasets/train_Store.csv")
    val_store.to_csv("./datasets/val_Store.csv")
    test_store.to_csv("./datasets/test_Store.csv")

    # Save final sets for region
    train_region.to_csv("./datasets/train_region.csv")
    val_region.to_csv("./datasets/val_region.csv")
    test_region.to_csv("./datasets/test_region.csv")

    # Save final sets for category
    train_category.to_csv("./datasets/train_category.csv")
    val_category.to_csv("./datasets/val_category.csv")
    test_category.to_csv("./datasets/test_category.csv")

    # Save final sets for product
    train_product.to_csv("./datasets/train_product.csv")
    val_product.to_csv("./datasets/val_product.csv")
    test_product.to_csv("./datasets/test_product.csv")
    
    print("--- PARTITION SUMMARY STORES ---")
    print(f"Train: {len(train_store)} | Val: {len(val_store)} | Test: {len(test_store)}")
    print("Final datasets saved: train_Store.csv, val_Store.csv, test_Store.csv")

    print("--- PARTITION SUMMARY REGIONS ---")
    print(f"Train: {len(train_region)} | Val: {len(val_region)} | Test: {len(test_region)}")
    print("Final datasets saved: train_region.csv, val_region.csv, test_region.csv")

    print("--- PARTITION SUMMARY CATEGORYS ---")
    print(f"Train: {len(train_region)} | Val: {len(val_region)} | Test: {len(test_region)}")
    print("Final datasets saved: train_category.csv, val_category.csv, test_category.csv")

    print("--- PARTITION SUMMARY PRODUCTS---")
    print(f"Train: {len(train_product)} | Val: {len(val_product)} | Test: {len(test_product)}")
    print("Final datasets saved: train_product.csv, val_product.csv, test_product.csv")

if __name__ == "__main__":
    main()