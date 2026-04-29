import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def run_extended_eda():
    # 1. Load the cleaned data
    df = pd.read_csv("./datasets/retail_store_inventory_cleaned.csv", parse_dates=['Date'])
    
    # Set plot style
    sns.set_theme(style="whitegrid")

    # --- ANALYSIS 1: STORE-LEVEL TIME SERIES ---
    # This proves why we need to model each store separately
    df_store_temp = df.groupby(['Date', 'Store ID'])['Units Sold'].sum().reset_index()
    
    g = sns.FacetGrid(df_store_temp, col="Store ID", hue="Store ID", 
                      col_wrap=4, height=3, aspect=1.5)
    g.map(sns.lineplot, "Date", "Units Sold")
    g.set_titles("Store {col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Individual Sales Trends per Store ID', fontsize=16)
    plt.show()


    # --- ANALYSIS 2: WEEKLY SEASONALITY HEATMAP ---
    # Shows which stores are busier on specific days
    df['DayOfWeek'] = df['Date'].dt.day_name()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot_table = df.pivot_table(index='DayOfWeek', columns='Store ID', 
                                 values='Units Sold', aggfunc='mean').reindex(days)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title('Heatmap: Average Sales by Day of Week and Store')
    plt.show()


def main():
    print("--- STEP 1: EXTENDED EDA ---")
    run_extended_eda()

if __name__ == "__main__":
    run_extended_eda()