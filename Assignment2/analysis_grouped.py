import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_extended_eda():
    # Load the cleaned dataset
    df = pd.read_csv("./datasets/retail_store_inventory_cleaned.csv", parse_dates=['Date'])
   

    sns.set_theme(style="whitegrid")

    # --- PLOT 1: WEEKLY INTENSITY BY REGION ---
    # Grouping by Region provides clearer geographic patterns than individual stores
    df['DayOfWeek'] = df['Date'].dt.day_name()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot_region = df.pivot_table(index='DayOfWeek', columns='Region', 
                                  values='Units Sold', aggfunc='mean').reindex(days)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_region, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Avg Units Sold'})
    plt.title('Average Sales: Region vs. Day of the Week', fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: PROMOTION IMPACT BY CATEGORY ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Category', y='Units Sold', hue='Holiday/Promotion', 
                palette='muted', capsize=.1)
    
    plt.title('Promotion Effectiveness by Category', fontsize=14)
    plt.legend(title='Promotion', labels=['No', 'Yes'])
    plt.ylabel('Units Sold (Mean)')
    plt.tight_layout()
    plt.show()

    # --- PLOT 3: SEASONAL DISTRIBUTION (Boxplot) ---
    plt.figure(figsize=(12, 6))
    # Ensure chronological order of seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    sns.boxplot(data=df, x='Seasonality', y='Units Sold', hue='Seasonality',
                order=season_order, palette='Set2', legend=False)
    
    plt.title('Sales Variability by Season', fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- PLOT 4: CLIMATE-SENSITIVE CATEGORY ANALYSIS ---
    # This plot crosses 'Weather Condition' cleaning results with sales data
    plt.figure(figsize=(12, 6))
    weather_impact = df.groupby(['Category', 'Weather Condition'])['Units Sold'].mean().unstack()
    
    sns.heatmap(weather_impact, annot=True, cmap="YlOrRd", fmt=".1f")
    plt.title('Impact of Weather on Sales by Category', fontsize=14)
    plt.ylabel('Category')
    plt.xlabel('Weather Condition')
    plt.tight_layout()
    plt.show()

def main():
    print("--- GENERATING INSIGHTS FROM DATA GROUPED ---")
    run_extended_eda()

if __name__ == "__main__":
    main()