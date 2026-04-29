# EXPLORATORY DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def importarcsv(): 
    df = pd.read_csv("./datasets/retail_store_inventory.csv")
    #we make sure that the dataset is correctly imported 
    #we can see the different columns names 
    print(df.iloc[0]) 
    return df

def initialinspection(df): 
    print("ANALYSIS:")
    print("First row:" + "\n" + str(df.iloc[0])) 
    print(" ")
    print("Data types:" + "\n" + str(df.dtypes)) 
    print(" ")
    print("Shape:" + "\n" + str(df.shape)) 
    print(" ")
    print("Missing values:" + "\n" + str(df.isnull().sum())) 
    print(" ")
    print("Unique values:" + "\n" + str(df.nunique()))
    print(" ")

def datacleaning(df): 
    #as we have seen, it do not recognice the date, so we convert it and make sure its correct
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data types:" + "\n" + str(df.dtypes)) 
    print(" ")

    #GUARDAR EL DATASET LIMPIO
    output_path = "./datasets/retail_store_inventory_cleaned.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Dataset saved into: {output_path}")
    print(" ")

    #There are not missing values so more cleaning is not needed
    return df

def eda(df): 

    print("OUTLIERS ANALYSIS")
    sns.set_theme(style="whitegrid")
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()
    print("/n")

    print("VISUALIZATION OF THE DATA")
    # CORRELATION MATRIX 
    # To see wich variable affects more tu the units sold column
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Numerical variables correlation')
    plt.show()

    # TEMPORAL SERIES
    # we group the data
    df_cat_temp = df.groupby(['Date', 'Category'])['Units Sold'].sum().reset_index()

    # we make the grid
    # 'col_wrap=3' makes three plots per row
    g = sns.FacetGrid(df_cat_temp, col="Category", hue="Category", 
                      col_wrap=3, height=4, aspect=1.5)

    # make lines for each plot
    g.map(sns.lineplot, "Date", "Units Sold")

    # stetical adjustments
    g.set_axis_labels("Fecha", "Unidades Vendidas")
    g.set_titles("{col_name}") #name of category as title
    
    # we rotate the date for reading it better
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Tendencies of sales per category', fontsize=16)
    
    plt.show()

    # SALES BY SEASON AND CATEGORY 
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Category', y='Units Sold', hue='Seasonality', palette='viridis')
    plt.title('Average Sales by Category and Season')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # SALES BY REGION
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Region', y='Units Sold', palette='viridis')
    plt.title('Average Sales by Region')
    plt.xticks(rotation=45)
    plt.show()


###############Time Plots ####################

    # we group by date to see the general tendencies
    df_daily = df.groupby('Date')['Units Sold'].sum().asfreq('D').fillna(0)

    # STATIONAL DESCOMPOSITION
    # we use an additive model because we have seen that the variance is quite stable over time
    decomposition = seasonal_decompose(df_daily, model='additive', period=30) #period 30 for month
    
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Seasonal Decomposition: Trend, Seasonality and Residuals', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # STATIONARITY ANALYSIS 
    rolmean = df_daily.rolling(window=7).mean()
    rolstd = df_daily.rolling(window=7).std()

    plt.figure(figsize=(12, 6))
    plt.plot(df_daily, color='blue', label='Original', alpha=0.3)
    plt.plot(rolmean, color='red', label='Rolling Mean (7d)')
    plt.plot(rolstd, color='black', label='Rolling Std Dev (7d)')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #IMPACT OF EXOGENOUS VARIABLES (Promotions Boxplot) 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Holiday/Promotion', y='Units Sold', palette='Set2')
    plt.title('Impact of Promotions on Units Sold')
    plt.xticks([0, 1], ['No Promo', 'With Promo'])
    plt.show()
     
def main():
    # Import dataset
    df = importarcsv()
    initialinspection(df)
    datacleaning(df)
    eda(df)

if __name__ == "__main__": 
    main()