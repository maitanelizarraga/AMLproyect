# EXPLORATORY DATA ANALYSIS

def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("retail_store_inventory.csv") 
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
    import pandas as pd
    #as we have seen, it do not recognice the date, so we convert it and make sure its correct
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data types:" + "\n" + str(df.dtypes)) 
    print(" ")

    #as we have seen, there are not missing values so more cleaning is not needed
    return df


def eda(df): 
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    
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
    # --- 1. MATRIZ DE CORRELACIÓN ---
    # Para ver qué variables influyen más en las "Units Sold"
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlación entre Variables Numéricas')
    plt.show()

    # --- 2. SERIES TEMPORALES POR SEPARADO (FACET GRID) ---
    # 1. Agrupamos los datos
    df_cat_temp = df.groupby(['Date', 'Category'])['Units Sold'].sum().reset_index()

    # 2. Creamos la cuadrícula (una columna por cada categoría)
    # 'col_wrap=3' hará que se muestren 3 gráficas por fila
    g = sns.FacetGrid(df_cat_temp, col="Category", hue="Category", 
                      col_wrap=3, height=4, aspect=1.5)

    # 3. Dibujamos las líneas en cada cuadro
    g.map(sns.lineplot, "Date", "Units Sold")

    # 4. Ajustes estéticos
    g.set_axis_labels("Fecha", "Unidades Vendidas")
    g.set_titles("{col_name}") # Pone el nombre de la categoría como título de cada cuadro
    
    # Rotamos las fechas de cada gráfica para que se lean bien
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Tendencia de Ventas por Categoría (Vista Individual)', fontsize=16)
    
    plt.show()

    # --- 3. VENTAS POR CATEGORÍA Y TEMPORADA ---
    # Útil para entender qué productos se mueven más según el clima/estación
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Category', y='Units Sold', hue='Seasonality', palette='viridis')
    plt.title('Ventas Promedio por Categoría y Estación')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # --- 4. VENTAS POR REGIÓN ---
    # Para identificar si hay regiones con mejor desempeño
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Region', y='Units Sold', palette='viridis')
    plt.title('Ventas Promedio por Región')
    plt.xticks(rotation=45)
    plt.show()

    

        




def datapartitioning(df):
    
    # we make sure that the data is recogniced as time and we sort it by date, as its crucial for time series analysis
    df = df.sort_values('Date').reset_index(drop=True)
    

    X = df.drop("Units Sold", axis=1)
    y = df["Units Sold"]
    
    
    # we calculate the split index for 80% training and 20% testing as it can not be randomly split due to the temporal nature of the data
    test_size = int(len(df) * 0.2)
    split_index = len(df) - test_size
    
    # separation of the data taking into account the temporal order
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training from {df['Date'].iloc[0]} to {df['Date'].iloc[split_index-1]}")
    print(f"Testing from {df['Date'].iloc[split_index]} to {df['Date'].iloc[-1]}")
    
    return X_train, X_test, y_train, y_test