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
    #as we have seen, there are not missing values so the cleaning is not needed
    return df

def eda(df): 
    import matplotlib.pyplot as plt 
    import seaborn as sns 


        




def datapartitioning(df):
    import pandas as pd
    
    # we make sure that the data is recogniced as time and we sort it by date, as its crucial for time series analysis
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    y = df["Units Sold"]
    
    exogenous_cols = [
        "Price", "Discount", "Weather Condition", 
        "Holiday/Promotion", "Competitor Pricing", "Seasonality"
    ]
    X = df[exogenous_cols]
    
    # we calculate the split index for 80% training and 20% testing as it can not be randomly split due to the temporal nature of the data
    test_size = int(len(df) * 0.2)
    split_index = len(df) - test_size
    
    # separation of the data taking into account the temporal order
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training from {df['Date'].iloc[0]} hasta {df['Date'].iloc[split_index-1]}")
    print(f"Testing from {df['Date'].iloc[split_index]} hasta {df['Date'].iloc[-1]}")
    
    return X_train, X_test, y_train, y_test