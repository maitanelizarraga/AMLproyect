# EXPLORATORY DATA ANALYSIS

def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("Digital_Payment_Fraud_Detection_Dataset.csv") 
    print(df.iloc[0]) 
    return df

def initialinspection(df): 
    print(df.columns) 
    print(df.dtypes) 
    print(df.shape) 
    print(df.isnull().sum()) 
    print(df.nunique())

def datacleaning(df): 
    # Eliminamos solo columnas que existan
    cols_to_drop = [col for col in ["transaction_id"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Rellenamos valores nulos en la columna correcta
    if "transaction_amount" in df.columns:
        df["transaction_amount"] = df["transaction_amount"].fillna(df["transaction_amount"].mean())

    return df

def eda(df): 
    import matplotlib.pyplot as plt 
    import seaborn as sns 

    #Fraud distribution
    sns.countplot(x="fraud_label", data=df)
    plt.title("Fraud distribution")
    plt.show()

    #Transaction amount distribution
    sns.histplot(df["transaction_amount"], bins=40)
    plt.title("Transaction amount distribution")
    plt.show()

    #Count of payment types
    sns.countplot(y="payment_mode", data=df)
    plt.title("Payment methods")
    plt.show()

    #Hours and quantities transactions
    sns.scatterplot(x="transaction_hour", y="transaction_amount", data=df)
    plt.title("Hour vs quantity transactions")
    plt.show()

    #Heatmap with numerical variables
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlations")
    plt.show()
    
    #Frauds by payment method
    sns.countplot(x="payment_mode", data=df[df["fraud_label"] == 1])
    plt.title("Frauds by payment method")
    plt.xlabel("Payment method")
    plt.ylabel("Number of frauds")
    plt.xticks(rotation=45)
    plt.show()



def main():
    df = importarcsv() 
    initialinspection(df)
    df = datacleaning(df)
    eda(df)
    
if __name__ == "__main__": 
    main()
