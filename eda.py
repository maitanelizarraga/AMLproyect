# EXPLORATORY DATA ANALYSIS

def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("Digital_Payment_Fraud_Detection_Dataset.csv") 
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
    #
    return df

def eda(df): 
    import matplotlib.pyplot as plt 
    import seaborn as sns 

    # 1. Fraud distribution
    ax = sns.countplot(x="fraud_label", data=df)
    plt.title("Fraud distribution")

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    plt.show()

    # 2. Device type distribution
    ax = sns.countplot(x="device_type", data=df)
    plt.title("Device type distribution")

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    plt.show()

    # 3. Previous failed attempts distribution
    ax = sns.countplot(x="previous_failed_attempts", data=df)
    plt.title("Previous failed attempts distribution")

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    plt.show()

    # 3. Count of payment types
    sns.countplot(y="payment_mode", data=df)
    plt.title("Payment methods")
    plt.show()

    # 4. Hours and quantities transactions
    sns.scatterplot(x="transaction_hour", y="transaction_amount", data=df)
    plt.title("Hour vs quantity transactions")
    plt.show()

    # 5. Heatmap with numerical variables
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlations")
    plt.show()
    
    # 6. Frauds by payment method
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
