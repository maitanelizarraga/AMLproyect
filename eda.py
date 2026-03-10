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


    # Fraud distribution
    ax = sns.countplot(x="fraud_label", data=df, color="#D2B48C")
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


    # Device type distribution
    # Count combinations
    counts = df.groupby(["device_type", "fraud_label"]).size().unstack(fill_value=0)

    # Plot stacked bar chart
    ax = counts.plot(kind="bar", stacked=True, color=["#3274a1", "#ff6666"])

    plt.title("Device type distribution (Fraud vs Non-Fraud)")
    plt.xlabel("Device type")
    plt.ylabel("Number of transactions")
    plt.legend(["Non-Fraud", "Fraud"])

    # Add labels
    for container in ax.containers:
        ax.bar_label(container)

    plt.show()

    

    # Location distribution (stacked fraud vs non-fraud)
    counts = df.groupby(["device_location", "fraud_label"]).size().unstack(fill_value=0)

    ax = counts.plot(kind="bar", stacked=True, color=["#3274a1", "#ff6666"])

    plt.title("Device location distribution (Fraud vs Non-Fraud)")
    plt.xlabel("Device location")
    plt.ylabel("Number of transactions")
    plt.xticks(rotation=45)
    plt.legend(["Non-Fraud", "Fraud"])

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container)

    plt.show()


    # Previous failed attempts distribution
    ax = sns.countplot(x="previous_failed_attempts", data=df, color="#D2B48C")
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


    # Payment method distribution (Fraud vs Non-Fraud)
    counts = df.groupby(["payment_mode", "fraud_label"]).size().unstack(fill_value=0)

    ax = counts.plot(kind="bar", stacked=True, color=["#3274a1", "#ff6666"])

    plt.title("Payment method distribution (Fraud vs Non-Fraud)")
    plt.xlabel("Payment method")
    plt.ylabel("Number of transactions")
    plt.xticks(rotation=45)
    plt.legend(["Non-Fraud", "Fraud"])

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container)

    plt.show()


    # Hours of the fraud distribution
    ax = sns.countplot(x="transaction_hour", data=df[df["fraud_label"] == 1], color="#ff6666")
    plt.title("Fraud count by hour")

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    plt.show()


    # Heatmap with numerical variables
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlations")
    plt.show()
        



def main():
    df = importarcsv() 
    initialinspection(df)
    df = datacleaning(df)
    eda(df)
    
if __name__ == "__main__": 
    main()
