# EXPLORATORY DATA ANALYSIS

def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("Digital_Payment_Fraud_Detection_Dataset.csv") 
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
    # As we have seen, there are not missing values so the cleaning is not needed
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


    #HEATMAP: Locations vs payment methods(Only fraud)
    fraud_df = df[df['fraud_label'] == 1]
    pivot_fraud = fraud_df.pivot_table(index='device_location', 
                                    columns='payment_mode', 
                                    values='transaction_id', 
                                    aggfunc='count').fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_fraud, annot=True, fmt='g', cmap='YlOrRd')
    plt.title('Concentration of fraud: Location vs Payment methods')
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

    # Correlation with numerical variables
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlations")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='fraud_label', y='transaction_amount', data=df, palette='viridis')
    plt.title('Distribution of Amounts: Legitimate (0) vs Fraudulent (1)')
    plt.show()

 
def datapartitioning(df):
    import pandas as pd
    import numpy as np
    
    # Fix the seed for reproducibility
    np.random.seed(42)
    
    # Separate features and target variable
    X = df.drop("fraud_label", axis=1)
    y = df["fraud_label"]
    
    # We create a random permutation of the indices of the dataset
    indices = np.random.permutation(len(df))
    
    # We store the size of the test set (20% of the dataset)
    test_size = int(len(df) * 0.2)
    
    # We separate the indices for the test and training sets
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Create the training and test sets
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test
