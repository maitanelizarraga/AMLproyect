
def importarcsv(): 
    import pandas as pd 
    df = pd.read_csv("Digital_Payment_Fraud_Detection_Dataset.csv") 
    #we make sure that the dataset is correctly imported 
    #we can see the different columns names 
    print(df.iloc[0]) 
    return df

def datapartitioning(df):
    from sklearn.model_selection import train_test_split 
    #we separate the features and the target variable 
    #X is for not fraud rows and y is for fraud rows 
    X = df.drop("is_fraud", axis=1) 
    y = df["is_fraud"] 
    #we split the dataset into training and testing sets 
    #the random_state value indicates to always get the same split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    return X_train, X_test, y_train, y_test 

def datacleaning(df): 
    #we check for missing values 
    print(df.isnull().sum()) 
    #we can see that there are no missing values in the dataset 
    #we can also check for duplicates 
    print(df.duplicated().sum()) 
    #we can see that there are no duplicates in the dataset 
    return df 

def main():
    df = importarcsv() 
    df = datacleaning(df) 
    
if __name__ == "__main__": 
    main()