from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda 

def main():
    # We import the dataset using the function we created in eda.py
    df = importarcsv()
    
    # We perform the initial inspection
    initialinspection(df)
    
    # We clean the data
    df = datacleaning(df)
    
    # We perform exploratory data analysis
    eda(df)
    
    # We partition the data
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

if __name__ == "__main__": 
    main()