from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda

def main():
    df = importarcsv()
    initialinspection(df)
    df = datacleaning(df)
    eda(df)
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print("\n" + "=" * 60)
    print("BASELINE MODELS")
    print("=" * 60)
    
    results = []

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()