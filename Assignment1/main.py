from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda
from induce_missingness import induce_missingness, run_diagnostics
import baselinemodels

def main():
    # 1. Data loading, inspection, cleaning, and EDA
    df = importarcsv()
    initialinspection(df)
    df = datacleaning(df)
    
    # Commented out to save time during development, but you can re-enable it to see the EDA results.
    # eda(df)

    # 2. Induce missingness and run diagnostics
    df = induce_missingness(df, seed=42)
    run_diagnostics(df, df)  # optional: remove if you don't want the table every run

    #TODO: DATA IMPUTATION
    
    # 3. Data partitioning
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print("\n" + "=" * 60)
    print("EJECUTANDO MODELOS BASE")
    print("=" * 60)
    
    # 4. Model training and evaluation
    best_model = baselinemodels.main(X_train, y_train, X_test, y_test)
    
    if best_model:
        print("\n" + "=" * 60)
        print(f"PROCESO COMPLETADO. El mejor modelo fue: {best_model['model']}")
        print("=" * 60)

if __name__ == "__main__": 
    main()