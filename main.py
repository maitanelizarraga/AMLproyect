from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda
from baselinemodels import numericalfeatures, logisticregression, randomforest, XGBoost

def main():
    # Primero importamos los datos
    df = importarcsv()
    
    # Luego hacemos la inspección inicial
    initialinspection(df)
    
    # Limpiamos los datos
    df = datacleaning(df)
    
    # Hacemos el análisis exploratorio (opcional)
    # eda(df)
    
    # Particionamos los datos
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELOS BASELINE")
    print("=" * 60)
    
    # Lista para guardar resultados
    results = []

    # Seleccionar solo características numéricas
    X_train_num, X_test_num = numericalfeatures(X_train, X_test)  # <-- Cambiar df por X_train, X_test

    print("\n1. Entrenando Logistic Regression...")
    results.append(logisticregression(X_train_num, y_train, X_test_num, y_test))
    
    print("\n2. Entrenando Random Forest...")
    results.append(randomforest(X_train_num, y_train, X_test_num, y_test))
    
    print("\n3. Entrenando XGBoost...")
    results.append(XGBoost(X_train, y_train, X_test, y_test))  # XGBoost puede usar más features
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__": 
    main()