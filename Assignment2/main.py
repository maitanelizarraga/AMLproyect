from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda

def main():
    # Primero importamos los datos
    df = importarcsv()
    
    # Luego hacemos la inspección inicial
    initialinspection(df)
    
    # Limpiamos los datos
    df = datacleaning(df)
    
    # Hacemos el análisis exploratorio (opcional)
    eda(df)
    
    # Particionamos los datos
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELOS BASELINE")
    print("=" * 60)
    
    # Lista para guardar resultados
    results = []

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__": 
    main()