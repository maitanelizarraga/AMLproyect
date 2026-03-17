from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda 

def main():
    # Primero importamos los datos
    df = importarcsv()
    
    # Luego hacemos la inspección inicial
    initialinspection(df)
    
    # Limpiamos los datos
    df = datacleaning(df)
    
    # Hacemos el análisis exploratorio
    eda(df)
    
    # Finalmente particionamos los datos
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    # Aquí puedes continuar con el resto de tu código
    print("Datos preparados correctamente")
    print(f"Tamaño del entrenamiento: {X_train.shape}")
    print(f"Tamaño del test: {X_test.shape}")

if __name__ == "__main__": 
    main()