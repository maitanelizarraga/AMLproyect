from eda import importarcsv, datapartitioning, initialinspection, datacleaning, eda
import baselinemodels

def main():
    # 1. Carga y Análisis exploratorio
    df = importarcsv()
    initialinspection(df)
    df = datacleaning(df)
    
    # Comenta la línea de abajo si no quieres ver los gráficos cada vez
    # eda(df) 
    
    # 2. Partición de datos
    X_train, X_test, y_train, y_test = datapartitioning(df)
    
    print("\n" + "=" * 60)
    print("EJECUTANDO MODELOS BASE")
    print("=" * 60)
    
    # 3. Ejecución de modelos (baselinemodels.main ya hace el filtrado numérico)
    best_model = baselinemodels.main(X_train, y_train, X_test, y_test)
    
    if best_model:
        print("\n" + "=" * 60)
        print(f"PROCESO COMPLETADO. El mejor modelo fue: {best_model['model']}")
        print("=" * 60)

if __name__ == "__main__": 
    main()