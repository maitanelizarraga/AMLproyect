import pandas as pd
from sklearn.model_selection import train_test_split

def datapartitioning(df): 
    """
    Divide el dataset en 70% Train, 15% Validation y 15% Test
    manteniendo la proporción de la variable objetivo (fraud_label).
    """
    # 1. Definir variables
    X = df.drop("fraud_label", axis=1)
    y = df["fraud_label"]
    
    # 2. Primera división: 70% entrenamiento, 30% para el resto (temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42, 
        stratify=y
    )
    
    # 3. Segunda división: El 30% restante se divide al 50% (0.5 de 0.3 = 0.15 cada uno)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    df = pd.read_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv")

    # Ejecutar partición
    X_train, X_val, X_test, y_train, y_val, y_test = datapartitioning(df)

    # Mostrar reporte de tamaños
    print("--- RESUMEN DE PARTICIÓN ---")
    print(f"Total registros: {len(df)}")
    print(f"Entrenamiento (70%): {len(X_train)} - Fraude: {y_train.sum()}")
    print(f"Validación    (15%): {len(X_val)}  - Fraude: {y_val.sum()}")
    print(f"Test          (15%): {len(X_test)}  - Fraude: {y_test.sum()}")
    print("----------------------------\n")

    
    pd.concat([X_train, y_train], axis=1).to_csv(f"./Assignment1/datasets/train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(f"./Assignment1/datasets/val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"./Assignment1/datasets/test.csv", index=False)
    
    print(f"¡Éxito! Archivos guardados en ./Assignment1/datasets/")

if __name__ == "__main__": 
    main()