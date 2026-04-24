import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

def create_multiclass_target(df):
    # Definimos condiciones para las clases A, B y C
    conditions = [
        (df['fraud_label'] == 0),
        (df['fraud_label'] == 1) & (df['transaction_amount'] < 500),
        (df['fraud_label'] == 1) & (df['transaction_amount'] >= 500)
    ]
    choices = ['A', 'B', 'C']
    df['fraud_category'] = np.select(conditions, choices, default='A')
    return df

def entrenar_modelos(df_multi):
    # 1. Selección de Features (X) y Target (y)
    # Es vital eliminar columnas que no son predictoras (como IDs o el label original)
    cols_to_drop = ['fraud_label', 'fraud_category']
    # Opcional: añade aquí IDs o nombres si existen, ej: 'transaction_id'
    
    X = df_multi.drop(columns=cols_to_drop)
    y = df_multi['fraud_category']

    # Convertir solo columnas categóricas necesarias a numéricas
    # y asegurarse de que el resto sean números
    X = pd.get_dummies(X, drop_first=True)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- ESTRATEGIA 1: One-vs-Rest (OvR) ---
    ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    ovr_model.fit(X_train, y_train)
    y_pred_ovr = ovr_model.predict(X_test)

    # --- ESTRATEGIA 2: Multinomial Logistic Regression (Softmax) ---
    # Nota: solver 'lbfgs' soporta multinomial por defecto
    softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    softmax_model.fit(X_train, y_train)
    y_pred_softmax = softmax_model.predict(X_test)

    # 2. Resultados
    print("\n" + "="*30)
    print("REPORTE ONE-VS-REST")
    print("="*30)
    print(classification_report(y_test, y_pred_ovr))

    print("\n" + "="*30)
    print("REPORTE MULTINOMIAL (SOFTMAX)")
    print("="*30)
    print(classification_report(y_test, y_pred_softmax))

def main():
    input_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv"
    output_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_multiclass.csv"
    
    try:
        df = pd.read_csv(input_path)
        df_multi = create_multiclass_target(df)
        
        # Guardar el dataset transformado
        df_multi.to_csv(output_path, index=False)
        print(f"Archivo guardado exitosamente en: {output_path}")
        
        # Ejecutar entrenamiento
        entrenar_modelos(df_multi)
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {input_path}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__": 
    main()