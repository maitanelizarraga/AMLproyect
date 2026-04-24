import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler 

def create_multiclass_target(df):
    """
    Crea tres categorías:
    A: No Fraude
    B: Fraude de monto bajo (< 25,000)
    C: Fraude de monto alto (>= 25,000)
    """
    conditions = [
        (df['fraud_label'] == 0),
        (df['fraud_label'] == 1) & (df['transaction_amount'] < 25000),
        (df['fraud_label'] == 1) & (df['transaction_amount'] >= 25000)
    ]
    choices = ['A', 'B', 'C']
    df['fraud_category'] = np.select(conditions, choices, default='A')
    return df

def entrenar_modelos(df_multi):
    # 1. Selección de features (solo numéricas para el escalado)
    # Excluimos el label original y la nueva categoría del set de entrenamiento X
    X = df_multi.select_dtypes(include=[np.number])
    cols_to_drop = ['fraud_label', 'fraud_category']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    y = df_multi['fraud_category']

    # 2. Escalado (Vital para Logistic Regression, opcional para RF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Entrenando con {X.shape[1]} características escaladas...")

    # --- MODELO 1: RANDOM FOREST (OVR) ---
    # Usamos OneVsRest aunque RF es multiclase nativo, para mantener simetría en la estrategia
    ovr_rf = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    )
    ovr_rf.fit(X_train, y_train)
    y_pred_rf = ovr_rf.predict(X_test)

    # --- MODELO 2: LOGISTIC REGRESSION (MULTINOMIAL) ---
    # Multinomial usa la función Softmax para calcular probabilidades entre las 3 clases
    softmax_logit = LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=2000, 
        class_weight='balanced',
        random_state=42
    )
    softmax_logit.fit(X_train, y_train)
    y_pred_logit = softmax_logit.predict(X_test)

    # --- RESULTADOS ---
    print("\n" + "="*50)
    print("REPORTE: RANDOM FOREST (ESTRATEGIA OVR)")
    print("="*50)
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    print("\n" + "="*50)
    print("REPORTE: LOGISTIC REGRESSION (MULTINOMIAL/SOFTMAX)")
    print("="*50)
    print(classification_report(y_test, y_pred_logit, zero_division=0))

def main():
    # Nota: Asegúrate de que la ruta sea correcta según tu carpeta de trabajo
    input_path = "./Assignment1/datasets/train_balanced_adasyn.csv"
    output_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_multiclass.csv"
    
    try:
        df = pd.read_csv(input_path)
        df_multi = create_multiclass_target(df)
        
        # Guardar el dataset transformado para auditoría
        df_multi.to_csv(output_path, index=False)
        print(f"Archivo guardado en: {output_path}")
        
        entrenar_modelos(df_multi)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {input_path}")

if __name__ == "__main__": 
    main()