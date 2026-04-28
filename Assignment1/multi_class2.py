import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import ADASYN # Asegúrate de tener imbalanced-learn instalado
from imblearn.over_sampling import SMOTE

def create_multiclass_target(df):
    conditions = [
        (df['fraud_label'] == 0),
        (df['fraud_label'] == 1) & (df['transaction_amount'] < 25000),
        (df['fraud_label'] == 1) & (df['transaction_amount'] >= 25000)
    ]
    choices = ['A', 'B', 'C']
    df = df.copy()
    df['fraud_category'] = np.select(conditions, choices, default='A')
    return df

def entrenar_modelos_robustos(df_multi):
    df_proc = df_multi.copy()
    
    # 1. Codificación de categóricas
    le = LabelEncoder()
    cat_cols = df_proc.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        if col != 'fraud_category':
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))

    # 2. Selección de Features (ELIMINAMOS transaction_amount para evitar trampa)
    # También eliminamos los IDs y las etiquetas
    cols_to_drop = ['transaction_id', 'user_id', 'fraud_label', 'fraud_category']
    X = df_proc.drop(columns=[c for c in cols_to_drop if c in df_proc.columns])
    y = df_proc['fraud_category']

    # 3. División TRAIN/TEST (Fundamental: hacer esto ANTES del balanceo)
    # El test_size del 20% será nuestra "realidad" desbalanceada
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Balanceo SÓLO del set de Entrenamiento
    print(f"Distribución original en entrenamiento: {y_train_raw.value_counts().to_dict()}")
    
    # Usamos ADASYN para equilibrar las clases A, B y C en el training
    sampler = SMOTE(random_state=42)
    X_train_res, y_train_res = sampler.fit_resample(X_train_raw, y_train_raw)
    
    print(f"Distribución tras ADASYN en entrenamiento: {y_train_res.value_counts().to_dict()}")

    # 5. Escalado (Ajustamos con el X_train_res y aplicamos a ambos)
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_res)
    X_test_final = scaler.transform(X_test)

    # --- MODELO: RANDOM FOREST ---
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train_final, y_train_res)
    y_probs = rf.predict_proba(X_test_final)

    umbral_fraude = 0.20
    y_pred_custom = []
    
    for i in range(len(y_probs)):
        prob_A = y_probs[i][0]
        prob_B = y_probs[i][1]
        prob_C = y_probs[i][2]
        
        # Priorizamos detectar C (fraude caro), luego B, y si no hay sospecha, A
        if prob_C > umbral_fraude:
            y_pred_custom.append('C')
        elif prob_B > umbral_fraude:
            y_pred_custom.append('B')
        else:
            y_pred_custom.append('A')

    # --- RESULTADOS ---
    print("\n" + "="*55)
    print(f"REPORTE CON UMBRAL AJUSTADO ({umbral_fraude})")
    print("="*55)
    print(classification_report(y_test, y_pred_custom, zero_division=0))

def main():
    input_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv"
    try:
        df = pd.read_csv(input_path)
        df_multi = create_multiclass_target(df)
        entrenar_modelos_robustos(df_multi)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()