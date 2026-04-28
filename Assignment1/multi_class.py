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
    Creates three categories:
    A: No Fraud
    B: Low-Amount Fraud (< 25,000)
    C: High-Amount Fraud (>= 25,000)
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
    # 1. Feature selection (only numeric for scaling)
    # We exclude the original label and new category from the training set X
    X = df_multi.select_dtypes(include=[np.number])
    cols_to_drop = ['fraud_label', 'fraud_category']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    y = df_multi['fraud_category']

    # 2. Scaling (Vital for Logistic Regression, optional for RF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training with {X.shape[1]} scaled features...")

    # --- MODEL 1: RANDOM FOREST (OVR) ---
    # We use OneVsRest although RF is natively multiclass, to maintain symmetry in the strategy
    ovr_rf = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    )
    ovr_rf.fit(X_train, y_train)
    y_pred_rf = ovr_rf.predict(X_test)

    # --- MODEL 2: LOGISTIC REGRESSION (MULTINOMIAL) ---
    # Multinomial uses Softmax function to calculate probabilities among the 3 classes
    softmax_logit = LogisticRegression( 
        solver='lbfgs', 
        max_iter=2000, 
        class_weight='balanced',
        random_state=42
    )
    softmax_logit.fit(X_train, y_train)
    y_pred_logit = softmax_logit.predict(X_test)

    # --- RESULTS ---
    print("\n" + "="*50)
    print("REPORT: RANDOM FOREST (OVR STRATEGY)")
    print("="*50)
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    print("\n" + "="*50)
    print("REPORT: LOGISTIC REGRESSION (MULTINOMIAL/SOFTMAX)")
    print("="*50)
    print(classification_report(y_test, y_pred_logit, zero_division=0))

def main():
    # Note: Make sure the path is correct according to your working folder
    input_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv"
    output_path = "./datasets/Digital_Payment_Fraud_Detection_Dataset_multiclass.csv"
    
    try:
        df = pd.read_csv(input_path)
        df_multi = create_multiclass_target(df)
        
        # Save the transformed dataset for audit
        df_multi.to_csv(output_path, index=False)
        print(f"File saved at: {output_path}")
        
        entrenar_modelos(df_multi)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")

if __name__ == "__main__": 
    main()