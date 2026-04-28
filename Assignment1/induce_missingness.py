"""
Two missing-data mechanisms are supported:
  - MCAR  Missing Completely At Random  (uniform random removal)
  - MAR   Missing At Random             (removal conditioned on another column)

"""

import pandas as pd
import numpy as np

PROTECTED_COLS = {"transaction_id", "user_id", "fraud_label"}

# 1. MCAR: Fallos técnicos aleatorios (ahora en más columnas)
DEFAULT_MCAR = [
    {"column": "ip_risk_score",    "rate": 0.10},
    {"column": "account_age_days", "rate": 0.05},
    {"column": "device_type",      "rate": 0.03}, # Fallo en el registro del user-agent
]

# 2. MAR: Pérdida dependiente de otras variables observadas
DEFAULT_MAR = [
    {
        "column":      "previous_failed_attempts",
        "cond_column": "device_type",
        "cond_value":  "Android",
        "rate":        0.15,
    },
    {
        "column":      "payment_mode",
        "cond_column": "device_location",
        "cond_value":  "Hyderabad",
        "rate":        0.20,
    },
    {
        # Nuevo: Gente mayor (account_age alta) usa menos apps y el campo login_attempts falla
        "column":      "login_attempts_last_24h",
        "cond_column": "is_international",
        "cond_value":  1,
        "rate":        0.30,
    }
]

# 3. MNAR: El valor se pierde por el valor MISMO de la columna
# Esto es muy común en fraude: los valores extremos tienden a ocultarse
DEFAULT_MNAR = [
    {
        "column": "transaction_amount", 
        "threshold_pct": 0.85, # El 15% de las transacciones más altas tienden a no registrar el monto
        "rate": 0.40 
    },
    {
        "column": "avg_transaction_amount",
        "threshold_pct": 0.10, # El 10% de las cuentas con menos saldo ocultan su promedio por privacidad
        "rate": 0.30
    }
]

def induce_missingness(df, seed=42, mcar_configs=None, mar_configs=None, mnar_configs=None, verbose=True):
    df_out = df.copy()
    rng = np.random.default_rng(seed)

    if mcar_configs is None: mcar_configs = DEFAULT_MCAR
    if mar_configs is None: mar_configs = DEFAULT_MAR
    if mnar_configs is None: mnar_configs = DEFAULT_MNAR

    if verbose:
        print("\n--- Iniciando inducción de valores faltantes ---")

    # --- MCAR ---
    for cfg in mcar_configs:
        col, rate = cfg["column"], cfg["rate"]
        n = int(rate * len(df_out))
        idx = rng.choice(df_out.index, size=n, replace=False)
        df_out.loc[idx, col] = np.nan
        if verbose:
            print(f"[MCAR] '{col}': {n} rows -> NaN (aleatorio puro, rate={rate*100:.0f}%)")

    # --- MAR ---
    for cfg in mar_configs:
        col, cond_col, cond_value, rate = cfg["column"], cfg["cond_column"], cfg["cond_value"], cfg["rate"]
        cond_idx = df_out[df_out[cond_col] == cond_value].index
        n = int(rate * len(cond_idx))
        idx = rng.choice(cond_idx, size=n, replace=False)
        df_out.loc[idx, col] = np.nan
        if verbose:
            print(f"[MAR ] '{col}': {n} rows -> NaN (condicionado a {cond_col}=='{cond_value}')")

    # --- MNAR ---
    for cfg in mnar_configs:
        col, thresh_pct, rate = cfg["column"], cfg["threshold_pct"], cfg["rate"]
        threshold_val = df_out[col].quantile(thresh_pct)
        
        if thresh_pct > 0.5:
            target_idx = df_out[df_out[col] >= threshold_val].index
            cond_desc = f"valores >= p{int(thresh_pct*100)}"
        else:
            target_idx = df_out[df_out[col] <= threshold_val].index
            cond_desc = f"valores <= p{int(thresh_pct*100)}"
            
        n = int(rate * len(target_idx))
        idx = rng.choice(target_idx, size=n, replace=False)
        df_out.loc[idx, col] = np.nan
        if verbose:
            print(f"[MNAR] '{col}': {n} rows -> NaN (basado en {cond_desc})")

    return df_out


def run_diagnostics(df_missing, df_original):
    """
    Analiza la calidad del dataset tras inducir nulos.
    Muestra el porcentaje de pérdida y detecta sesgos entre clases (Fraude vs No Fraude).
    """
    # Identificamos columnas que tienen nulos
    cols = [c for c in df_missing.columns if df_missing[c].isnull().any()]

    rows = []
    for col in cols:
        overall = df_missing[col].isnull().mean() * 100
        
        # Calculamos la tasa de nulos por cada clase de fraude
        by_class = df_missing.groupby("fraud_label")[col].apply(lambda s: s.isnull().mean() * 100)
        fraud_0 = by_class.get(0, 0.0)
        fraud_1 = by_class.get(1, 0.0)
        
        # El Delta nos dice si la pérdida de datos es "justa" o si afecta más al fraude
        delta = abs(fraud_1 - fraud_0)
        
        # Identificamos el mecanismo probable (esto es para tu documentación)
        mechanism = "MCAR/MAR"
        if col in [cfg["column"] for cfg in DEFAULT_MNAR]:
            mechanism = "MNAR (Monto)"
            
        rows.append({
            "column":           col,
            "mechanism":        mechanism,
            "overall_missing%": round(overall, 2),
            "missing%_fraud=0": round(fraud_0, 2),
            "missing%_fraud=1": round(fraud_1, 2),
            "class_delta":      round(delta, 2),
            "warning":          "ALTO SESGO" if delta > 5 else "OK",
        })

    summary = pd.DataFrame(rows).set_index("column")

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO AVANZADO DE VALORES FALTANTES (MCAR, MAR, MNAR)")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)
    print(f"Total de nulos generados: {df_missing.isnull().sum().sum()}")
    print("Nota: Un class_delta alto indica que el modelo tendrá dificultades para aprender de esa clase.\n")

    return summary

def main():
    input_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset.csv"
    output_path = "./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_missing.csv"
    
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset original cargado: {df.shape}")

        # Aplicamos la inducción con los 3 mecanismos
        df_missing = induce_missingness(df, seed=42)
        
        # Ejecutamos el diagnóstico mejorado
        run_diagnostics(df_missing, df)

        # Guardamos
        df_missing.to_csv(output_path, index=False)
        print(f"Dataset con nulos guardado en: {output_path}")

    except Exception as e:
        print(f"Error en el proceso: {e}")

if __name__ == "__main__": 
    main()


