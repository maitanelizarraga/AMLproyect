import data_partition
import eda
import adv_stat_models
import data_agrupation
import baseline_models
import lts_model
import chronos_model

import pandas as pd
from tabulate import tabulate
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def imprimir_tabla_desde_archivos():
    df_base = pd.read_csv("./results/baseline_results.csv")
    df_adv  = pd.read_csv("./results/adv_stat_results.csv")
    df_lstm = pd.read_csv("./results/lstm_results.csv")
    df_chro = pd.read_csv("./results/chronos_results.csv")

    print("\n" + "="*60)
    print("FINAL COMPARISON OF MODELS (by aggregation level)")
    print("="*60)

    # --- STORE LEVEL ---
    print("\nSTORE LEVEL (5 stores)")
    store_models = pd.DataFrame([
        {"Model": "SARIMAX", "MAE": df_adv['SARIMAX_MAE'].mean()},
        {"Model": "Holt-Winters", "MAE": df_base[(df_base['Type']=='Store') & (df_base['Model']=='Holt-Winters')]['MAE'].mean()},
        {"Model": "SARIMA", "MAE": df_adv['SARIMA_MAE'].mean()},
        {"Model": "Naive", "MAE": df_base[(df_base['Type']=='Store') & (df_base['Model']=='Naive')]['MAE'].mean()}
    ]).sort_values("MAE")
    print(tabulate(store_models, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
    print(f"Best Store Model: {store_models.iloc[0]['Model']}")

    # --- PRODUCT LEVEL ---
    print("\nPRODUCT LEVEL (20 products)")
    product_models = pd.DataFrame([
        {"Model": "LSTM", "MAE": df_lstm['MAE'].mean()},
        {"Model": "Chronos", "MAE": df_chro['MAE'].mean()},
        {"Model": "Holt-Winters", "MAE": df_base[(df_base['Type']=='Product') & (df_base['Model']=='Holt-Winters')]['MAE'].mean()},
        {"Model": "ARIMA", "MAE": df_base[(df_base['Type']=='Product') & (df_base['Model']=='ARIMA')]['MAE'].mean()},
        {"Model": "Naive", "MAE": df_base[(df_base['Type']=='Product') & (df_base['Model']=='Naive')]['MAE'].mean()}
    ]).sort_values("MAE")
    print(tabulate(product_models, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
    print(f"Best Product Model: {product_models.iloc[0]['Model']}")

    # --- REGION LEVEL ---
    print("\nREGION LEVEL (4 regions)")
    region_models = df_base[df_base['Type']=='Region'].groupby('Model')['MAE'].mean().reset_index().sort_values("MAE")
    print(tabulate(region_models, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
    print(f"Best Region Model: {region_models.iloc[0]['Model']}")

    # --- CATEGORY LEVEL ---
    print("\nCATEGORY LEVEL (5 categories)")
    category_models = df_base[df_base['Type']=='Category'].groupby('Model')['MAE'].mean().reset_index().sort_values("MAE")
    print(tabulate(category_models, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
    print(f"Best Category Model: {category_models.iloc[0]['Model']}")

    print("\n" + "="*60)
    print("SARIMAX wins for stores (uses exogenous variables).")
    print("LSTM beats Chronos for products (specialized training).")
    print("="*60)

def main():
    set_seed(42)
    # 1.Eda and data preparation
    print("\n--- STEP 1: EDA AND DATA PREPARATION ---")
    eda.main() 
    
    # 2.Data agrupation
    print("\n--- STEP 2: DATA AGRUPATION ---")
    data_agrupation.main()
    
    # 3.Data partitioning
    print("\n--- STEP 3: DATA PARTITIONING ---")
    data_partition.main() 
    
    # 4. Baseline Models
    print("\n--- STEP 4: BASELINE MODELS ---")
    baseline_models.main()

    # 5. Advanced Statistical Models (SARIMA & SARIMAX)
    print("\n--- STEP 5: ADVANCED STATISTICAL MODELS ---")
    adv_stat_models.main() 

    # 6. LSTM
    print("\n--- STEP 6: LSTM ---")
    lts_model.main()

    # 7. Chronos (zero-shot)
    print("\n--- STEP 7: CHRONOS (zero-shot) ---")
    chronos_model.main()

    # 8. Final comparative table
    print("\n--- STEP 8: FINAL COMPARATIVE TABLE ---")
    imprimir_tabla_desde_archivos()


    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()