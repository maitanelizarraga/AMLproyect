import data_partition
import eda
import adv_stat_models
import analysis_grouped
import data_agrupation
import baseline_models
import lts_model
import chronos_model

import pandas as pd
from tabulate import tabulate

def imprimir_tabla_desde_archivos():
    df_base = pd.read_csv("./results/baseline_results.csv")
    df_adv  = pd.read_csv("./results/adv_stat_results.csv")
    df_lstm = pd.read_csv("./results/lstm_results.csv")
    df_chro = pd.read_csv("./results/chronos_results.csv")

    # Mean of values for each models
    summ = [
        {"Model": "Naive", "MAE": df_base[df_base['Model'] == 'Naive']['MAE'].mean()},
        {"Model": "Holt-Winters", "MAE": df_base[df_base['Model'] == 'Holt-Winters']['MAE'].mean()},
        {"Model": "SARIMA", "MAE": df_adv['SARIMA_MAE'].mean()},
        {"Model": "SARIMAX", "MAE": df_adv['SARIMAX_MAE'].mean()},
        {"Model": "LSTM", "MAE": df_lstm['MAE'].mean()},
        {"Model": "Chronos", "MAE": df_chro['MAE'].mean()}
    ]

    df_final = pd.DataFrame(summ).sort_values("MAE")

    print("\n" + "="*50)
    print("FINAL COMPARISON OF MODELS")
    print("="*50)
    print(tabulate(df_final, headers='keys', tablefmt='psql', showindex=False))
    
    print(f"\nTHE BEST MODEL IS: {df_final.iloc[0]['Model']}")
    print("="*50)


def main():
    # 1.Eda and data preparation
    print("\n--- STEP 1: EDA AND DATA PREPARATION ---")
    eda.main() 
    
    # 2.Data agrupation
    print("\n--- STEP 2: DATA AGRUPATION ---")
    data_agrupation.main()
    
    # 3.Data partitioning
    print("\n--- STEP 3: DATA PARTITIONING ---")
    data_partition.main() 
    
    # 4.Analysis Grouped
    print("\n--- STEP 4: ANALYSIS GROUPED ---")
    analysis_grouped.main()

    # 5. Baseline Models
    print("\n--- STEP 5: BASELINE MODELS ---")
    baseline_models.main()

    # 6. Advanced Statistical Models (SARIMA & SARIMAX)
    print("\n--- STEP 6: ADVANCED STATISTICAL MODELS ---")
    adv_stat_models.main() 

    # 7. LSTM
    print("\n--- STEP 7: LSTM ---")
    lts_model.main()

    # 8. Chronos (zero-shot)
    print("\n--- STEP 8: CHRONOS (zero-shot) ---")
    chronos_model.main()

    # 9. Final comparative table
    print("\n--- STEP 9: FINAL COMPARATIVE TABLE ---")
    imprimir_tabla_desde_archivos()


    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()