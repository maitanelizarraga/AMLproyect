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
    try:
        # Cargar los resultados guardados por cada script
        df_base = pd.read_csv("./results/baseline_results.csv")
        df_adv  = pd.read_csv("./results/adv_stat_results.csv")
        df_lstm = pd.read_csv("./results/lstm_results.csv")
        df_chro = pd.read_csv("./results/chronos_results.csv")

        # Consolidar promedios globales para la tabla final
        resumen = [
            {"Modelo": "Naive", "MAE": df_base[df_base['Model'] == 'Naive']['MAE'].mean()},
            {"Modelo": "Holt-Winters", "MAE": df_base[df_base['Model'] == 'Holt-Winters']['MAE'].mean()},
            {"Modelo": "SARIMA", "MAE": df_adv['SARIMA_MAE'].mean()},
            {"Modelo": "SARIMAX", "MAE": df_adv['SARIMAX_MAE'].mean()},
            {"Modelo": "LSTM", "MAE": df_lstm['MAE'].mean()},
            {"Modelo": "Chronos", "MAE": df_chro['MAE'].mean()}
        ]

        df_final = pd.DataFrame(resumen).sort_values("MAE")

        print("\n" + "="*50)
        print("TABLA COMPARATIVA FINAL (DESDE ARCHIVOS CSV)")
        print("="*50)
        print(tabulate(df_final, headers='keys', tablefmt='psql', showindex=False))
        
        print(f"\nEL MEJOR MODELO ES: {df_final.iloc[0]['Modelo']}")
        print("="*50)

    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}. Ejecuta primero todos los modelos.")


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

    print("\n--- STEP 9: FINAL COMPARATIVE TABLE ---")
    imprimir_tabla_desde_archivos()


    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()