import data_partition
import eda
import adv_stat_models
import analysis_grouped
import data_agrupation
import baseline_models


def main():
    # 1. Limpieza inicial
    eda.main() 
    
    # 2. Agrupación (Nuevo paso separado)
    print("\n--- STEP 2: DATA AGRUPATION ---")
    data_agrupation.main()
    
    # 3. Particionamiento
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



    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()