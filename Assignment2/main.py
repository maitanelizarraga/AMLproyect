import data_partition
import eda
import adv_stat_models
import analysis_grouped
import data_agrupation
import baseline_models
import dl_foundations_models


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

    # 7. Deep Learning & Foundation Models
    print("\n--- STEP 7: DEEP LEARNING & FOUNDATION MODELS ---")
    dl_foundations_models.main()


    # 7. LSTM
    print("\n--- STEP 7: LSTM ---")
    lstm_model.main()

    # 8. Chronos (zero-shot)
    print("\n--- STEP 8: CHRONOS (zero-shot) ---")
    chronos_model.main()




    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)

if __name__ == "__main__": 
    main()