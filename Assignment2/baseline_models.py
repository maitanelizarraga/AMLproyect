import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore")

def run_sarima(train, val, target_col):
    """Pure SARIMA model (Baseline): Captures daily and weekly seasonality."""
    model = SARIMAX(train[target_col], 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)
    return results.get_forecast(steps=len(val)).predicted_mean

def run_sarimax(train, val, target_col, exog_cols):
    """SARIMAX model (Evolved): Uses external variables like Price and Discount."""
    model = SARIMAX(train[target_col], 
                    exog=train[exog_cols],
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)
    return results.get_forecast(steps=len(val), exog=val[exog_cols]).predicted_mean

def run_random_walk(train, val, target_col):
    """Random Walk Forecast (Naive Baseline): Forecasts the last observed value."""
    last_value = train[target_col].iloc[-1]
    return [last_value] * len(val)

def main():
    # 1. Load the pre-partitioned data (already aggregated by store and date)
    train_full = pd.read_csv("./datasets/train.csv", parse_dates=['Date'], index_col='Date')
    val_full = pd.read_csv("./datasets/val.csv", parse_dates=['Date'], index_col='Date')

    target = 'Units Sold'
    exog_vars = ['Price', 'Discount', 'Competitor Pricing', 'Inventory Level']
    
    stores = train_full['Store ID'].unique()
    performance_report = []

    print(f"Running models for {len(stores)} stores...\n")

    for store_id in stores:
        print(f"Processing Store {store_id}...")
        # Filter store-specific data
        train_store = train_full[train_full['Store ID'] == store_id]
        val_store = val_full[val_full['Store ID'] == store_id]

        try:
            # 2. Execute all models
            p_rw = run_random_walk(train_store, val_store, target)
            p_sarima = run_sarima(train_store, val_store, target)
            p_sarimax = run_sarimax(train_store, val_store, target, exog_vars)

            # 3. Calculate metrics (MAE)
            mae_rw = mean_absolute_error(val_store[target], p_rw)
            mae_sarima = mean_absolute_error(val_store[target], p_sarima)
            mae_sarimax = mean_absolute_error(val_store[target], p_sarimax)

            performance_report.append({
                'Store': store_id,
                'RW_MAE': mae_rw,
                'SARIMA_MAE': mae_sarima,
                'SARIMAX_MAE': mae_sarimax
            })
        except Exception as e:
            print(f"Skipping store {store_id} due to error: {e}")

    # 4. Generate Final Report
    report_df = pd.DataFrame(performance_report)
    print("\n" + "="*55)
    print("FINAL PERFORMANCE REPORT (MAE COMPARISON)")
    print("="*55)
    # Reordering columns to show evolution: RW -> SARIMA -> SARIMAX
    cols = ['Store', 'RW_MAE', 'SARIMA_MAE', 'SARIMAX_MAE']
    print(report_df[cols].to_string(index=False))
    
    print("\nGlobal Averages:")
    print(report_df[cols].mean(numeric_only=True).round(2))

if __name__ == "__main__": 
    main()