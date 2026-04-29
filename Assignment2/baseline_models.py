import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    """Calculates standard error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def run_naive(train, val):
    """Naïve Forecasting: Predicts the last observed value."""
    last_value = train.iloc[-1]
    return [last_value] * len(val)

def run_moving_average(train, val, window=7):
    """Moving Average (MA): Approach basic features using a sliding window."""
    history = list(train.values)
    predictions = []
    for i in range(len(val)):
        avg = np.mean(history[-window:])
        predictions.append(avg)
        history.append(avg) # Dynamic update
    return predictions

def run_holt_winters(train, val):
    """Exponential Smoothing (Holt-Winters): Handles trend and seasonality."""
    # We use additive seasonality as seen in EDA (period 7 for weekly)
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
    return model.forecast(len(val))

def run_arima(train, val):
    """ARIMA Model: Identifies if the series requires differencing."""
    # Baseline (1,1,1) to check stationarity and autocorrelation
    model = ARIMA(train, order=(1, 1, 1)).fit()
    return model.forecast(len(val))

def main():
    # 1. Load pre-partitioned and aggregated data
    train_full = pd.read_csv("./datasets/train.csv", parse_dates=['Date'], index_col='Date')
    val_full = pd.read_csv("./datasets/val.csv", parse_dates=['Date'], index_col='Date')

    target = 'Units Sold'
    stores = train_full['Store ID'].unique()
    all_results = []

    print(f"Initializing Time-Series Analysis for {len(stores)} stores...\n")

    for store_id in stores:
        # Filter data for specific store
        train_s = train_full[train_full['Store ID'] == store_id][target]
        val_s = val_full[val_full['Store ID'] == store_id][target]
        region = train_full[train_full['Store ID'] == store_id]['Region'].iloc[0]

        # Ensure we have enough data for seasonal models
        if len(train_s) < 14:
            continue

        # Execute models according to the required complexity
        models = {
            "Naive": run_naive(train_s, val_s),
            "Moving Average": run_moving_average(train_s, val_s),
            "Holt-Winters": run_holt_winters(train_s, val_s),
            "ARIMA": run_arima(train_s, val_s)
        }

        # Evaluate performance
        for name, preds in models.items():
            mae, rmse = calculate_metrics(val_s, preds)
            all_results.append({
                "Region": region,
                "Store ID": store_id,
                "Model": name,
                "MAE": mae,
                "RMSE": rmse
            })

    # 2. Final Comparative Report
    results_df = pd.DataFrame(all_results)
    
    # Global average comparison
    print("--- GLOBAL MODEL PERFORMANCE (AVG) ---")
    summary = results_df.groupby("Model")[["MAE", "RMSE"]].mean().sort_values("MAE")
    print(summary)

    # Regional analysis (as requested: predicting sales by region and store)
    print("\n--- PERFORMANCE BY REGION ---")
    regional_summary = results_df.groupby(["Region", "Model"])[["MAE"]].mean().unstack()
    print(regional_summary)

    # Identifying the best method for each store
    print("\n--- BEST MODEL IDENTIFIED PER STORE ---")
    best_models = results_df.loc[results_df.groupby("Store ID")["MAE"].idxmin()]
    print(best_models[["Region", "Store ID", "Model", "MAE"]])

if __name__ == "__main__": 
    main()