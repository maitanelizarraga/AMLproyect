import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    #Calculates standard error metrics (MAE and RMSE)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def run_naive(train, val):
    #Naive Forecasting: Predicts using the last observed value
    last_value = train.iloc[-1]
    return [last_value] * len(val)

def run_moving_average(train, val, window=7):
    #Moving Average (MA): Uses a sliding window to forecast
    history = list(train.values)
    predictions = []
    for _ in range(len(val)):
        avg = np.mean(history[-window:])
        predictions.append(avg)
        history.append(avg)  # Dynamic update for multi-step forecast
    return predictions

def run_holt_winters(train, val):
    #Holt-Winters: Exponential smoothing considering trend and weekly seasonality
    try:
        # period=7 because the data is aggregated daily and shows weekly patterns
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
        return model.forecast(len(val))
    except:
        # Fallback to Naive if model fails to converge
        return run_naive(train, val)

def run_arima(train, val):
    #ARIMA Model: Basic (1,1,1) configuration for time-series baseline.
    try:
        model = ARIMA(train, order=(1, 1, 1)).fit()
        return model.forecast(len(val))
    except:
        return run_naive(train, val)

def main():

    train_full = pd.read_csv("./datasets/train_Store.csv", parse_dates=['Date'], index_col='Date')
    val_full = pd.read_csv("./datasets/val_Store.csv", parse_dates=['Date'], index_col='Date')


    target = 'Units Sold'
    stores = train_full['Store ID'].unique()
    all_results = []

    print(f"--- STARTING BASELINE MODEL EVALUATION ({len(stores)} Stores) ---")

    for store_id in stores:
        # Filter store-specific data
        train_s = train_full[train_full['Store ID'] == store_id][target]
        val_s = val_full[val_full['Store ID'] == store_id][target]
        region = train_full[train_full['Store ID'] == store_id]['Region'].iloc[0]

        # Minimum data requirement for seasonal models (at least 2 full weeks)
        if len(train_s) < 14:
            continue

        # Execute Baseline Models
        models = {
            "Naive": run_naive(train_s, val_s),
            "Moving Average": run_moving_average(train_s, val_s),
            "Holt-Winters": run_holt_winters(train_s, val_s),
            "ARIMA": run_arima(train_s, val_s)
        }

        # Calculate metrics for each model
        for name, preds in models.items():
            mae, rmse = calculate_metrics(val_s, preds)
            all_results.append({
                "Region": region,
                "Store ID": store_id,
                "Model": name,
                "MAE": mae,
                "RMSE": rmse
            })

    # 2. GENERATE COMPARATIVE REPORTS
    results_df = pd.DataFrame(all_results)
    
    # Global average performance
    print("\n" + "="*40)
    print("GLOBAL MODEL PERFORMANCE (AVG)")
    print("="*40)
    summary = results_df.groupby("Model")[["MAE", "RMSE"]].mean().sort_values("MAE")
    print(summary)

    # Regional analysis
    print("\n" + "="*40)
    print("PERFORMANCE BY REGION (MAE)")
    print("="*40)
    regional_summary = results_df.groupby(["Region", "Model"])[["MAE"]].mean().unstack()
    print(regional_summary)

    # Best model per store identification
    print("\n" + "="*40)
    print("BEST MODEL PER STORE")
    print("="*40)
    best_idx = results_df.groupby("Store ID")["MAE"].idxmin()
    best_models = results_df.loc[best_idx].sort_values("Region")
    print(best_models[["Region", "Store ID", "Model", "MAE"]].to_string(index=False))

if __name__ == "__main__": 
    main()