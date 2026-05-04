import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# --- BASELINE MODELS ---
def run_naive(train, horizon):
    last_value = train.iloc[-1]
    return [last_value] * horizon

def run_moving_average(train, horizon, window=7):
    history = list(train.values)
    predictions = []
    for _ in range(horizon):
        avg = np.mean(history[-window:])
        predictions.append(avg)
        history.append(avg)
    return predictions

def run_holt_winters(train, horizon):
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
        return model.forecast(horizon)
    except:
        return run_naive(train, horizon)

def run_arima(train, horizon):
    try:
        model = ARIMA(train, order=(1, 1, 1)).fit()
        return model.forecast(horizon)
    except:
        return run_naive(train, horizon)

def evaluate_models(train_data, target_data, group_name, group_id):
    """Encapsulates the evaluation logic for a single group (Store or Region)."""
    horizon = len(target_data)
    if len(train_data) < 14: # Requirement for seasonal models
        return []

    models = {
        "Naive": run_naive(train_data, horizon),
        "Moving Average": run_moving_average(train_data, horizon),
        "Holt-Winters": run_holt_winters(train_data, horizon),
        "ARIMA": run_arima(train_data, horizon)
    }

    results = []
    for name, preds in models.items():
        mae, rmse = calculate_metrics(target_data, preds)
        results.append({
            "Type": group_name,
            "ID": group_id,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })
    return results

def main():
    # 1. LOAD DATASETS
    # Note: Using Validation set as our target for baseline comparison
    path = "./datasets"
    
    # Store Data
    train_store = pd.read_csv(f"{path}/train_Store.csv", parse_dates=['Date'], index_col='Date')
    test_store = pd.read_csv(f"{path}/test_Store.csv", parse_dates=['Date'], index_col='Date')
    
    # Region Data
    train_region = pd.read_csv(f"{path}/train_region.csv", parse_dates=['Date'], index_col='Date')
    test_region = pd.read_csv(f"{path}/test_region.csv", parse_dates=['Date'], index_col='Date')

    all_results = []
    target_col = 'Units Sold'

    # 2. EVALUATE PER STORE
    print("Evaluating Stores...")
    stores = train_store['Store ID'].unique()
    for s_id in stores:
        t_s = train_store[train_store['Store ID'] == s_id][target_col]
        v_s = test_store[test_store['Store ID'] == s_id][target_col]
        if not v_s.empty:
            all_results.extend(evaluate_models(t_s, v_s, "Store", s_id))

    # 3. EVALUATE PER REGION
    print("Evaluating Regions...")
    regions = train_region['Region'].unique()
    for r_id in regions:
        t_r = train_region[train_region['Region'] == r_id][target_col]
        v_r = test_region[test_region['Region'] == r_id][target_col]
        if not v_r.empty:
            all_results.extend(evaluate_models(t_r, v_r, "Region", r_id))

    # 4. REPORTS
    results_df = pd.DataFrame(all_results)
    
    
    results_df.to_csv("./results/baseline_results.csv", index=False)
    print(f"\nResults saved to ./results/baseline_results.csv")

    # --- PERFORMANCE SUMMARY ---
    print("\n" + "="*40)
    print("GLOBAL MODEL PERFORMANCE (AVG MAE)")
    print("="*40)
    print(results_df.groupby(["Type", "Model"])["MAE"].mean().unstack())

    # --- BEST MODELS ---
    print("\n" + "="*40)
    print("BEST MODEL PER CATEGORY")
    print("="*40)
    best_models = results_df.loc[results_df.groupby(["Type", "ID"])["MAE"].idxmin()]
    print(best_models[["Type", "ID", "Model", "MAE"]].head(10)) # Showing first 10 for brevity

if __name__ == "__main__":
    main()