import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import warnings
warnings.filterwarnings("ignore")

def load_chronos():
    """
    Loads the pre-trained Chronos-T5-Tiny model from HuggingFace.
      - amazon/chronos-t5-tiny   (~8M params) <--  we use the smallest model for speed

    device_map="cpu": runs on CPU since we don't need a GPU for this scale.
    torch_dtype=float32: standard precision; float16 would be faster on GPU.
    """
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return pipeline


def forecast_product(pipeline, train_vals: np.ndarray, horizon: int) -> np.ndarray:
    """
    Runs a zero-shot Chronos forecast for a single product.

    train_vals:  historical sales array used as context
    horizon:     number of future days to predict (= length of val set)

    Returns a 1-D array of median predicted sales for each future day.
    """
    # unsqueeze(0) adds a batch dimension: shape (T,) → (1, T)
    context = torch.tensor(train_vals, dtype=torch.float32).unsqueeze(0)

    # pipeline.predict returns shape (batch=1, num_samples, horizon)
    # num_samples defaults to 20 — Chronos draws 20 possible futures.
    forecast = pipeline.predict(context, prediction_length=horizon)

    # Collapse the 20 samples into one point estimate using the median.
    # axis=0 operates over the samples dimension of forecast[0].
    return np.median(forecast[0].numpy(), axis=0)


def main():
    train_df = pd.read_csv("./datasets/train_product.csv", parse_dates=["Date"], index_col="Date")
    val_df   = pd.read_csv("./datasets/val_product.csv",   parse_dates=["Date"], index_col="Date")

    print("Loading Chronos-T5-Tiny (zero-shot, no training needed)...\n")
    pipeline = load_chronos()

    products = train_df["Product ID"].unique()
    results  = []

    print(f"Running Chronos for {len(products)} products...\n")

    for pid in products:
        train_s = train_df[train_df["Product ID"] == pid]["Units Sold"].values
        val_s   = val_df[val_df["Product ID"]     == pid]["Units Sold"].values

        if len(train_s) == 0 or len(val_s) == 0:
            continue

        try:
            preds = forecast_product(pipeline, train_s, horizon=len(val_s))
            mae   = mean_absolute_error(val_s, preds)
            rmse  = np.sqrt(mean_squared_error(val_s, preds))
            results.append({"Product ID": pid, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
            print(f"  {pid} → MAE={mae:.2f}  RMSE={rmse:.2f}")
        except Exception as e:
            print(f"  {pid} skipped: {e}")

    report = pd.DataFrame(results)
    print("\n" + "=" * 45)
    print("CHRONOS — GLOBAL AVERAGES (product level)")
    print("=" * 45)
    print(report[["MAE", "RMSE"]].mean().round(2))
    report.to_csv("./datasets/chronos_results.csv", index=False)
    print("\nFull results saved to datasets/chronos_results.csv")

    #  COMPARISON WITH LSTM
    # This comparison is the key deliverable of section 4.3: does a zero-shot
    # foundation model beat a small LSTM trained specifically on our data?
    try:
        lstm_report = pd.read_csv("./datasets/lstm_results.csv")
        print("\n" + "=" * 45)
        print("FINAL COMPARISON: LSTM vs CHRONOS")
        print("=" * 45)
        merged = lstm_report.merge(report, on="Product ID", suffixes=("_LSTM", "_Chronos"))
        print(merged.to_string(index=False))
        print("\nAverages:")
        print(f"  LSTM    → MAE={lstm_report['MAE'].mean():.2f}  RMSE={lstm_report['RMSE'].mean():.2f}")
        print(f"  Chronos → MAE={report['MAE'].mean():.2f}  RMSE={report['RMSE'].mean():.2f}")
    except FileNotFoundError:
        print("\n(Run lstm_model.py first to see the side-by-side comparison)")


if __name__ == "__main__":
    main()