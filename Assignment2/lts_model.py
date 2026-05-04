import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

#  HYPERPARAMETERS 
SEQ_LEN = 14   # how many past days the LSTM uses to predict
               # the next day. 14 = two weeks, enough to capture weekly patterns
               # without making the sequences too long for our small dataset.
               # Increasing this (e.g. 30) captures more history but needs more data.

HIDDEN  = 32   # Number of hidden units in the LSTM cell. Controls the model's
               # "memory capacity". 32 is intentionally small to avoid overfitting
               # on short product-level series. Larger values (64, 128) are better
               # when you have thousands of training samples.

EPOCHS  = 20   # How many full passes through the training data.
               # 20 is enough for convergence on small time series.
               # Too many epochs risk overfitting; too few leave the model undertrained.

LR      = 1e-3 # Learning rate for the Adam optimizer (0.001).
               # Controls how big each gradient update step is.
               # 1e-3 is the standard Adam default and works well here.
               # Lower values (1e-4) train more slowly but more stably.

BATCH   = 16   # Number of sequences processed together in each gradient update.
               # 16 is small, which suits our limited data per product.
               # Larger batches (32, 64) are faster but need more data to generalize.


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN):
        super().__init__()
        # Single-layer LSTM: takes sequences of shape (batch, seq_len, input_size)
        # and outputs a hidden state at every time step.
        # We use only one layer to keep the model simple and fast.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully-connected layer: maps the last hidden state (hidden_size,)
        # to a single scalar → the next-day sales prediction.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)
        # We only care about the last time step's output (out[:, -1, :])
        # because it summarises the entire sequence seen so far.
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(series: np.ndarray, seq_len: int):
    """
    Converts a 1-D time series into supervised learning pairs (X, y)
    using a sliding window of size seq_len.

    Each row of X is the input window; the corresponding y is the value to predict.
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


def train_and_eval(train_vals, val_vals):
    # Scale → build sequences → train LSTM → evaluate on validation set.

    # MinMaxScaler compresses all values to [0, 1].
    # LSTMs train much better on normalized data because the sigmoid/tanh
    # gates inside the cell saturate with large raw values (e.g. sales in hundreds).
    # We fit ONLY on train to avoid leaking validation statistics.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val_vals.reshape(-1, 1)).flatten()

    # Guard: we need at least SEQ_LEN + 1 points to form one training sequence.
    if len(train_scaled) <= SEQ_LEN or len(val_scaled) == 0:
        return None, None

    X_tr, y_tr = make_sequences(train_scaled, SEQ_LEN)

    # For validation sequences we prepend the last SEQ_LEN train points so
    # the first val prediction has a full context window available.
    X_va, _ = make_sequences(
        np.concatenate([train_scaled[-SEQ_LEN:], val_scaled]), SEQ_LEN
    )

    # Add a feature dimension: shape (samples, seq_len) → (samples, seq_len, 1)
    # The "1" is input_size — we have one feature (Units Sold).
    # With exogenous variables you would increase this dimension.
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).unsqueeze(-1)

    # shuffle=False is critical for time series: we must preserve chronological
    # order so the model never "sees the future" during training.
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=False)

    model     = LSTMForecaster(input_size=1)
    # Adam adapts the learning rate per parameter, making it more robust than
    # plain SGD for recurrent networks where gradients can vary wildly.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # MSELoss penalises large errors more heavily (squares them), which pushes
    # the model to avoid big mispredictions — desirable for inventory planning.
    criterion = nn.MSELoss()

    # TRAINING LOOP
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()          # reset gradients from previous batch
            loss = criterion(model(xb), yb)
            loss.backward()                # compute gradients via backprop through time
            optimizer.step()               # update weights


    # Inference: torch.no_grad() disables gradient tracking → faster and less memory.
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_va_t).numpy()

    # Reverse the MinMax scaling to get predictions back in original units (sales count).
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    truth = val_vals[: len(preds)]  # align lengths in case of rounding

    mae  = mean_absolute_error(truth, preds)
    rmse = np.sqrt(mean_squared_error(truth, preds))
    return mae, rmse


def main():
    train_df = pd.read_csv("./datasets/train_product.csv", parse_dates=["Date"], index_col="Date")
    test_df   = pd.read_csv("./datasets/test_product.csv",   parse_dates=["Date"], index_col="Date")

    products = train_df["Product ID"].unique()
    results  = []

    print(f"Running LSTM for {len(products)} products...\n")

    for pid in products:
        train_s = train_df[train_df["Product ID"] == pid]["Units Sold"].values
        test_s   = test_df[test_df["Product ID"]     == pid]["Units Sold"].values

        mae, rmse = train_and_eval(train_s, test_s)
        if mae is None:
            continue

        results.append({"Product ID": pid, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
        print(f"  {pid} → MAE={mae:.2f}  RMSE={rmse:.2f}")

    report = pd.DataFrame(results)
    print("\n" + "=" * 45)
    print("LSTM — GLOBAL AVERAGES (product level)")
    print("=" * 45)
    print(report[["MAE", "RMSE"]].mean().round(2))
    report.to_csv("./results/lstm_results.csv", index=False)
    print("\nFull results saved to ./results/lstm_results.csv")


if __name__ == "__main__":
    main()