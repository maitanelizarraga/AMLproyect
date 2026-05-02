import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# --- 1. LSTM ARCHITECTURE ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

# --- 2. TRAINING LSTM ---
def train_and_save_lstm(train_df, product_id, folder="./models"):
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df[['Units Sold']].values)
    
    seq_length = 14 
    X_train, y_train = create_sequences(train_data, seq_length)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    model = LSTMForecaster(1, 64, 2, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    if not os.path.exists(folder): os.makedirs(folder)
    path = f"{folder}/lstm_{product_id}.pth"
    torch.save(model.state_dict(), path)
    return model, scaler, path

# --- 3. CHRONOS FALLBACK ---
def run_chronos_inference(train_series, forecast_length):
    # Simulamos el comportamiento de base de un modelo foundation (Naive)
    return np.full(forecast_length, train_series.iloc[-1])

# --- 4. MAIN ---
def main():
    print("--- STEP 7: DEEP LEARNING & FOUNDATION MODELS ---")
    try:
        train_full = pd.read_csv("./datasets/train_product.csv", parse_dates=['Date'])
        val_full = pd.read_csv("./datasets/val_product.csv", parse_dates=['Date'])
    except:
        print("Error: Datasets no encontrados.")
        return
    
    products = train_full['Product ID'].unique()[:5] 
    results = []

    for prod_id in products:
        print(f"\n[Producto: {prod_id}]")
        train_p = train_full[train_full['Product ID'] == prod_id].sort_values('Date')
        val_p = val_full[val_full['Product ID'] == prod_id].sort_values('Date')

        if len(train_p) < 20: continue

        model, scaler, _ = train_and_save_lstm(train_p, prod_id)
        
        # Predicción LSTM
        model.eval()
        train_scaled = scaler.transform(train_p[['Units Sold']].values)
        curr_seq = torch.from_numpy(train_scaled[-14:]).float().unsqueeze(0)
        
        lstm_preds = []
        for _ in range(len(val_p)):
            with torch.no_grad():
                p = model(curr_seq)
                lstm_preds.append(p.item())
                curr_seq = torch.cat((curr_seq[:, 1:, :], p.unsqueeze(0)), dim=1)
        
        final_lstm = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
        
        # Predicción Chronos (Naive)
        c_preds = run_chronos_inference(train_p['Units Sold'], len(val_p))

        results.append({
            'Product': prod_id,
            'LSTM_MAE': round(float(mean_absolute_error(val_p['Units Sold'], final_lstm)), 2),
            'Chronos_MAE': round(float(mean_absolute_error(val_p['Units Sold'], c_preds)), 2)
        })

    print("\n" + "="*40)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()