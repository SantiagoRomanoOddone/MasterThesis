import pandas as pd 
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy as dc
from train.splits.fixed_split import fixed_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Prepare DataFrame for LSTM
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('fecha_comercial', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'cant_vta(t-{i})'] = df['cant_vta'].shift(i)
    df.dropna(inplace=True)
    return df

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


# ----Classes   
class DataScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform(self, df, columns_to_scale):
        scaled_data = self.scaler.fit_transform(df[columns_to_scale])
        df[columns_to_scale] = scaled_data
        return df

    def inverse_transform(self, array, feature_count):
        dummies = np.zeros((array.shape[0], feature_count))
        dummies[:, 0] = array
        return self.scaler.inverse_transform(dummies)[:, 0]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


    def train_one_epoch(self, epoch, train_loader, optimizer, loss_function):
        self.train()
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            output = self(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}: Training Loss: {avg_loss:.4f}')

    def validate_one_epoch(self, test_loader, loss_function):
        self.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
                output = self(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss = running_loss / len(test_loader)
        print(f'Validation Loss: {avg_loss:.4f}')

# Function to reverse scaling
def reverse_scaling(predictions, actual, scaler, lookback):
    dummy = np.zeros((predictions.shape[0], lookback + 1))
    dummy[:, 0] = predictions
    dummy_actual = np.zeros((actual.shape[0], lookback + 1))
    dummy_actual[:, 0] = actual.flatten()

    predictions_rescaled = scaler.inverse_transform(dummy)[:, 0]
    actual_rescaled = scaler.inverse_transform(dummy_actual)[:, 0]

    return predictions_rescaled, actual_rescaled

# Main function for LSTM model
def lstm_model(data):
    # data = data[['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta']]
    excluded_columns = ['nombre_sku', 'imp_vta', 'stock','cluster', 'lag_1', 'lag_7', 'lag_30']
    data = data.drop(columns=excluded_columns)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    lookback = 7
    prepared_data = prepare_dataframe_for_lstm(data, lookback)
    prepared_data.reset_index(inplace=True)

    train_df, test_df = fixed_split(prepared_data)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_np = scaler.fit_transform(train_df.drop(columns=['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo']).to_numpy())
    test_np = scaler.transform(test_df.drop(columns=['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo']).to_numpy())

    X_train, y_train = train_np[:, 1:], train_np[:, 0]
    X_test, y_test = test_np[:, 1:], test_np[:, 0]

    X_train = np.flip(X_train, axis=1).reshape((-1, lookback, 1))
    X_test = np.flip(X_test, axis=1).reshape((-1, lookback, 1))
    y_train, y_test = y_train.reshape((-1, 1)), y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train.copy()).float()
    X_test = torch.tensor(X_test.copy()).float()
    y_train = torch.tensor(y_train.copy()).float()
    y_test = torch.tensor(y_test.copy()).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # model = LSTM(1, 64, 2, device)
    model = LSTM(len(features[1:]), 128, 3, device)
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # early_stopping 
    early_stopping = EarlyStopping(patience=5)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train_one_epoch(epoch, train_loader, optimizer, loss_function)
        model.validate_one_epoch(test_loader, loss_function)

        val_loss = sum([loss_function(model(batch[0].to(device)), batch[1].to(device)).item() for batch in test_loader]) / len(test_loader)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    with torch.no_grad():
        # train_predictions = model(X_train.to(device)).to('cpu').numpy().flatten()
        test_predictions = model(X_test.to(device)).to('cpu').numpy().flatten()

    # train_predictions_rescaled, y_train_rescaled = reverse_scaling(train_predictions, y_train, scaler, lookback)
    test_predictions_rescaled, y_test_rescaled = reverse_scaling(test_predictions, y_test, scaler, lookback)

    # Create DataFrame for predictions
    test_df['cant_vta_pred'] = test_predictions_rescaled
    test_df['cant_vta_actual'] = y_test_rescaled

    # Reorder and return columns for consistency
    test_df = test_df[['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']]

    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.title('Train Predictions vs Actual')
    # plt.plot(y_train_rescaled, label='Actual')
    # plt.plot(train_predictions_rescaled, label='Predicted')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.title('Test Predictions vs Actual')
    # plt.plot(y_test_rescaled, label='Actual')
    # plt.plot(test_predictions_rescaled, label='Predicted')
    # plt.legend()
    # plt.show()

    return test_df




if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    cluster_data = features[features['cluster'] == cluster_number]

    # Testing
    pdv_codigo = 1
    codigo_barras_sku = 7894900027013

    data = cluster_data[(cluster_data['codigo_barras_sku'] == codigo_barras_sku) & (cluster_data['pdv_codigo'] == pdv_codigo)]


    result = lstm_model(data)
    print(result)

    