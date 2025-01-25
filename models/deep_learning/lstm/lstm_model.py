import pandas as pd 
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy as dc
from train.splits.fixed_split import fixed_split
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('fecha_comercial', inplace=True)

    for i in range(1, n_steps+1):
        df[f'cant_vta(t-{i})'] = df['cant_vta'].shift(i)

    df.dropna(inplace=True)

    return df




if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]


    combinations = features[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    pdv_codigo = 1
    codigo_barras_sku = 7894900027013


    data = features[(features['codigo_barras_sku'] == codigo_barras_sku) & (features['pdv_codigo'] == pdv_codigo)]

    data = data[['fecha_comercial', 'cant_vta']]

    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    
    shifted_df_as_np = shifted_df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)   

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    # this is needed for LSTM, it needs to go from the oldest to the newest value. Recurrently getting the most recent value
    X = dc(np.flip(X, axis=1))

    # the last month as the target aprox
    split_index = int(len(X) * 0.95)

    # split the data
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # turning to tensors 
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build the dataset
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break


    model = LSTM(1, 4, 1)
    model.to(device)
    model
