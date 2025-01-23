import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LSTMRegressor:
    def __init__(self, input_size, hidden_size=50, num_layers=1, lr=0.001, epochs=50, batch_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = None
        self.optimizer = None

        self.excluded_columns = ['fecha_comercial', 'codigo_barras_sku', 'nombre_sku', 'imp_vta',
                                 'cant_vta', 'stock', 'pdv_codigo', 'cluster']
        self.return_columns = ['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            return out

    def preprocess_data(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]
        target = 'cant_vta'

        X = data[features]
        y = data[[target]]

        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y)

        return X_scaled, y_scaled

    def fit(self, data):
        X_scaled, y_scaled = self.preprocess_data(data)

        # Reshape for LSTM input
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32).view(-1, 1, self.input_size)
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

        # Train/test split
        train_size = int(0.8 * len(X_scaled))
        X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

        # Model setup
        self.model = self.LSTMModel(self.input_size, self.hidden_size, self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i+self.batch_size].to(self.device)
                y_batch = y_train[i:i+self.batch_size].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss = self.criterion(val_outputs, y_val.to(self.device))
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    def predict(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]
        X = data[features]
        X_scaled = self.scaler.transform(X)
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32).view(-1, 1, self.input_size).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_scaled).cpu().numpy()

        y_pred = self.scaler.inverse_transform(y_pred_scaled).astype(int)
        y_pred[y_pred < 0] = 0

        y_pred_df = pd.DataFrame(y_pred, columns=['cant_vta_pred'])
        y_pred_df.index = data.index

        df_pred = pd.merge(data, y_pred_df, left_index=True, right_index=True)

        df_pred = df_pred[self.return_columns]

        return df_pred
