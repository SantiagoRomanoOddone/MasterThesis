import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the last time step
        out = self.fc(out[:, -1, :])
        return out

class PyTorchRNNRegressor:
    def __init__(self, input_size, hidden_size=32, num_layers=1, lr=0.001, epochs=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr  # Ensure consistent naming
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNNRegressor(input_size, hidden_size, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # Use self.lr here
        self.excluded_columns = ['fecha_comercial', 'codigo_barras_sku', 'nombre_sku', 'imp_vta',
                                 'cant_vta', 'stock', 'pdv_codigo', 'cluster']
        self.return_columns = ['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']

    def fit(self, data):
        # Exclude non-feature columns
        features = [col for col in data.columns if col not in self.excluded_columns]
        target = 'cant_vta'

        # Filter numeric columns only and handle missing values
        X = data[features].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        y = data[target].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

        # Ensure all data types are valid for PyTorch
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add time dimension
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

        print("Training complete!")

    def predict(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]
        X = data[features].values

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add time dimension

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy()

        y_pred[y_pred < 0] = 0
        y_pred_df = pd.DataFrame(y_pred, columns=['cant_vta_pred'])
        y_pred_df.index = data.index

        df_pred = pd.merge(data, y_pred_df, left_index=True, right_index=True)
        return df_pred
