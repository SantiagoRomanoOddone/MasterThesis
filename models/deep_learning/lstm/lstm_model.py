import pandas as pd 
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy as dc
from train.splits.fixed_split import fixed_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('fecha_comercial', inplace=True)

    for i in range(1, n_steps+1):
        df[f'cant_vta(t-{i})'] = df['cant_vta'].shift(i)

    df.dropna(inplace=True)

    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
# input_size: number of features, 1 in this case
# hidden_size: number of hidden units
# num_stacked_layers: number of stacked LSTM layers, the more we have the more complex the model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        # the output of the LSTM is the input of the fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        batch_size = x.size(0)
        # initialize the hidden state and the cell state
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


    def train_one_epoch(self, epoch, train_loader, optimizer, loss_function):
        self.train()
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            output = self(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

    def validate_one_epoch(self, test_loader, loss_function):
        self.eval()
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()


def lstm_model(data):

    data = data[['fecha_comercial', 'cant_vta']]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Apply the prepare_dataframe_for_lstm function
    lookback = 7
    prepared_data = prepare_dataframe_for_lstm(data, lookback)

    prepared_data.reset_index(inplace=True)

    # Apply train-test split
    train_df, test_df = fixed_split(prepared_data)

    # Convert train and test data to numpy arrays
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_np = scaler.fit_transform(train_df.drop(columns=['fecha_comercial']).to_numpy())
    test_np = scaler.transform(test_df.drop(columns=['fecha_comercial']).to_numpy())

    # Separate features and labels for train and test sets
    X_train = train_np[:, 1:]
    y_train = train_np[:, 0]

    X_test = test_np[:, 1:]
    y_test = test_np[:, 0]

    # Flip X arrays for LSTM compatibility
    X_train = np.flip(X_train, axis=1)
    X_test = np.flip(X_test, axis=1)

    # Reshape for LSTM input
    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Convert to tensors
    X_train = torch.tensor(X_train.copy()).float()
    X_test = torch.tensor(X_test.copy()).float()
    y_train = torch.tensor(y_train.copy()).float()
    y_test = torch.tensor(y_test.copy()).float()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build the dataset
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Sanity check for one batch
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    # Define and train the LSTM model
    model = LSTM(1, 4, 1)
    model.to(device)

    learning_rate = 0.001
    num_epochs = 10

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train_one_epoch(epoch, train_loader, optimizer, loss_function)
        model.validate_one_epoch(test_loader, loss_function)

    # Evaluate and plot results
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

        plt.plot(y_train, label='Actual Close')
        plt.plot(predicted, label='Predicted Close')
        plt.xlabel('Day')
        plt.ylabel('Close')
        plt.legend()
        plt.show()

    train_predictions = predicted.flatten()

    # Reverse scaling for train predictions
    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])

    # Reverse scaling for train actual values
    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])

    # Plot train results
    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    # Reverse scaling for test predictions
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    # Reverse scaling for test actual values
    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

    # Plot test results
    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()



if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]
    combinations = features[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    # Testing
    pdv_codigo = 1
    codigo_barras_sku = 7894900027013

    