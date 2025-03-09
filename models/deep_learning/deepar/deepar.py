# !pip install numpy==1.23.5
# !pip install --upgrade mxnet==1.6.0
# !pip install gluonts
# !pip install lightning
import numpy as np
import mxnet as mx
import random
np.random.seed(7)
mx.random.seed(7)
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm.autonotebook import tqdm
from gluonts.torch import DeepAREstimator
from gluonts.mx.trainer import Trainer
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import sys

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False
pd.set_option('display.max_columns', None)


# Constants
CLUSTER_NUMBER = 3
SKU = 7894900027013
FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
END_TRAIN = pd.Timestamp("2024-10-31")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")

DATA_PATH = "/content/features.parquet"

# Set random seeds for reproducibility
def set_random_seed(seed=42):
    import random
    import numpy as np
    import torch  # If using PyTorch backend
    import mxnet as mx  # If using MXNet backend

    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if 'mxnet' in sys.modules:
        mx.random.seed(seed)

# Load and preprocess data
def load_and_preprocess_data(data_path, cluster_number, sku, end_test, start_test):
    features = pd.read_parquet(data_path)
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster"] == cluster_number]
    
    filtered = features[(features["codigo_barras_sku"] == sku)].copy()
    filtered = filtered[filtered['fecha_comercial'] <= end_test]
    validation = filtered[filtered['fecha_comercial'] >= start_test]
    filtered = filtered[filtered['fecha_comercial'] < start_test]
    
    return filtered, validation

# Prepare dataset for DeepAR
def prepare_dataset(data, end_test, freq, prediction_length):
    df = data.pivot(index="fecha_comercial", columns="pdv_codigo", values="cant_vta")
    date_range = pd.date_range(start=df.index.min(), end=end_test, freq=freq)
    df = df.reindex(date_range)
    df.columns = [f"pdv_codigo_{col}" for col in df.columns]
    df_input = df.reset_index().rename(columns={"index": "date"})
    
    ts_code = np.arange(len(df_input.columns[1:]), dtype=int)
    ts_code_mapping = dict(zip(df_input.columns[1:], ts_code))
    df_values = df_input.iloc[:, 1:].astype(float)
    
    df_train = df_values.iloc[:-prediction_length, :].values
    df_test = df_values.iloc[:, :].values
    
    return df_train, df_test, ts_code, df_input

# Create ListDataset
def create_list_dataset(data, ts_code, start_date, freq):
    """
    Create a ListDataset for GluonTS from the given data.
    """
    return ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_date,
            FieldName.FEAT_STATIC_CAT: [fsc]
        }
        for target, fsc in zip(data.T, ts_code)
    ], freq=freq)

# Train the DeepAR model
def train_deepar_model(df_train, ts_code, start_date, freq, prediction_length):
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_layers=2,
        hidden_size=32,
        lr=0.001,
        weight_decay=1e-08,
        dropout_rate=0.1,
        patience=10,
        num_feat_dynamic_real=0,
        num_feat_static_cat=1,
        num_feat_static_real=0,
        cardinality=[len(np.unique(ts_code))],
        embedding_dimension=None,
        scaling=True,
        default_scale=None,
        lags_seq=None,
        time_features=None,
        num_parallel_samples=100,
        batch_size=32,
        num_batches_per_epoch=50,
        imputation_method=None,
        trainer_kwargs={"max_epochs": 5},
        train_sampler=None,
        validation_sampler=None,
        nonnegative_pred_samples=False,
    )
    
    train_ds = create_list_dataset(df_train, ts_code, start_date, freq)
    predictor = estimator.train(training_data=train_ds)
    return predictor

# Make predictions
def make_predictions(predictor, df_test, ts_code, start_date, freq):
    test_ds = create_list_dataset(df_test, ts_code, start_date, freq)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )
    
    print("Obtaining time series conditioning values ...")
    tss = list(tqdm(ts_it, total=len(df_test)))
    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(df_test)))
    
    return tss, forecasts

# Process results
def process_results(tss, forecasts, df_input, start_test, freq, prediction_length, sku):
    all_results = []
    
    for i, (tss_series, forecast) in enumerate(zip(tss, forecasts)):
        latest_tss = tss_series.iloc[-prediction_length:].values.flatten()
        predictions = forecast.mean
        pdv_codigo_name = df_input.columns[i + 1]
        
        results = pd.DataFrame({
            'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
            'cant_vta': latest_tss,
            'cant_vta_pred_deepar': predictions,
            'pdv_codigo': pdv_codigo_name
        })
        all_results.append(results)
    
    final_results = pd.concat(all_results, ignore_index=True)
    final_results['codigo_barras_sku'] = sku
    final_results.rename(columns={'date': 'fecha_comercial'}, inplace=True)
    final_results['pdv_codigo'] = final_results['pdv_codigo'].str.extract(r'(\d+)$').astype(int)
    final_results['fecha_comercial'] = pd.to_datetime(final_results['fecha_comercial'])
    final_results['codigo_barras_sku'] = final_results['codigo_barras_sku'].astype(int)
    final_results['pdv_codigo'] = final_results['pdv_codigo'].astype(int)
    final_results.drop(columns=['cant_vta'], inplace=True)
    
    return final_results

# Main function
def main():
    set_random_seed(42)
    
    # Load and preprocess data
    filtered, validation = load_and_preprocess_data(data_path = DATA_PATH, 
                                                    cluster_number= CLUSTER_NUMBER,
                                                    sku = SKU, 
                                                    end_test = END_TEST, 
                                                    start_test = START_TEST)
    
    # Prepare dataset
    df_train, df_test, ts_code, df_input = prepare_dataset(data = filtered, 
                                                           end_test = END_TEST, 
                                                           freq = FREQ, 
                                                            prediction_length = PREDICTION_LENGTH)
    
    # Train the model
    predictor = train_deepar_model(df_train = df_train, 
                                   ts_code = ts_code, 
                                   start_date = START_TRAIN, 
                                   freq = FREQ, 
                                   prediction_length = PREDICTION_LENGTH)
    
    # Make predictions
    tss, forecasts = make_predictions(predictor = predictor, 
                                      df_test = df_test, 
                                      ts_code = ts_code, 
                                      start_date = START_TRAIN, 
                                      freq = FREQ)
    
    # Process results
    final_results = process_results(tss = tss, 
                                    forecasts = forecasts, 
                                    df_input = df_input, 
                                    start_test = START_TEST, 
                                    freq = FREQ, 
                                    prediction_length = PREDICTION_LENGTH, 
                                    sku = SKU)
    
    return final_results

# Run the main function
if __name__ == "__main__":
    final_results = main()
    print(final_results)