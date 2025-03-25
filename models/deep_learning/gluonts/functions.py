from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm import tqdm

import pandas as pd
import sys
import numpy as np


def check_data_requirements(data, prediction_length):
    if len(data) < prediction_length:
        print(f"Not enough data points. Required: {prediction_length}, Available: {len(data)}")
        return False
    if data.isnull().any().any():
        print("Data contains NaNs.")
        return False
    if (data['cant_vta'] <= 0).all():
        print("All values in 'cant_vta' are zero or negative.")
        return False
    return True


# Set random seeds for reproducibility
def set_random_seed(seed=42):
    import random
    import numpy as np
    import torch 

    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

# Prepare dataset for DeepAR
def prepare_dataset(data, start_train, end_test, freq, prediction_length):
    df = data.pivot(index="fecha_comercial", columns="pdv_codigo", values="cant_vta")
    # Adding the prediction_length to the dataset
    date_range = pd.date_range(start=df.index.min(), end=end_test, freq=freq)
    df = df.reindex(date_range)
    df.columns = [f"pdv_codigo_{col}" for col in df.columns]
    df_input = df.reset_index().rename(columns={"index": "date"})

    ts_code = np.arange(len(df_input.columns[1:]), dtype=int)
    ts_code_mapping = dict(zip(df_input.columns[1:], ts_code))
    df_values = df_input.iloc[:, 1:].astype(float)

    # leaving the last month of df_train for validation
    df_train = df_values.iloc[:-prediction_length * 2, :].values
    df_val = df_values.iloc[:-prediction_length, :].values
    df_test = df_values.iloc[:, :].values

    # Create ListDataset
    train_ds = create_list_dataset(df_train, ts_code, start_train, freq)
    val_ds = create_list_dataset(df_val, ts_code, start_train, freq)
    test_ds = create_list_dataset(df_test, ts_code, start_train, freq)

    return train_ds, val_ds, test_ds, ts_code, df_input

# Create ListDataset
def create_list_dataset(data, ts_code, start_date, freq):
    return ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_date,
            FieldName.FEAT_STATIC_CAT: [fsc]
        }
        for target, fsc in zip(data.T, ts_code)
    ], freq=freq)

# Make predictions
def make_predictions(predictor, test_ds):

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
        # num_samples=50,

    )

    print("Obtaining time series conditioning values ...")
    tss = list(tqdm(ts_it, total=len(test_ds)))
    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(test_ds)))

    return tss, forecasts

# Process results
def process_results(tss, 
                    forecasts, 
                    df_input, 
                    start_test, 
                    freq, 
                    prediction_length, 
                    sku,
                    model_name,
                    median=False):
    all_results = []

    for i, (tss_series, forecast) in enumerate(zip(tss, forecasts)):
        latest_tss = tss_series.iloc[-prediction_length:].values.flatten()
        predictions_mean = forecast.mean
        pdv_codigo_name = df_input.columns[i + 1]

        if median: 
            predictions_median = forecast.median
            results = pd.DataFrame({
                'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
                'cant_vta': latest_tss,
                f'cant_vta_pred_{model_name}_mean': predictions_mean,
                f'cant_vta_pred_{model_name}_median': predictions_median,
                'pdv_codigo': pdv_codigo_name,
                'codigo_barras_sku': sku
            })
        else:
            results = pd.DataFrame({
                'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
                'cant_vta': latest_tss,
                f'cant_vta_pred_{model_name}_mean': predictions_mean,
                'pdv_codigo': pdv_codigo_name,
                'codigo_barras_sku': sku
            })
        all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.rename(columns={'date': 'fecha_comercial'}, inplace=True)
    final_results['pdv_codigo'] = final_results['pdv_codigo'].str.extract(r'(\d+)$').astype(int)
    final_results['fecha_comercial'] = pd.to_datetime(final_results['fecha_comercial'])
    final_results['codigo_barras_sku'] = final_results['codigo_barras_sku'].astype(int)
    final_results['pdv_codigo'] = final_results['pdv_codigo'].astype(int)
    final_results.drop(columns=['cant_vta'], inplace=True)

    return final_results