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
    return ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_date,
            FieldName.FEAT_STATIC_CAT: [fsc]
        }
        for target, fsc in zip(data.T, ts_code)
    ], freq=freq)

# Make predictions
def make_predictions(predictor, df_test, ts_code, start_date, freq):
    test_ds = create_list_dataset(df_test, ts_code, start_date, freq)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
        # num_samples=50,

    )

    print("Obtaining time series conditioning values ...")
    tss = list(tqdm(ts_it, total=len(df_test)))
    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(df_test)))

    return tss, forecasts