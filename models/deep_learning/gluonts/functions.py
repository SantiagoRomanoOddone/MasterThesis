from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random 
import pandas as pd
import sys
import numpy as np
from typing import List
from lightning.pytorch.callbacks import EarlyStopping
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    month_of_year,
    day_of_month,
    day_of_week,
    day_of_year,
    week_of_year,
)


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

def create_temporal_features(date_index):
    """
    Creates temporal features from a datetime index.
    """
    temporal_features = pd.DataFrame(index=date_index)
    temporal_features['year'] = temporal_features.index.year
    temporal_features['month'] = temporal_features.index.month
    temporal_features['day'] = temporal_features.index.day
    temporal_features['day_of_week'] = temporal_features.index.dayofweek
    temporal_features['is_weekend'] = temporal_features['day_of_week'].isin([5, 6]).astype(int)
    temporal_features['quarter'] = temporal_features.index.quarter
    temporal_features['week_of_year'] = temporal_features.index.isocalendar().week.astype(int)
    temporal_features['day_of_year'] = temporal_features.index.dayofyear
    temporal_features['is_month_start'] = temporal_features.index.is_month_start.astype(int)
    temporal_features['is_month_end'] = temporal_features.index.is_month_end.astype(int)
    temporal_features['is_first_week'] = (temporal_features['week_of_year'] == 1).astype(int)
    temporal_features['is_last_week'] = (temporal_features['week_of_year'].isin([52, 53])).astype(int)
    
    return temporal_features

def prepare_dataset(data, start_train, end_test, freq, prediction_length, temporal_features=None):
    # Step 1: Pivot the data (keeping fecha_comercial as index)
    df = data.pivot(index="fecha_comercial", columns="pdv_codigo", values="cant_vta")

    # Step 2: Create complete date range
    full_date_range = pd.date_range(start=df.index.min(), end=end_test, freq=freq)

    # Step 3: Create base DataFrame with all required dates
    full_index_df = pd.DataFrame(index=full_date_range)

    # Step 4: Reindex original data to complete range
    df = df.reindex(full_date_range)

    # Step 6: Prepare final DataFrame structure
    df.columns = [f"pdv_codigo_{col}" for col in df.columns]
    df_input = df.reset_index().rename(columns={"index": "date"})
    ts_code = np.arange(len(df_input.columns[1:]), dtype=int)
    ts_code_mapping = dict(zip(df_input.columns[1:], ts_code))
    df_values = df_input.iloc[:, 1:].astype(float)
    
    # Step 7: Split data (train/val/test)
    df_train = df_values.iloc[:-prediction_length * 2, :].values
    df_val = df_values.iloc[:-prediction_length, :].values
    df_test = df_values.iloc[:, :].values

    if temporal_features:
        # Step 5: Generate temporal features
        temporal_features = create_temporal_features(full_index_df.index)
        
        # Step 8: Split temporal features accordingly
        temporal_features_train = temporal_features.iloc[:-prediction_length * 2, :].values
        temporal_features_val = temporal_features.iloc[:-prediction_length, :].values
        temporal_features_test = temporal_features.iloc[:, :].values

        scaler = MinMaxScaler()
        scaler.fit(temporal_features_train)  # Fit ONLY on train
        temporal_features_train = scaler.transform(temporal_features_train)
        temporal_features_val = scaler.transform(temporal_features_val)
        temporal_features_test = scaler.transform(temporal_features_test)

        # Step 9: Create datasets
        train_ds = create_list_dataset(
            df_train, ts_code, start_train, freq, temporal_features_train
        )
        val_ds = create_list_dataset(
            df_val, ts_code, start_train, freq, temporal_features_val
        )
        test_ds = create_list_dataset(
            df_test, ts_code, start_train, freq, temporal_features_test
        )
    else:
        # Step 9: Create datasets
        train_ds = create_list_dataset(
            df_train, ts_code, start_train, freq
        )
        val_ds = create_list_dataset(
            df_val, ts_code, start_train, freq
        )
        test_ds = create_list_dataset(
            df_test, ts_code, start_train, freq
        )
        
    return train_ds, val_ds, test_ds, ts_code, df_input

def create_list_dataset(data, ts_code, start_date, freq, temporal_features=None):
    if temporal_features is not None:
        return ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start_date,
                FieldName.FEAT_STATIC_CAT: [fsc],
                FieldName.FEAT_DYNAMIC_REAL: temporal_features.T
            }
            for target, fsc in zip(data.T, ts_code)
        ], freq=freq)
    else:
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
                    ):
    all_results = []

    for i, (tss_series, forecast) in enumerate(zip(tss, forecasts)):
        latest_tss = tss_series.iloc[-prediction_length:].values.flatten()
        predictions_mean = forecast.mean
        pdv_codigo_name = df_input.columns[i + 1]

        
        predictions_median = forecast.median
        results = pd.DataFrame({
            'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
            'cant_vta': latest_tss,
            f'cant_vta_pred_{model_name}_mean': predictions_mean,
            f'cant_vta_pred_{model_name}_median': predictions_median,
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

def train_best_model(val_ds,
                    model_class, hyperparams, fixed_params=None, best_epochs=None):
    '''
    Train the final model with best hyperparameters
    
    Parameters:
    -----------
    train_ds : Training dataset
    model_class : The model estimator class (e.g. DeepAREstimator)
    hyperparams : Best hyperparameters from search
    fixed_params : Dict of fixed parameters for the model (optional)
    best_epochs : Optimal number of epochs determined during hyperparameter search
    '''
    if fixed_params is None:
        fixed_params = {}
    updated_fixed_params = fixed_params.copy()
    if best_epochs is not None:
        updated_fixed_params['trainer_kwargs']['max_epochs'] = best_epochs
    # Combine all parameters
    all_params = {**fixed_params, **hyperparams}
    # Remove early stopping for final training (since we're using the optimal epochs)
    if 'callbacks' in updated_fixed_params['trainer_kwargs']:
        updated_fixed_params['trainer_kwargs']['callbacks'] = [
            cb for cb in updated_fixed_params['trainer_kwargs']['callbacks'] 
            if not isinstance(cb, EarlyStopping)
        ]
    # Combine all parameters
    all_params = {**updated_fixed_params, **hyperparams}
    # Create and train estimator
    estimator = model_class(
        **all_params
    )
    
    predictor = estimator.train(training_data=val_ds)
    return predictor


def get_custom_time_features(freq: str) -> List[TimeFeature]:
    return [
        month_of_year,
        day_of_month,
        day_of_week,
        day_of_year,
        week_of_year  
    ]

