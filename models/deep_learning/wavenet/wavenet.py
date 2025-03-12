from gluonts.torch import WaveNetEstimator
from gluonts.time_feature import get_seasonality
from models.deep_learning.gluonts.functions import (check_data_requirements, 
                                                    set_random_seed, 
                                                    prepare_dataset, 
                                                    create_list_dataset,
                                                    make_predictions)

import numpy as np
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
import pandas as pd
import numpy as np


CLUSTER_NUMBER = 3
FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")

# Train the DeepAR model
def train_wavenet_model(df_train, ts_code, start_date, freq, prediction_length):

    estimator = WaveNetEstimator(
        freq=freq,  # Required: Frequency of the time series (e.g., "D" for daily)
        prediction_length=prediction_length,  # Required: Length of the prediction horizon
        num_bins=1024,  # Optional: Number of bins for discretization (default is 1024)
        num_residual_channels=24,  # Optional: Number of residual channels (default is 24)
        num_skip_channels=32,  # Optional: Number of skip channels (default is 32)
        dilation_depth=None,  # Optional: Depth of the dilation layers (default is None, which auto-computes)
        num_stacks=1,  # Optional: Number of stacks of dilated convolutions (default is 1)
        temperature=1.0,  # Optional: Temperature for sampling (default is 1.0)
        num_feat_dynamic_real=0,  # Optional: Number of dynamic real features (default is 0)
        num_feat_static_cat=0,  # Optional: Number of static categorical features (default is 0)
        num_feat_static_real=0,  # Optional: Number of static real features (default is 0)
        cardinality=[1],  # Optional: Cardinality of static categorical features (default is [1])
        seasonality=get_seasonality(FREQ),  # Optional: Seasonality of the time series (default is inferred from freq)
        embedding_dimension=5,  # Optional: Dimension of embeddings for categorical features (default is 5)
        use_log_scale_feature=True,  # Optional: Whether to use log scale feature (default is True)
        time_features=None,  # Optional: List of time features (default is None, which uses default features)
        lr=0.001,  # Optional: Learning rate (default is 0.001)
        weight_decay=1e-08,  # Optional: Weight decay for regularization (default is 1e-08)
        train_sampler=None,  # Optional: Custom training sampler (default is None)
        validation_sampler=None,  # Optional: Custom validation sampler (default is None)
        batch_size=32,  # Optional: Batch size (default is 32)
        num_batches_per_epoch=50,  # Optional: Number of batches per epoch (default is 50)
        num_parallel_samples=100,  # Optional: Number of parallel samples for prediction (default is 100)
        negative_data=False,  # Optional: Whether to allow negative data (default is False)
        trainer_kwargs={"max_epochs": 5},  # Optional: Additional trainer arguments (e.g., max_epochs)
    )

    train_ds = create_list_dataset(df_train, ts_code, start_date, freq)
    predictor = estimator.train(training_data=train_ds)
    return predictor

# Process results
def process_wavenet_results(tss, forecasts, df_input, start_test, freq, prediction_length, sku):
    all_results = []

    for i, (tss_series, forecast) in enumerate(zip(tss, forecasts)):
        latest_tss = tss_series.iloc[-prediction_length:].values.flatten()
        pdv_codigo_name = df_input.columns[i + 1]

        results = pd.DataFrame({
            'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
            'cant_vta': latest_tss,
            'cant_vta_pred_wavenet': forecast.mean,
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

# Main function
def wavenet_main(features):

    set_random_seed(42)

    unique_skus = features['codigo_barras_sku'].unique()

    min_data_points = PREDICTION_LENGTH + 100 
    valid_skus = []
    for sku in unique_skus:
        sku_data = features[features['codigo_barras_sku'] == sku]
        if check_data_requirements(sku_data, min_data_points):
            valid_skus.append(sku)

    all_final_results = []
    for sku in valid_skus:
      print(f"Processing SKU: {sku}")
      filtered = features[(features["codigo_barras_sku"] == sku)].copy()
      # Skip if no data is available for the SKU
      if len(filtered) == 0:
          print(f"No data available for SKU: {sku}")
          continue

      # Check for NaNs or invalid values
      if filtered.isnull().any().any():
          print(f"SKU {sku} contains NaNs. Skipping.")
          continue

      # Prepare dataset
      try:
          df_train, df_test, ts_code, df_input = prepare_dataset(
              data=filtered,
              end_test=END_TEST,
              freq=FREQ,
              prediction_length=PREDICTION_LENGTH
          )
      except ValueError as e:
          print(f"Skipping SKU {sku} in prepare dataset due to error: {e}")
          continue

      # Train the model
      try:
          predictor = train_wavenet_model(
              df_train=df_train,
              ts_code=ts_code,
              start_date=START_TRAIN,
              freq=FREQ,
              prediction_length=PREDICTION_LENGTH
          )
      except ValueError as e:
          print(f"Skipping SKU {sku} in training due to error: {e}")
          continue

      # Make predictions
      tss, forecasts = make_predictions(
          predictor=predictor,
          df_test=df_test,
          ts_code=ts_code,
          start_date=START_TRAIN,
          freq=FREQ
      )

      # Process results
      final_results = process_wavenet_results(
          tss=tss,
          forecasts=forecasts,
          df_input=df_input,
          start_test=START_TEST,
          freq=FREQ,
          prediction_length=PREDICTION_LENGTH,
          sku=sku
      )

      # Append results for the current SKU
      all_final_results.append(final_results)


    combined_results = pd.concat(all_final_results, ignore_index=True)
    return combined_results


if __name__ == "__main__":
    # Constants
    CLUSTER_NUMBER = 3
    FREQ = "D"
    PREDICTION_LENGTH = 30
    START_TRAIN = pd.Timestamp("2022-12-01")
    START_TEST = pd.Timestamp("2024-11-01")
    END_TEST = pd.Timestamp("2024-11-30")

    DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/features.parquet"

    features = pd.read_parquet(DATA_PATH)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 'cant_vta', 'cluster']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    filtered = features[features["cluster"] == CLUSTER_NUMBER]
    filtered = filtered[filtered['fecha_comercial'] <= END_TEST]
    validation = filtered[filtered['fecha_comercial'] >= START_TEST]
    filtered = filtered[filtered['fecha_comercial'] < START_TEST]

    filter = filtered['codigo_barras_sku'].unique()[:3]

    filtered = filtered[filtered['codigo_barras_sku'].isin(filter)]


    final_results = wavenet_main(filtered)
    print(final_results)