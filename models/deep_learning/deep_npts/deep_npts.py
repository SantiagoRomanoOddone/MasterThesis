from gluonts.torch import DeepNPTSEstimator
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributions")




CLUSTER_NUMBER = 3
FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")

# Train the DeepAR model
def train_deepnpts_model(df_train, ts_code, start_date, freq, prediction_length):
    estimator = DeepNPTSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=prediction_length * 2,  # Typically, context_length is set to be equal to or greater than prediction_length
            num_hidden_nodes=[50, 50],  # Example architecture with two hidden layers of 50 nodes each
            batch_norm=True,  # Apply batch normalization
            use_feat_static_cat=True,  # Use static categorical features if available
            num_feat_static_real=0,  # Number of static real-valued features
            num_feat_dynamic_real=0,  # Number of dynamic real-valued features
            cardinality=[len(np.unique(ts_code))],  # Cardinality of categorical features
            embedding_dimension=None,  # Default embedding dimensions
            input_scaling="min_max_scaling",  # Apply min-max scaling to input features
            dropout_rate=0.1,  # Dropout rate for regularization
            epochs=5,  # Number of training epochs
            lr=0.001,  # Learning rate
            batch_size=32,  # Batch size
            num_batches_per_epoch=50,  # Number of batches per epoch
            cache_data=False,  # Whether to cache data for faster training
            loss_scaling=None, # Scaling applied to the loss function
        )


    train_ds = create_list_dataset(df_train, ts_code, start_date, freq)
    predictor = estimator.train(training_data=train_ds)
    return predictor

# Process results
def process_deepnpts_results(tss, forecasts, df_input, start_test, freq, prediction_length, sku):
    all_results = []

    for i, (tss_series, forecast) in enumerate(zip(tss, forecasts)):
        latest_tss = tss_series.iloc[-prediction_length:].values.flatten()
        pdv_codigo_name = df_input.columns[i + 1]

        results = pd.DataFrame({
            'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
            'cant_vta': latest_tss,
            'cant_vta_pred_deepnpts': forecast.mean,
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
def deepnpts_main(features):

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
          predictor = train_deepnpts_model(
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
      final_results = process_deepnpts_results(
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


    final_results = deepnpts_main(filtered)
    print(final_results)