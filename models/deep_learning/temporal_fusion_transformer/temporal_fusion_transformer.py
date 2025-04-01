import pandas as pd
from train.hyperparam_search.hyperparam_search import hyperparameter_search
from gluonts.torch import TemporalFusionTransformerEstimator
from models.deep_learning.gluonts.functions import (check_data_requirements, 
                                                    set_random_seed, 
                                                    prepare_dataset, 
                                                    make_predictions,
                                                    process_results,
                                                    train_best_model,
                                                    get_custom_time_features)

import numpy as np
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
from metrics.metrics import Metrics
from lightning.pytorch.callbacks import EarlyStopping


FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")
N_TRIALS = 10

def get_tft_hyperparameter_space(ts_code):
    """Returns hyperparameter search space with types for Bayesian optimization"""
    tft_space = {
        # --- Architecture ---
        "hidden_dim": {
            "type": "categorical",
            "values": [32, 64, 128]
        },
        "variable_dim": {
            "type": "categorical",
            "values": [16, 24, 32]
        },
        "num_heads": {
            "type": "categorical",
            "values": [4, 8]
        },
        # --- Optimization ---
        "lr": {
            "type": "float",
            "low": 0.001,
            "high": 0.01,
            "log": True
        },
        "weight_decay": {
            "type": "float",
            "low": 1e-8,
            "high": 1e-4,
            "log": True
        },
        # --- Regularization ---
        "dropout_rate": {
        "type": "float",
        "low": 0.15,
        "high": 0.35
        },
        # --- Training ---
        "batch_size": {
        "type": "categorical",
        "values": [32, 64]
        },
        # --- Time Series Context ---
        "context_length": {
            "type": "categorical",
            "values": [PREDICTION_LENGTH, 2*PREDICTION_LENGTH]  
        }
    }
    
    # Fixed params remain the same
    tft_fixed = {
        "context_length": PREDICTION_LENGTH,
        "static_cardinalities": [len(np.unique(ts_code))],
        "trainer_kwargs": {"max_epochs": 20,
                            "callbacks": [
                            EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
                            ]
                           },
        "time_features": get_custom_time_features(FREQ),
        "freq": FREQ,
        "prediction_length": PREDICTION_LENGTH,
    }
    return tft_space, tft_fixed

# Main function
def tft_main(features):
    set_random_seed(42)

    unique_skus = features['codigo_barras_sku'].unique()

    min_data_points = PREDICTION_LENGTH + 100  # Adjust this threshold as needed
    valid_skus = []
    for sku in unique_skus:
        sku_data = features[features['codigo_barras_sku'] == sku]
        if check_data_requirements(sku_data, min_data_points):
            valid_skus.append(sku)

    all_final_results = []
    # valid_skus = valid_skus[40:50]
    for sku in valid_skus:
        print(f"Processing SKU: {sku}")
        filtered = features[(features["codigo_barras_sku"] == sku)].copy()

        # Prepare dataset
        try:
            train_ds , val_ds, test_ds, ts_code, df_input = prepare_dataset(
                data=filtered,
                start_train=START_TRAIN,
                end_test=END_TEST,
                freq=FREQ,
                prediction_length=PREDICTION_LENGTH,
                # temporal_features=True
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} in prepare dataset due to error: {e}")
            continue

        # Train the model
        try:
            # Get TFT-specific parameters
            tft_space, tft_fixed = get_tft_hyperparameter_space(ts_code)
            best_params, best_epochs = hyperparameter_search(
                train_ds, val_ds, PREDICTION_LENGTH,
                model_class=TemporalFusionTransformerEstimator,
                hyperparameter_space=tft_space,
                n_trials=N_TRIALS,
                fixed_params=tft_fixed,
                type='bayesian'
            )
            # Train the final model with the best hyperparameters
            predictor = train_best_model(
                val_ds=val_ds,  
                model_class=TemporalFusionTransformerEstimator,
                hyperparams=best_params,
                fixed_params=tft_fixed,
                best_epochs=best_epochs 
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} in training due to error: {e}")
            continue

        # Make predictions
        tss, forecasts = make_predictions(
                predictor=predictor,
                test_ds =test_ds 
        )

        # Process results
        final_results = process_results(
            tss=tss,
            forecasts=forecasts,
            df_input=df_input,
            start_test=START_TEST,
            freq=FREQ,
            prediction_length=PREDICTION_LENGTH,
            sku=sku,
            model_name="tft"
        )

        # Append results for the current SKU
        all_final_results.append(final_results)

    combined_results = pd.concat(all_final_results, ignore_index=True)
    return combined_results




if __name__ == "__main__":

    # Constants
    CLUSTER_NUMBER = 0
    FREQ = "D"
    PREDICTION_LENGTH = 30
    START_TRAIN = pd.Timestamp("2022-12-01")
    START_TEST = pd.Timestamp("2024-11-01")
    END_TEST = pd.Timestamp("2024-11-30")

    DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/cleaned_features.parquet"

    features = pd.read_parquet(DATA_PATH)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 
        'cant_vta','cluster_sku']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    filtered = features[features["cluster_sku"] == CLUSTER_NUMBER]
    filtered = filtered[filtered['fecha_comercial'] <= END_TEST]
    validation = filtered[filtered['fecha_comercial'] >= START_TEST]
    filtered = filtered[filtered['fecha_comercial'] < START_TEST]

    filter = filtered['codigo_barras_sku'].unique()[:1]
    filtered = filtered[filtered['codigo_barras_sku'].isin(filter)]

    final_results = tft_main(filtered)

    validation = validation[validation['codigo_barras_sku'].isin(filter)]
    test_df = pd.merge(validation, final_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    summary_df = Metrics().create_summary_dataframe(test_df)
    print(summary_df)
    print(final_results)

    # summary_df.to_csv(f'results/metrics_cluster_{CLUSTER_NUMBER}_tft_random.csv', index=False)