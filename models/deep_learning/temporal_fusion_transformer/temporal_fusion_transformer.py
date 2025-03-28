import pandas as pd
from gluonts.torch import TemporalFusionTransformerEstimator
from models.deep_learning.gluonts.functions import (check_data_requirements, 
                                                    set_random_seed, 
                                                    prepare_dataset, 
                                                    make_predictions,
                                                    general_random_search,
                                                    process_results,
                                                    train_best_model)
import numpy as np
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")
N_TRIALS = 4  # Number of trials for random search

def get_tft_hiperparameter_space():
    """Returns hyperparameter search space and fixed parameters for TFT model"""
    tft_space = {
        "hidden_dim": [16, 32, 64, 128],        # Hidden layer size
        "variable_dim": [8, 16, 32],             # Variable dimension
        "num_heads": [2, 4, 8],                  # Attention heads
        "dropout_rate": [0.1, 0.2, 0.3],         # Dropout rate
        "lr": [0.001, 0.005, 0.01],              # Learning rate
        "weight_decay": [1e-8, 1e-6, 1e-4],       # Regularization
        "batch_size": [16, 32, 64],               # Batch size
        "patience": [5, 10, 20]                   # Early stopping patience
    }
    
    tft_fixed = {
        "context_length": PREDICTION_LENGTH,
        "trainer_kwargs": {"max_epochs": 5}
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
                prediction_length=PREDICTION_LENGTH
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} in prepare dataset due to error: {e}")
            continue

        # Train the model
        try:
            # Get TFT-specific parameters
            tft_space, tft_fixed = get_tft_hiperparameter_space()

            # Random Search
            best_tft_params = general_random_search(
                train_ds, val_ds, ts_code, FREQ, PREDICTION_LENGTH,
                model_class=TemporalFusionTransformerEstimator,
                hyperparameter_space=tft_space,
                n_trials=N_TRIALS,
                fixed_params=tft_fixed
            )

            # Train final model
            predictor = train_best_model(
                val_ds=val_ds,  
                ts_code=ts_code,
                freq=FREQ,
                prediction_length=PREDICTION_LENGTH,
                model_class=TemporalFusionTransformerEstimator,
                hyperparams=best_tft_params,
                fixed_params=tft_fixed
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
    pass
    # # Constants
    # CLUSTER_NUMBER = 0
    # FREQ = "D"
    # PREDICTION_LENGTH = 30
    # START_TRAIN = pd.Timestamp("2022-12-01")
    # START_TEST = pd.Timestamp("2024-11-01")
    # END_TEST = pd.Timestamp("2024-11-30")

    # DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/cleaned_features.parquet"

    # features = pd.read_parquet(DATA_PATH)
    # features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 
    #     'cant_vta','cluster_sku']]
    # features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    # filtered = features[features["cluster_sku"] == CLUSTER_NUMBER]
    # filtered = filtered[filtered['fecha_comercial'] <= END_TEST]
    # validation = filtered[filtered['fecha_comercial'] >= START_TEST]
    # filtered = filtered[filtered['fecha_comercial'] < START_TEST]

    # filter = filtered['codigo_barras_sku'].unique()[:1]

    # filtered = filtered[filtered['codigo_barras_sku'].isin(filter)]

    # final_results = tft_main(filtered)
    # print(final_results)