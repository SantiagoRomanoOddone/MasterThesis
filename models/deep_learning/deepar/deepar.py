from gluonts.torch import DeepAREstimator
from models.deep_learning.gluonts.functions import (check_data_requirements, 
                                                    set_random_seed, 
                                                    prepare_dataset, 
                                                    make_predictions,
                                                    process_results)

import numpy as np
import pandas as pd
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


CLUSTER_NUMBER = 3
FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")
N_TRIALS = 2  # Number of trials for random search


def train_best_model(val_ds, ts_code, freq, prediction_length, hyperparams):
    '''Train the model with the best hyperparameters'''
    estimator = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=hyperparams["num_layers"],
    hidden_size=hyperparams["hidden_size"],
    lr=hyperparams["lr"],
    weight_decay=hyperparams["weight_decay"],
    dropout_rate=hyperparams["dropout_rate"],
    batch_size=hyperparams["batch_size"],
    num_feat_static_cat=1,
    num_feat_dynamic_real=0,
    num_feat_static_real=0,
    cardinality=[len(np.unique(ts_code))],
    num_parallel_samples=100,
    trainer_kwargs={"max_epochs": 5},
    )
    predictor = estimator.train(training_data=val_ds)
    return predictor


def random_search_params(train_ds, val_ds, ts_code, freq, prediction_length):
    '''Random search for hyperparameters'''
    
    # Hyperparameter search space
    hyperparameter_space = {
        "num_layers": [1, 2, 3],
        "hidden_size": [16, 32, 64, 128],
        "lr": [0.001, 0.005, 0.01],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
        "weight_decay": [1e-8, 1e-6, 1e-4],
    }

    # Randomly sample N sets of hyperparameters
    random_hyperparameter_sets = [
        {key: random.choice(values) for key, values in hyperparameter_space.items()}
        for _ in range(N_TRIALS)
    ]

    best_rmse = float("inf")
    best_hyperparams = None

    for hyperparams in random_hyperparameter_sets:
        print(f"Training with hyperparams: {hyperparams}")

        # Define the model with the sampled hyperparameters
        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_layers=hyperparams["num_layers"],
            hidden_size=hyperparams["hidden_size"],
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            dropout_rate=hyperparams["dropout_rate"],
            batch_size=hyperparams["batch_size"],
            num_feat_static_cat=1,
            num_feat_dynamic_real=0,
            num_feat_static_real=0,
            cardinality=[len(np.unique(ts_code))],
            num_parallel_samples=100,
            trainer_kwargs={"max_epochs": 5},
        )
        predictor = estimator.train(training_data=train_ds)

        # Make validation predictions
        tss, forecasts = make_predictions(
            predictor=predictor,
            test_ds=val_ds
        )

        # Compute RMSE as evaluation metricvalidation_step`
        predictions_mean = np.array([forecast.mean for forecast in forecasts])
        actuals = np.array([ts.iloc[-prediction_length:].values for ts in tss])

        actuals = actuals.reshape(predictions_mean.shape)

        rmse = np.sqrt(np.mean((predictions_mean - actuals) ** 2))

        print(f"RMSE for this model: {rmse}")

        # Store the best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_hyperparams = hyperparams

    print(f"Best Hyperparameters: {best_hyperparams}, RMSE: {best_rmse}")
    return best_hyperparams

# Main function
def deepar_main(features):
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
            # Random Search
            best_params= random_search_params(
                train_ds=train_ds,
                val_ds=val_ds,
                ts_code=ts_code,
                freq=FREQ,
                prediction_length=PREDICTION_LENGTH
            )
            # Train the final model with the best hyperparameters
            predictor = train_best_model(
                val_ds=val_ds,
                ts_code=ts_code,
                freq=FREQ,
                prediction_length=PREDICTION_LENGTH,
                hyperparams=best_params
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
            model_name="deepar",
            median=True
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


    final_results = deepar_main(filtered)
    print(final_results)