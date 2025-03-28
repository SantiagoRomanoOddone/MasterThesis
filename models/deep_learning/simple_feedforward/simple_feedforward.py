from gluonts.torch import SimpleFeedForwardEstimator
from models.deep_learning.gluonts.functions import (check_data_requirements, 
                                                    set_random_seed, 
                                                    prepare_dataset, 
                                                    make_predictions,
                                                    general_random_search,
                                                    process_results,
                                                    train_best_model)

import pandas as pd
import numpy as np
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
import pandas as pd
from metrics.metrics import Metrics


FREQ = "D"
PREDICTION_LENGTH = 30
START_TRAIN = pd.Timestamp("2022-12-01")
START_TEST = pd.Timestamp("2024-11-01")
END_TEST = pd.Timestamp("2024-11-30")
N_TRIALS = 4 

def get_hiperparameter_space(prediction_length):
   
    sff_space = {
        "context_length": [5 * prediction_length, 10 * prediction_length, 15 * prediction_length],  # Context window
        "hidden_dimensions": [[20, 20], [50, 50], [100, 50, 50]],  # Hidden layer sizes
        "lr": [0.001, 0.005, 0.01],  # Learning rate
        "weight_decay": [1e-8, 1e-6, 1e-4],  # Weight decay regularization
        "batch_norm": [True, False],  # Batch normalization
        "batch_size": [16, 32, 64],  # Batch size
        "num_batches_per_epoch": [25, 50, 100],  # Batches per epoch
    }
    
    sff_fixed = {
        "prediction_length": prediction_length,
        "trainer_kwargs": {"max_epochs": 5},
        # "num_feat_dynamic_real": 12
    }
    
    return sff_space, sff_fixed

# Main function
def sff_main(features):
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
                prediction_length=PREDICTION_LENGTH,
                temporal_features=True # Use temporal features
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} in prepare dataset due to error: {e}")
            continue

        # Train the model
        try:

            # Random Search
            sff_space, sff_fixed = get_hiperparameter_space(PREDICTION_LENGTH)
            best_params= general_random_search(
            train_ds, val_ds, PREDICTION_LENGTH,
            model_class=SimpleFeedForwardEstimator,
            hyperparameter_space=sff_space,
            n_trials=N_TRIALS,
            fixed_params=sff_fixed
            )   

            # Train the final model with the best hyperparameters
            predictor = train_best_model(
            val_ds=val_ds,  
            model_class=SimpleFeedForwardEstimator,
            hyperparams=best_params,
            fixed_params=sff_fixed
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
            model_name="sff",
            median=False
        )

        # Append results for the current SKU
        all_final_results.append(final_results)


    combined_results = pd.concat(all_final_results, ignore_index=True)
    return combined_results

if __name__ == "__main__":
    # Constants
    CLUSTER_NUMBER = 1
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

    filter = filtered['codigo_barras_sku'].unique()[:2]
    filtered = filtered[filtered['codigo_barras_sku'].isin(filter)]


    final_results = sff_main(filtered)
    
    validation = validation[validation['codigo_barras_sku'].isin(filter)]
    test_df = pd.merge(validation, final_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    summary_df = Metrics().create_summary_dataframe(test_df)
    print(summary_df['rmse_cant_vta_pred_sff_mean'].mean(), summary_df['rmse_cant_vta_pred_sff_mean'].median())
    print(summary_df)


    


