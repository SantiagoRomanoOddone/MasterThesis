import os
import json
import pandas as pd
from pathlib import Path
from gluonts.torch import SimpleFeedForwardEstimator
from metrics.metrics import Metrics
from models.deep_learning.gluonts.functions import (
    check_data_requirements, 
    set_random_seed,
    prepare_dataset,
    train_best_model,
    make_predictions,
    process_results,
)


def run_inference_for_skus(cluster_number: int, prediction_length: int, freq: str,
                            start_train, start_test, end_test, data_path: str, 
                            hyperparam_dir: str):
    
    set_random_seed(42)

    # Load and filter features
    features = pd.read_parquet(data_path)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 
                         'cant_vta', 'cluster_sku']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)

    features = features[features["cluster_sku"] == cluster_number]
    features = features[features['fecha_comercial'] <= end_test]
    
    validation = features[features['fecha_comercial'] >= start_test]
    filtered = features[features['fecha_comercial'] < start_test]

    all_final_results = []

    unique_skus = filtered["codigo_barras_sku"].unique()
    min_data_points = prediction_length + 100

    for sku in unique_skus:
        sku_data = filtered[filtered['codigo_barras_sku'] == sku]
        if not check_data_requirements(sku_data, min_data_points):
            print(f"Skipping SKU {sku} due to insufficient data.")
            continue

        json_path = Path(hyperparam_dir) / f"best_hyperparameters_sku_{sku}_cluster_{cluster_number}.json"
        if not json_path.exists():
            print(f"No hyperparameter file for SKU {sku}, skipping.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        print(f"Running inference for SKU {sku}")

        try:
            train_ds, val_ds, test_ds, ts_code, df_input = prepare_dataset(
                data=sku_data,
                start_train=start_train,
                end_test=end_test,
                freq=freq,
                prediction_length=prediction_length,
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} due to error in dataset prep: {e}")
            continue

        best_params = data["best_params"]
        best_epochs = data["best_epochs"]
        sff_fixed = {
            "prediction_length": prediction_length,
            "trainer_kwargs": {
                "max_epochs": best_epochs
            }
        }

        try:
            predictor = train_best_model(
                val_ds=val_ds,
                model_class=SimpleFeedForwardEstimator,
                hyperparams=best_params,
                fixed_params=sff_fixed,
                best_epochs=best_epochs
            )
        except ValueError as e:
            print(f"Skipping SKU {sku} due to training error: {e}")
            continue

        # Predict
        tss, forecasts = make_predictions(
            predictor=predictor,
            test_ds=test_ds
        )

        # Process results
        final_results = process_results(
            tss=tss,
            forecasts=forecasts,
            df_input=df_input,
            start_test=start_test,
            freq=freq,
            prediction_length=prediction_length,
            sku=sku,
            model_name="sff"
        )

        all_final_results.append(final_results)

    if not all_final_results:
        print("No results generated.")
        return None

    combined_results = pd.concat(all_final_results, ignore_index=True)

    # Merge with real November data
    validation = validation[validation['codigo_barras_sku'].isin(combined_results['codigo_barras_sku'].unique())]
    test_df = pd.merge(validation, combined_results, 
                       on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], 
                       how='left')
    
    return test_df

if __name__ == "__main__":
    CLUSTER_NUMBER = 1
    MODEL = "sff"
    test_df = run_inference_for_skus(
        cluster_number=CLUSTER_NUMBER,
        prediction_length=30,
        freq="D",
        start_train=pd.Timestamp("2022-12-01"),
        start_test=pd.Timestamp("2024-11-01"),
        end_test=pd.Timestamp("2024-11-30"),
        data_path="/Users/santiagoromano/Documents/code/MasterThesis/features/processed/cleaned_features.parquet",
        hyperparam_dir="/Users/santiagoromano/Documents/code/MasterThesis/results/simple_feedforward"
    )

    if test_df is not None:
        summary_df = Metrics().create_summary_dataframe(test_df)
        print(summary_df['rmse_cant_vta_pred_sff_mean'].median(), summary_df['rmse_cant_vta_pred_sff_mean'].mean())
        summary_df.to_csv(f'results/inference/metrics_cluster_{CLUSTER_NUMBER}_{MODEL}.csv', index=False)

