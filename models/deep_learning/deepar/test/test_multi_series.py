import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
import numpy as np
from metrics.metrics import Metrics



def deepar(features):

    # Ensure the date column is datetime
    features['fecha_comercial'] = pd.to_datetime(features['fecha_comercial'])

    # Define the forecast horizon (e.g., 31 days)
    prediction_length = 31

    # Create a list of series grouped by (pdv_codigo, codigo_barras_sku)
    series_list = []
    for (store, sku), group in features.groupby(['pdv_codigo', 'codigo_barras_sku']):
        group = group.sort_values('fecha_comercial')
        # Ignore very short series that do not have enough history
        if len(group) < prediction_length + 1:
            continue
        target = group['cant_vta'].values
        start_date = group['fecha_comercial'].iloc[0]
        # Define static features (e.g., store and product)
        feat_static_cat = [int(store), int(sku)]
        series_list.append({
            "start": start_date,
            "target": target,
            "feat_static_cat": feat_static_cat
        })

    # Create the full dataset (with complete series) for prediction
    full_dataset = ListDataset(series_list, freq="D")

    # For training, use the historical part of each series (excluding the last 'prediction_length' points)
    train_series_list = []
    for series in series_list:
        if len(series["target"]) > prediction_length:
            train_series_list.append({
                "start": series["start"],
                "target": series["target"][:-prediction_length],
                "feat_static_cat": series["feat_static_cat"]
            })
    train_dataset = ListDataset(train_series_list, freq="D")

    # Train the global DeepAR model
    model = DeepAREstimator(
        prediction_length=prediction_length,
        freq="D",
        trainer_kwargs={"max_epochs": 5}  # Adjust epochs as needed
    ).train(train_dataset)

    # Generate forecasts using the full dataset
    forecast_it = model.predict(full_dataset)
    forecasts = list(forecast_it)

    # Build a DataFrame with forecasts for each series
    results = []
    for series, forecast in zip(series_list, forecasts):
        # Retrieve static features (store and sku)
        store, sku = series["feat_static_cat"]

        # Retrieve the series start date
        start_date = series["start"]
        # Convert start_date to Timestamp if it is a Period
        if isinstance(start_date, pd.Period):
            start_date = start_date.to_timestamp()
            
        # Compute forecast start: last training date + 1 day.
        forecast_start_date = start_date + pd.Timedelta(days=len(series["target"]) - prediction_length)
        # Convert forecast_start_date if needed
        if isinstance(forecast_start_date, pd.Period):
            forecast_start_date = forecast_start_date.to_timestamp()

        # Create forecast dates for the horizon
        dates = pd.date_range(start=forecast_start_date, periods=prediction_length, freq="D")
        # Ensure predictions are non-negative
        pred_series = np.maximum(0, forecast.mean)
        df_pred = pd.DataFrame({
            'fecha_comercial': dates,
            'codigo_barras_sku': sku,
            'pdv_codigo': store,
            'cant_vta_pred_deepar': pred_series
        })
        results.append(df_pred)

    final_results = pd.concat(results, ignore_index=True)
    return final_results

if __name__ == '__main__':
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet')
    features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]

    results = deepar(features)

    # Merge the results with the test data
    test_df = features
    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo', 'fecha_comercial'], how='left')

    # Evaluate the model using metrics
    summary_df = Metrics().create_summary_dataframe(test_df)

    # Display the counts of best RMSE and MSE values
    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())