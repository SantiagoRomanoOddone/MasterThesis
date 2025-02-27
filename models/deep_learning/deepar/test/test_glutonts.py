import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import DeepAREstimator
from gluonts.dataset.split import split
from train.splits.fixed_split import fixed_split



if __name__ == '__main__':


    # Load and sort data
    cluster_number = 3
    features = pd.read_parquet("features/processed/features.parquet")
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)

    # Filter data for specific product/store
    filtered = features[
        (features["cluster"] == cluster_number) & 
        (features["pdv_codigo"] == 1) & 
        (features["codigo_barras_sku"] == 7894900027082)
    ].copy() 

    train_df, test_df = fixed_split(filtered)

    # Ensure fecha_comercial is datetime
    filtered["fecha_comercial"] = pd.to_datetime(filtered["fecha_comercial"])

    # Ensure target is numeric and remove NaNs
    filtered["cant_vta"] = pd.to_numeric(filtered["cant_vta"], errors="coerce")
    filtered = filtered.dropna(subset=["cant_vta"])

    # Select relevant columns
    filtered = filtered[['fecha_comercial', 'cant_vta']]

    # Fill missing dates
    filtered = filtered.set_index("fecha_comercial")
    full_date_range = pd.date_range(start=filtered.index.min(), end=filtered.index.max(), freq="D")
    filtered = filtered.reindex(full_date_range)
    filtered = filtered.ffill()  # Forward fill missing values
    filtered = filtered.reset_index().rename(columns={"index": "fecha_comercial"})

    # Create the PandasDataset
    dataset = PandasDataset(
        filtered,
        target="cant_vta",
        timestamp="fecha_comercial",
        freq="D"
    )

    # Split the data for training and testing
    training_data, test_gen = split(dataset, offset=-36)
    test_data = test_gen.generate_instances(prediction_length=12, windows=3)

    # Train the model and make predictions
    model = DeepAREstimator(
        prediction_length=12, freq="D", trainer_kwargs={"max_epochs": 5}
    ).train(training_data)

    forecasts = list(model.predict(test_data.input))

    # Extract predictions and combine with original data
    predictions = []
    for forecast in forecasts:
        # Get the predicted values
        pred_values = forecast.mean
        # Get the corresponding timestamps
        pred_dates = pd.date_range(start=forecast.start_date.to_timestamp(), periods=len(pred_values), freq="D")
        # Append to predictions list
        predictions.append(pd.DataFrame({
            "fecha_comercial": pred_dates,
            "cant_vta_pred": pred_values
        }))

    # Combine all predictions into a single DataFrame
    predictions_df = pd.concat(predictions, ignore_index=True)

    # Merge predictions with the original data
    result_df = pd.merge(test_df, predictions_df, on="fecha_comercial", how="left")

    # Add codigo_barras_sku and pdv_codigo columns
    result_df["codigo_barras_sku"] = 7894900027082
    result_df["pdv_codigo"] = 1

    # Reorder columns
    result_df = result_df[['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']]

    # Save or display the result
    print(result_df)

    