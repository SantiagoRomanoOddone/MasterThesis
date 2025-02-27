import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch import DeepAREstimator
from gluonts.dataset.split import split



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

    # Plot predictions
    plt.plot(filtered["fecha_comercial"], filtered["cant_vta"], color="black")
    for forecast in forecasts:
        forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    plt.show()