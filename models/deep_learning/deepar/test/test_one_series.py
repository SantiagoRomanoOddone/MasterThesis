import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator

# Load and sort the dataset
features = pd.read_parquet("features/processed/features.parquet")
features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

# Filter by cluster, store, and product
cluster_number = 3
filtered = features[features['cluster'] == cluster_number]
filtered = filtered[(filtered['pdv_codigo'] == 1) & (filtered['codigo_barras_sku'] == 7894900027082)]

# Ensure the date column is datetime
filtered['fecha_comercial'] = pd.to_datetime(filtered['fecha_comercial'])

# Extract the time series and the start date
target = filtered['cant_vta'].values
start_date = filtered['fecha_comercial'].iloc[0]

# Optional: incorporate static variables (in this case, pdv_codigo and codigo_barras_sku)
feat_static_cat = [int(filtered['pdv_codigo'].iloc[0]), int(filtered['codigo_barras_sku'].iloc[0])]

# Define the length of the forecast horizon
prediction_length = 31

# Create the training dataset: use the historical part (without the last 'prediction_length' points)
train_target = target[:-prediction_length]

train_dataset = ListDataset(
    [{
        "start": start_date,
        "target": train_target,
        "feat_static_cat": feat_static_cat
    }],
    freq="D"
)

# Create the full dataset (with the complete series) for prediction;
# the model will use the historical data to generate the forecast for the next 31 days.
full_dataset = ListDataset(
    [{
        "start": start_date,
        "target": target,
        "feat_static_cat": feat_static_cat
    }],
    freq="D"
)

# Train the DeepAR model
model = DeepAREstimator(
    prediction_length=prediction_length,
    freq="D",
    trainer_kwargs={"max_epochs": 5}  # Increase the epochs in production
).train(train_dataset)

# Generate forecasts using the full dataset
forecast_it = model.predict(full_dataset)
forecasts = list(forecast_it)

# Plot the actual values and the forecast
plt.figure(figsize=(10, 6))
plt.plot(filtered['fecha_comercial'], target, color="black", label="Actual Sales")
for forecast in forecasts:
    forecast.plot(color="red")
plt.legend(fontsize="xx-large")
plt.title("Forecast for pdv_codigo=1 & codigo_barras_sku=7894900027082")
plt.xlabel("Date")
plt.ylabel("cant_vta")
plt.show()