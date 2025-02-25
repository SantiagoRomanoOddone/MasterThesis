import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator

# Cargar y ordenar el dataset
features = pd.read_parquet("features/processed/features.parquet")
features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

# Filtrar por cluster, tienda y producto
cluster_number = 3
filtered = features[features['cluster'] == cluster_number]
filtered = filtered[(filtered['pdv_codigo'] == 1) & (filtered['codigo_barras_sku'] == 7894900027082)]

# Asegurarse de que la columna fecha es datetime
filtered['fecha_comercial'] = pd.to_datetime(filtered['fecha_comercial'])

# Extraer la serie de tiempo y la fecha de inicio
target = filtered['cant_vta'].values
start_date = filtered['fecha_comercial'].iloc[0]

# Opcional: incorporar las variables estáticas (en este caso, pdv_codigo y codigo_barras_sku)
feat_static_cat = [int(filtered['pdv_codigo'].iloc[0]), int(filtered['codigo_barras_sku'].iloc[0])]

# Definir la longitud del horizonte a pronosticar
prediction_length = 31

# Crear el dataset de entrenamiento: usamos la parte histórica (sin los últimos 'prediction_length' puntos)
train_target = target[:-prediction_length]

train_dataset = ListDataset(
    [{
        "start": start_date,
        "target": train_target,
        "feat_static_cat": feat_static_cat
    }],
    freq="D"
)

# Creamos el dataset completo (con la serie completa) para la predicción;
# el modelo usará los datos históricos para generar el pronóstico de los siguientes 31 días.
full_dataset = ListDataset(
    [{
        "start": start_date,
        "target": target,
        "feat_static_cat": feat_static_cat
    }],
    freq="D"
)

# Entrenar el modelo DeepAR
model = DeepAREstimator(
    prediction_length=prediction_length,
    freq="D",
    trainer_kwargs={"max_epochs": 5}  # Aumenta los epochs en producción
).train(train_dataset)

# Generar pronósticos usando el dataset completo
forecast_it = model.predict(full_dataset)
forecasts = list(forecast_it)

# Graficar los valores reales y el pronóstico
plt.figure(figsize=(10, 6))
plt.plot(filtered['fecha_comercial'], target, color="black", label="Valores reales")
for forecast in forecasts:
    forecast.plot(color="red")
plt.legend(fontsize="xx-large")
plt.title("Pronóstico para pdv_codigo=1 & codigo_barras_sku=7894900027082")
plt.xlabel("Fecha")
plt.ylabel("cant_vta")
plt.show()