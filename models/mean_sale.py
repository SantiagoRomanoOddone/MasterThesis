import pandas as pd
import numpy as np
from metrics.metrics import Metrics

features = pd.read_parquet('features/processed/features.parquet')
features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

cluster_3 = features[features['cluster'] == 3]
pdv_1_sku_7894900027013 = cluster_3[(cluster_3['codigo_barras_sku'] == 7894900027013) & (cluster_3['pdv_codigo'] == 1)]

split_date = '2024-11-10'
train_df = pdv_1_sku_7894900027013[pdv_1_sku_7894900027013['fecha_comercial'] < split_date]
test_df = pdv_1_sku_7894900027013[pdv_1_sku_7894900027013['fecha_comercial'] >= split_date]

target = 'cant_vta'
test_df['rolling_mean_30'] = test_df[target].rolling(window=30, min_periods=1).mean()

y_pred_rolling_mean = test_df['rolling_mean_30']
y_actual = test_df[target]

mse_rolling_mean = Metrics.mean_squared_error(y_actual, y_pred_rolling_mean)
rmse_rolling_mean = Metrics.root_mean_squared_error(y_actual, y_pred_rolling_mean)

print(f'Mean Squared Error (Rolling Mean 30 days): {mse_rolling_mean}')
print(f'Root Mean Squared Error (Rolling Mean 30 days): {rmse_rolling_mean}')
