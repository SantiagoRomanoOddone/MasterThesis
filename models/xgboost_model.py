
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from metrics.metrics import Metrics


features = pd.read_parquet('features/processed/features.parquet')
features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

cluster_3 = features[features['cluster'] == 3]

pdv_1_sku_7894900027013 = cluster_3[(cluster_3['codigo_barras_sku'] == 7894900027013) & (cluster_3['pdv_codigo'] == 1)]

split_date = '2024-11-10'
train_df = pdv_1_sku_7894900027013[pdv_1_sku_7894900027013['fecha_comercial'] < split_date]
test_df = pdv_1_sku_7894900027013[pdv_1_sku_7894900027013['fecha_comercial'] >= split_date]
print(train_df.shape, test_df.shape)

features = ['imp_vta', 'stock', 'year', 'month', 'day', 'day_of_week',
            'is_weekend', 'quarter', 'week_of_year', 'day_of_year', 'is_month_start', 'is_month_end', 'is_first_week',
            'is_last_week', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30', 'lag_1', 'lag_7',
            'lag_30', 'diff_1', 'diff_7', 'diff_30']
target = 'cant_vta'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = Metrics.mean_squared_error(y_test, y_pred)
rmse = Metrics.root_mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
