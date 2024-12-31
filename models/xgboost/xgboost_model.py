import xgboost as xgb
import pandas as pd

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.verbose = 0
        self.excluded_columns = ['fecha_comercial', 'codigo_barras_sku', 'nombre_sku', 'imp_vta',
                                'cant_vta', 'stock','pdv_codigo', 'cluster']
        self.return_columns = ['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']



    def fit(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]
        target = 'cant_vta'

        X = data[features]
        y = data[target]

        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth)
        self.model.fit(X, y)

    def predict(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]

        X = data[features]
        y_pred = self.model.predict(X).astype(int)

        y_pred[y_pred < 0] = 0

        y_pred_df = pd.DataFrame(y_pred, columns=['cant_vta_pred'])
        y_pred_df.index = data.index

        df_pred = pd.merge(data, y_pred_df, left_index=True, right_index=True)

        df_pred = df_pred[self.return_columns]

        return df_pred 
