import pandas as pd
import numpy as np
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
import mxnet as mx

class DeepARModel:
    def __init__(self, freq='D', prediction_length=31):
        self.freq = freq
        self.prediction_length = prediction_length
        self.model = None

    def format_data(self, data, is_train=True):
        series = []
        for (pdv_codigo, codigo_barras_sku), group in data.groupby(['pdv_codigo', 'codigo_barras_sku']):
            # En entrenamiento, se usa la serie completa histórica.
            # En predicción, se utiliza la serie histórica (del train) y se le agregan N valores nulos al final.
            if is_train:
                target = group['cant_vta'].tolist()
            else:
                target = group['cant_vta'].tolist() + [np.nan] * self.prediction_length

            start = pd.to_datetime(group['fecha_comercial'].min())

            # Características dinámicas (deben tener la misma longitud que la serie histórica)
            dynamic_feat = [
                group['day_of_week'].tolist(),
                group['is_weekend'].tolist(),
                group['month'].tolist(),
                group['quarter'].tolist(),
                group['week_of_year'].tolist(),
                group['is_month_start'].tolist(),
                group['is_month_end'].tolist(),
                group['is_first_week'].tolist(),
                group['is_last_week'].tolist()
            ]
            # Características estáticas: p.ej., identificadores de tienda y producto
            static_cat = [pdv_codigo, codigo_barras_sku]

            series.append({
                "start": start,
                "target": target,
                "item_id": f"{pdv_codigo}_{codigo_barras_sku}",
                "feat_static_cat": static_cat,
                "feat_dynamic_real": dynamic_feat
            })

        return ListDataset(series, freq=self.freq)

    def train(self, train_data):
        train_ds = self.format_data(train_data, is_train=True)
        self.model = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.prediction_length * 4,
            num_cells=120,
            num_layers=4,
            dropout_rate=0.1,
            trainer=Trainer(
                epochs=50,            # Puedes ajustar la cantidad de épocas
                learning_rate=0.001,
                ctx=mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
            )
        ).train(training_data=train_ds)

    def predict(self, historical_data):
        # historical_data debe ser el conjunto con la serie histórica completa (train_df)
        test_ds = self.format_data(historical_data, is_train=False)
        predictor = self.model.predict(test_ds)

        predictions = []
        identifiers = []
        for entry in predictor:
            # Se calcula el promedio de las trayectorias muestreadas para obtener el pronóstico
            pred_series = entry.samples.mean(axis=0).tolist()
            predictions.append(pred_series)
            identifiers.append(entry.item_id)

        return predictions, identifiers
