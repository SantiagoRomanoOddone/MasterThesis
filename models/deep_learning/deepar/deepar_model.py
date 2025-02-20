import pandas as pd
import numpy as np
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
import mxnet as mx

class DeepARModel:
    def __init__(self, freq='D', prediction_length=30):
        self.freq = freq
        self.prediction_length = prediction_length
        self.model = None

    def format_data(self, data):
        series = []
        for (pdv_codigo, codigo_barras_sku), group in data.groupby(['pdv_codigo', 'codigo_barras_sku']):
            target = group['cant_vta'].tolist()
            start = pd.to_datetime(group['fecha_comercial'].min())
            series.append({
                "start": start,
                "target": target,
                "item_id": f"{pdv_codigo}_{codigo_barras_sku}"
            })
        return ListDataset(series, freq=self.freq)

    def train(self, train_data):
        train_ds = self.format_data(train_data)
        self.model = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer=Trainer(epochs=5, ctx=mx.cpu())
        ).train(training_data=train_ds)

    def predict(self, test_data):
        test_ds = self.format_data(test_data)
        predictor = self.model.predict(test_ds)

        predictions = []
        identifiers = []
        for entry in predictor:
            pred_series = entry.mean.tolist()
            predictions.append(pred_series)
            identifiers.append(entry.item_id)

        return predictions, identifiers  # Return predictions + item IDs for mapping
