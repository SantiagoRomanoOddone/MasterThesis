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

    def format_data(self, data, is_train=True):
        series = []
        for (pdv_codigo, codigo_barras_sku), group in data.groupby(['pdv_codigo', 'codigo_barras_sku']):

            target = group['cant_vta'].tolist() if is_train else None   
            
            start = pd.to_datetime(group['fecha_comercial'].min())

            # Dynamic features (date-based)
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

            # Static features (store ID, product ID)
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
            context_length=self.prediction_length * 2,  # Lookback window
            num_cells=120,  # Increase model complexity
            num_layers=4,   # Deeper network
            dropout_rate=0.1,
            trainer=Trainer(epochs=50, learning_rate=0.001, ctx=mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu())
        ).train(training_data=train_ds)

    def predict(self, test_data):
        test_ds = self.format_data(test_data, is_train=False)
        predictor = self.model.predict(test_ds)

        predictions = []
        identifiers = []
        for entry in predictor:
            pred_series = entry.mean.tolist()
            predictions.append(pred_series)
            identifiers.append(entry.item_id)

        return predictions, identifiers  # Return predictions + item IDs for mapping
