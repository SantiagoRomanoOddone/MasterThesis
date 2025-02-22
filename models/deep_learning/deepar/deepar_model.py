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

    def format_data(self, historical_data, forecast_data=None):
        """
        Formats data into a ListDataset for GluonTS.
        If forecast_data is None, it is training mode and uses the full historical series.
        If forecast_data is provided, the historical series is extended with the future horizon:
          - The target is extended with [NaN] for the forecast period.
          - Dynamic features are concatenated with their corresponding future values.
        """
        dynamic_cols = [
            'day_of_week', 'is_weekend', 'month', 'quarter',
            'week_of_year', 'is_month_start', 'is_month_end',
            'is_first_week', 'is_last_week'
        ]
        
        series = []
        # Get all items present in the historical dataset
        items = historical_data.groupby(['pdv_codigo', 'codigo_barras_sku']).groups.keys()
        
        for pdv_codigo, codigo_barras_sku in items:
            # Extract historical data for the item and sort by date
            hist_group = historical_data[
                (historical_data['pdv_codigo'] == pdv_codigo) &
                (historical_data['codigo_barras_sku'] == codigo_barras_sku)
            ].sort_values('fecha_comercial')
            
            target = hist_group['cant_vta'].tolist()
            dyn_features = {col: hist_group[col].tolist() for col in dynamic_cols}
            
            # If forecast_data is provided, extend the series with forecast information
            if forecast_data is not None:
                forecast_group = forecast_data[
                    (forecast_data['pdv_codigo'] == pdv_codigo) &
                    (forecast_data['codigo_barras_sku'] == codigo_barras_sku)
                ].sort_values('fecha_comercial')
                
                # Skip the series if no future data is available
                if forecast_group.empty:
                    continue

                # Extend the target with NaNs for the future horizon
                target += [np.nan] * len(forecast_group)
                
                # Extend each dynamic feature with the future values
                for col in dynamic_cols:
                    dyn_features[col].extend(forecast_group[col].tolist())
            
            # Convert dynamic features into a list of lists (maintaining order)
            dynamic_feat = [dyn_features[col] for col in dynamic_cols]
            
            start = pd.to_datetime(hist_group['fecha_comercial'].min())
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
        train_ds = self.format_data(train_data)
        self.model = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.prediction_length * 4,
            num_cells=120,
            num_layers=4,
            dropout_rate=0.1,
            trainer=Trainer(
                epochs=50,           
                learning_rate=0.001,
                ctx=mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
            )
        ).train(training_data=train_ds)

    def predict(self, historical_data, forecast_data):
        """
        Predicts by using the historical data and extending it with forecast_data.
        The target is extended (historical + NaNs) and dynamic features are concatenated.
        """
        forecast_ds = self.format_data(historical_data, forecast_data)
        predictor = self.model.predict(forecast_ds)
        predictions = []
        identifiers = []
        for entry in predictor:
            # Compute the mean forecast from the sample trajectories
            pred_series = entry.samples.mean(axis=0).tolist()
            predictions.append(pred_series)
            identifiers.append(entry.item_id)
        return predictions, identifiers
