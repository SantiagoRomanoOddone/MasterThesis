import pandas as pd
import numpy as np
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from sklearn.preprocessing import LabelEncoder
import mxnet as mx

class DeepARModel:
    def __init__(self, freq='D', prediction_length=31):
        self.freq = freq
        self.prediction_length = prediction_length
        self.model = None
        self.pdv_encoder = None
        self.sku_encoder = None

    def format_data(self, historical_data, forecast_data=None, pdv_encoder=None, sku_encoder=None):
        """
        Formats data into a ListDataset for GluonTS.
        If forecast_data is provided, the historical series is extended by:
          - Padding the target with np.nan for the forecast horizon.
          - Concatenating the dynamic features for the forecast period.
        """
        dynamic_cols = [
            'day_of_week', 'is_weekend', 'month', 'quarter',
            'week_of_year', 'is_month_start', 'is_month_end',
            'is_first_week', 'is_last_week'
        ]
        
        series = []
        # Obtain all unique (pdv_codigo, codigo_barras_sku) combinations.
        items = historical_data.groupby(['pdv_codigo', 'codigo_barras_sku']).groups.keys()
        
        for pdv_codigo, codigo_barras_sku in items:
            # Extract historical data for the item and sort by date.
            hist_group = historical_data[
                (historical_data['pdv_codigo'] == pdv_codigo) &
                (historical_data['codigo_barras_sku'] == codigo_barras_sku)
            ].sort_values('fecha_comercial')
            
            target = hist_group['cant_vta'].tolist()
            dyn_features = {col: hist_group[col].tolist() for col in dynamic_cols}
            
            # If forecast_data is provided, extend the series with forecast information.
            if forecast_data is not None:
                forecast_group = forecast_data[
                    (forecast_data['pdv_codigo'] == pdv_codigo) &
                    (forecast_data['codigo_barras_sku'] == codigo_barras_sku)
                ].sort_values('fecha_comercial')
                
                # Skip the series if no future data is available.
                if forecast_group.empty:
                    continue

                # Ensure the forecast horizon exactly equals prediction_length.
                forecast_horizon = self.prediction_length
                forecast_group = forecast_group.head(forecast_horizon)
                
                # Extend the target with np.nan values for the forecast period.
                target += [np.nan] * forecast_horizon
                
                # Extend each dynamic feature with the future values.
                for col in dynamic_cols:
                    dyn_features[col].extend(forecast_group[col].tolist())
            
            # Build dynamic features in the expected format (list of lists: one per feature).
            dynamic_feat = [dyn_features[col] for col in dynamic_cols]
            start = pd.to_datetime(hist_group['fecha_comercial'].min())
            
            # Encode static categorical features if encoders are provided.
            if pdv_encoder is not None and sku_encoder is not None:
                static_cat = [
                    int(pdv_encoder.transform([pdv_codigo])[0]),
                    int(sku_encoder.transform([codigo_barras_sku])[0])
                ]
            else:
                static_cat = [pdv_codigo, codigo_barras_sku]
            
            series.append({
                "start": start,
                "target": target,
                "item_id": f"{pdv_codigo}_{codigo_barras_sku}",
                "feat_static_cat": static_cat,
                "feat_dynamic_real": dynamic_feat
            })
        
        return ListDataset(series, freq=self.freq)

    def train(self, train_data, forecast_data=None):
        # Create encoders for the static features based on the training data.
        self.pdv_encoder = LabelEncoder().fit(train_data['pdv_codigo'].unique())
        self.sku_encoder = LabelEncoder().fit(train_data['codigo_barras_sku'].unique())
        
        # Format the training dataset, including future dynamic features if provided.
        train_ds = self.format_data(train_data, forecast_data=forecast_data,
                                    pdv_encoder=self.pdv_encoder, sku_encoder=self.sku_encoder)
        
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
        Produces forecasts using historical data extended with forecast information.
        The target is extended (historical data + np.nan) and dynamic features are concatenated.
        """
        forecast_ds = self.format_data(historical_data, forecast_data,
                                       pdv_encoder=self.pdv_encoder, sku_encoder=self.sku_encoder)
        predictor = self.model.predict(forecast_ds)
        predictions = []
        identifiers = []
        for entry in predictor:
            # Compute the mean forecast from the sample trajectories.
            pred_series = entry.samples.mean(axis=0).tolist()
            predictions.append(pred_series)
            identifiers.append(entry.item_id)
        return predictions, identifiers
