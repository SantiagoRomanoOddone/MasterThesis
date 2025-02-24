import pandas as pd
import numpy as np
from models.deep_learning.deepar.deepar_model import DeepARModel
from train.splits.fixed_split import fixed_split
from metrics.metrics import Metrics

def deepar(cluster_data):
    # Split data: train_df contains historical series and test_df contains the forecast horizon (for evaluation)
    train_df, test_df = fixed_split(cluster_data)

    # Train the model using historical data along with forecast dynamic features.
    # (Passing test_df so that known future covariates are used during training.)
    model = DeepARModel()
    model.train(train_df, forecast_data=test_df)

    # For prediction, use forecast information (test_df) to get future dynamic features
    predictions, identifiers = model.predict(train_df, test_df)

    # Construct a DataFrame with forecasts using the dates from the forecast horizon.
    results = []
    for item_id, pred_series in zip(identifiers, predictions):
        pdv_codigo, codigo_barras_sku = item_id.split('_')
        
        # Get the first date of the forecast horizon for each SKU (from test_df)
        sku_mask = test_df['codigo_barras_sku'] == int(codigo_barras_sku)
        start_date = test_df.loc[sku_mask, 'fecha_comercial'].min()
        if pd.isna(start_date):
            continue
        
        dates = pd.date_range(start=start_date, periods=len(pred_series), freq='D')
        df_pred = pd.DataFrame({
            'fecha_comercial': dates,
            'codigo_barras_sku': int(codigo_barras_sku),
            'pdv_codigo': int(pdv_codigo),
            'cant_vta_pred_deepar': np.maximum(0, pred_series)
        })
        results.append(df_pred)

    final_results = pd.concat(results, ignore_index=True)
    return final_results


if __name__ == '__main__':
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet')
    features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]

    train_df, test_df = fixed_split(features)

    results = deepar(features)

    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo', 'fecha_comercial'], how='left')
    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())
