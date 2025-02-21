import pandas as pd
from models.deep_learning.deepar.deepar_model import DeepARModel
from train.splits.fixed_split import fixed_split
from metrics.metrics import Metrics
import numpy as np

def deepar(cluster_data):
    print('[START] DeepAR model')

    cluster_data['cant_vta'] = np.log1p(cluster_data['cant_vta'])

    # Split data
    train_df, test_df = fixed_split(cluster_data)
    
    model = DeepARModel()
    model.train(train_df)

    # Get predictions and corresponding item IDs
    predictions, identifiers = model.predict(test_df)

    # Convert predictions into a structured DataFrame
    results = []
    for item_id, pred_series in zip(identifiers, predictions):
        pdv_codigo, codigo_barras_sku = item_id.split('_')  # Extract identifiers
        
        # Get the first date from test_df for this SKU
        start_date = test_df[test_df['codigo_barras_sku'] == int(codigo_barras_sku)]['fecha_comercial'].min()
        
        # Generate date range to ensure correct length
        dates = pd.date_range(start=start_date, periods=len(pred_series), freq='D')

        df_pred = pd.DataFrame({
            'fecha_comercial': dates,
            'codigo_barras_sku': int(codigo_barras_sku),
            'pdv_codigo': int(pdv_codigo),
            'cant_vta_pred_deepar': np.maximum(0, pred_series)  # Avoid negatives
        })
        results.append(df_pred)

    final_results = pd.concat(results, ignore_index=True)
    final_results['cant_vta_pred_deepar'] = np.expm1(final_results['cant_vta_pred_deepar'])

    print('[END] DeepAR model')
    return final_results

if __name__ == '__main__':
    # Load data
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]

    # Split data
    train_df, test_df = fixed_split(features)

    # Train and evaluate DeepAR model
    results = deepar(features)

    # Merge predictions with test data
    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo', 'fecha_comercial'], how='left')

    # Evaluate metrics
    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())
