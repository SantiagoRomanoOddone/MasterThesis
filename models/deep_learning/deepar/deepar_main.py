import pandas as pd
from models.deep_learning.deepar.deepar_model import DeepARModel
from train.splits.fixed_split import fixed_split
from metrics.metrics import Metrics

def deepar(cluster_data):
    print('[START] DeepAR model')
    
    model = DeepARModel()
    model.train(cluster_data)
    predictions = model.predict(cluster_data)
    
    results = []
    for (pdv_codigo, codigo_barras_sku), prediction in zip(cluster_data.groupby(['pdv_codigo', 'codigo_barras_sku']), predictions):
        df_pred = prediction.to_dataframe()
        df_pred['pdv_codigo'] = pdv_codigo
        df_pred['codigo_barras_sku'] = codigo_barras_sku
        results.append(df_pred)
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'prediction': 'cant_vta_pred_deepar'}, inplace=True)
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