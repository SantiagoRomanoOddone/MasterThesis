import pandas as pd

from train.splits.fixed_split import fixed_split
from models.deep_learning.lstm.lstm_model import lstm_model
from metrics.metrics import Metrics

def lstm(cluster_data):
    print('[START] LSTM model')

    combinations = cluster_data[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    results = []

    for _, row in combinations.iterrows():
        pdv_codigo = row['pdv_codigo']
        codigo_barras_sku = row['codigo_barras_sku']
        print(f"Processing pdv_codigo: {pdv_codigo}, codigo_barras_sku: {codigo_barras_sku}")

        data = cluster_data[(cluster_data['codigo_barras_sku'] == codigo_barras_sku) & (cluster_data['pdv_codigo'] == pdv_codigo)]
        
        # Split
        train_df, test_df = fixed_split(data)
        if train_df.empty or test_df.empty:
            continue
    
        if train_df['cant_vta'].nunique() == 1:
            print(f"Skipping due to equal train targets")
            continue
        
        # TODO: Clean this up
        result = lstm_model(data)
        results.append(result)
    
    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_lstm_pdv_sku'}, inplace=True)
    print('[END] LSTM model')
    return final_results


if __name__ == '__main__':
    
    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

    features = features[features['cluster'] == cluster_number]
    
    train_df, test_df = fixed_split(features)
    
    # Testing LSTM model
    results = lstm(features)

    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')
    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())