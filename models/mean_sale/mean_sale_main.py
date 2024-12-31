import pandas as pd
import numpy as np
from train.splits.fixed_split import fixed_split
from metrics.metrics import Metrics

def mean_sale(cluster_data):
    print('[START] Mean Sale model')
    # Get unique combinations of pdv_codigo and codigo_barras_sku
    combinations = cluster_data[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    # List to store the results
    results = []

    # Iterate over each combination of pdv_codigo and codigo_barras_sku
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
        
        train_df['rolling_mean_30'] = train_df['cant_vta'].rolling(window=30, min_periods=1).mean()

        last_rolling_mean = train_df['rolling_mean_30'].iloc[-1]
        test_df['cant_vta_pred'] = last_rolling_mean

        return_columns = ['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']
        result = test_df[return_columns]
        results.append(result)

    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_mean_pdv_sku'}, inplace=True)

    print('[END] Mean Sale model')
    return final_results

if __name__ == '__main__':

    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

    features = features[features['cluster'] == cluster_number]
    
    train_df, test_df = fixed_split(features)
    
    # Testing catboost models 
    results = mean_sale(features)

    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')

    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())