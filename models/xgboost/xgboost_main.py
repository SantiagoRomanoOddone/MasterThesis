from metrics.metrics import Metrics
from train.transformations.onehot_encoding_pdv import onehot_encoding_pdv
from models.xgboost.xgboost_model import XGBoostRegressor
from train.splits.fixed_split import fixed_split
import pandas as pd
import numpy as np



def xgboost(cluster_data):
    print('[START] XGBoost model')
    
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

        # Train an XGBoost model
        model = XGBoostRegressor()
        model.fit(train_df)
        # Make predictions
        result = model.predict(test_df)

        results.append(result)

    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_xgb_pdv_sku'}, inplace=True)
    print('[END] XGBoost model')
    return final_results

def xgboost_by_product(cluster_data):
    print('[START] XGBoost model by product')
    
    products = cluster_data['codigo_barras_sku'].unique()
    results = []

    for codigo_barras_sku in products:
        print(f"Processing codigo_barras_sku: {codigo_barras_sku}")

        data = cluster_data[cluster_data['codigo_barras_sku'] == codigo_barras_sku]
        data = onehot_encoding_pdv(data)
        # Split
        train_df, test_df = fixed_split(data)
        if train_df.empty or test_df.empty:
            continue
    
        if train_df['cant_vta'].nunique() == 1:
            print(f"Skipping codigo_barras_sku: {codigo_barras_sku} due to equal train targets")
            continue

        # Train an XGBoost model
        model = XGBoostRegressor()
        model.fit(train_df)
        # Make predictions
        result = model.predict(test_df)

        results.append(result)

    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_xgb_sku'}, inplace=True)
    print('[END] XGBoost model')
    return final_results

if __name__ == '__main__':
   
    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

    features = features[features['cluster'] == cluster_number]
    
    train_df, test_df = fixed_split(features)
    
    # Testing catboost models 
    results = xgboost(features)
    results_sku = xgboost_by_product(features)

    test_df = test_df.merge(results_sku, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')
    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')

    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())
