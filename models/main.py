from models.catboost.main import catboost, catboost_by_product
# from models.mean_sale.main import analyze_mean
from models.xgboost.main import xgboost, xgboost_by_product
from train.splits.fixed_split import fixed_split
from train.transformations.onehot_encoding_pdv import onehot_encoding_pdv
from metrics.metrics import Metrics
import numpy as np
import pandas as pd




if __name__ == '__main__':

    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

    features = features[features['cluster'] == cluster_number]
    
    train_df, test_df = fixed_split(features)
    
    # Testing catboost models 
    cb_results = catboost(features)
    cb_results_sku = catboost_by_product(features)
    xgb_results = xgboost(features)
    xgb_results_sku = xgboost_by_product(features)

    test_df = pd.merge(test_df, cb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, cb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, xgb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, xgb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')

    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())



