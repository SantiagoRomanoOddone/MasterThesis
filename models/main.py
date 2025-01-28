from models.catboost.catboost_main import catboost, catboost_by_product
from models.mean_sale.mean_sale_main import mean_sale
from models.xgboost.xgboost_main import xgboost, xgboost_by_product
from models.deep_learning.lstm.lstm_main import lstm
from train.splits.fixed_split import fixed_split
from train.transformations.onehot_encoding_pdv import onehot_encoding_pdv
import matplotlib.pyplot as plt
from metrics.metrics import Metrics
import numpy as np
import pandas as pd

def plot_combinations(data, num_combinations):
    # Get unique combinations of 'pdv_codigo' and 'codigo_barras_sku'
    combinations = data[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates().head(num_combinations)

    for _, row in combinations.iterrows():
        pdv_codigo = row['pdv_codigo']
        codigo_barras_sku = row['codigo_barras_sku']
        
        # Filter data for the current combination
        subset = data[(data['pdv_codigo'] == pdv_codigo) & (data['codigo_barras_sku'] == codigo_barras_sku)]
        
        # Ensure 'fecha_comercial' is in datetime format for better plotting
        subset['fecha_comercial'] = pd.to_datetime(subset['fecha_comercial'])
        
        plt.figure(figsize=(12, 6))
        
        # Plot each 'cant_vta' column
        plt.plot(subset['fecha_comercial'], subset['cant_vta'], label='Actual Sales')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_cb_pdv_sku'], label='CB PDV SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_cb_sku'], label='CB SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_xgb_pdv_sku'], label='XGB PDV SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_xgb_sku'], label='XGB SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_mean_pdv_sku'], label='Mean PDV SKU Prediction')
        # plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_lstm_pdv_sku'], label='LSTM PDV SKU Prediction')
        
        plt.xlabel('Fecha Comercial')
        plt.ylabel('Sales')
        plt.title(f'Actual vs Predicted Sales for PDV: {pdv_codigo}, SKU: {codigo_barras_sku}')
        plt.legend()
        plt.grid(True)
        plt.show()



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
    mean_sale_results = mean_sale(features)
    # lstm_results = lstm(features)

    test_df = pd.merge(test_df, cb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, cb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, xgb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, xgb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, mean_sale_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, lstm_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')

    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())


    # Analysis 
    plot_combinations(test_df, 10)



