from models.catboost_model.catboost_main import catboost, catboost_by_product
from models.mean_sale.mean_sale_main import mean_sale
# from models.xgboost.xgboost_main import xgboost, xgboost_by_product
# from models.deep_learning.lstm.lstm_main import lstm
# from models.lightgbm.lightgbm_main import lightgbm, lightgbm_by_product
from train.splits.fixed_split import fixed_split

from models.deep_learning.deepar.deepar import deepar_main
from models.deep_learning.temporal_fusion_transformer.temporal_fusion_transformer import tft_main

# testing
from models.deep_learning.d_linear.d_linear import dlinear_main
from models.deep_learning.deep_npts.deep_npts import deepnpts_main
from models.deep_learning.patch_tst.patch_tst import patchtst_main
from models.deep_learning.simple_feedforward.simple_feedforward import sff_main
from models.deep_learning.wavenet.wavenet import wavenet_main



import matplotlib.pyplot as plt
from metrics.metrics import Metrics
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
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_lgbm_pdv_sku'], label='LGBM PDV SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_lgbm_sku'], label='LGBM SKU Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_deepar'], label='DeepAR Prediction')
        plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_mean_pdv_sku'], label='Mean PDV SKU Prediction')
        # plt.plot(subset['fecha_comercial'], subset['cant_vta_pred_lstm_pdv_sku'], label='LSTM PDV SKU Prediction')
        
        plt.xlabel('Fecha Comercial')
        plt.ylabel('Sales')
        plt.title(f'Actual vs Predicted Sales for PDV: {pdv_codigo}, SKU: {codigo_barras_sku}')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == '__main__':

    # Constants
    CLUSTER_NUMBER = 0
    FREQ = "D"
    PREDICTION_LENGTH = 30
    START_TRAIN = pd.Timestamp("2022-12-01")
    START_TEST = pd.Timestamp("2024-11-01")
    END_TEST = pd.Timestamp("2024-11-30")

    DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/features.parquet"

    features = pd.read_parquet(DATA_PATH)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 'cant_vta', 'cluster']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster"] == CLUSTER_NUMBER]

    filter = features['codigo_barras_sku'].unique()[:10]
    features = features[features['codigo_barras_sku'].isin(filter)]

    train_df, test_df = fixed_split(features)

    mean_sale_results = mean_sale(features)
    deepar_results = deepar_main(features)
    tft_results = tft_main(features)
    d_linear_results = dlinear_main(features)
    deep_npts_results = deepnpts_main(features)
    patch_tst_results = patchtst_main(features)
    simple_feedforward_results = sff_main(features)
    wavenet_results = wavenet_main(features)

    
    # Testing catboost models 
    # cb_results = catboost(features)
    # cb_results_sku = catboost_by_product(features)
    # xgb_results = xgboost(features)
    # xgb_results_sku = xgboost_by_product(features)
    # lgbm_results = lightgbm(features)
    # lgbm_results_sku = lightgbm_by_product(features)
    # lstm_results = lstm(features)

    # test_df = pd.merge(test_df, cb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, cb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, xgb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, xgb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, lgbm_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, lgbm_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, mean_sale_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, deepar_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, tft_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, d_linear_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, deep_npts_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, patch_tst_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, simple_feedforward_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    test_df = pd.merge(test_df, wavenet_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')


    summary_df = Metrics().create_summary_dataframe(test_df)
    # summary_df.to_csv('summary_custer_3_total.csv', index=False)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())


    # # Analysis 
    # plot_combinations(test_df, 10)



