# from models.catboost_model.catboost_main import catboost, catboost_by_product
from models.mean_sale.mean_sale_main import mean_sale
# from models.xgboost.xgboost_main import xgboost, xgboost_by_product
# from models.deep_learning.lstm.lstm_main import lstm
# from models.lightgbm.lightgbm_main import lightgbm, lightgbm_by_product
from train.splits.fixed_split import fixed_split

# from models.deep_learning.deepar.deepar import deepar_main
# from models.deep_learning.temporal_fusion_transformer.temporal_fusion_transformer import tft_main

# testing
# from models.deep_learning.d_linear.d_linear import dlinear_main
# from models.deep_learning.deep_npts.deep_npts import deepnpts_main
# from models.deep_learning.patch_tst.patch_tst import patchtst_main
# from models.deep_learning.simple_feedforward.simple_feedforward import sff_main
# from models.deep_learning.wavenet.wavenet import wavenet_main
import numpy as np
import time
# import matplotlib.pyplot as plt
from metrics.metrics import Metrics
import pandas as pd


if __name__ == '__main__':

    np.random.seed(42) 

    # Constants
    CLUSTER_NUMBER = 2
    FREQ = "D"
    PREDICTION_LENGTH = 30
    START_TRAIN = pd.Timestamp("2022-12-01")
    START_TEST = pd.Timestamp("2024-11-01")
    END_TEST = pd.Timestamp("2024-11-30")
    MODEL = 'mean_sale'  # Change this to the model you want to run

    DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/cleaned_features.parquet"

    features = pd.read_parquet(DATA_PATH)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 'cant_vta', 'cluster_sku']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster_sku"] == CLUSTER_NUMBER]

    # # Randomly select 10 SKUs
    # random_skus = np.random.choice(features['codigo_barras_sku'].unique(), size=10, replace=False)
    # features = features[features['codigo_barras_sku'].isin(random_skus)]
    
    # # Filter BY number of stores associated with the SKU
    filter = features.groupby('codigo_barras_sku').agg({'pdv_codigo': 'nunique'}).reset_index().sort_values('pdv_codigo', ascending=False).head(10)['codigo_barras_sku'].tolist()
    features = features[features['codigo_barras_sku'].isin(filter)]

    # # TEST
    # filter = 7891991000727
    # features = features[features['codigo_barras_sku'].isin([filter])]

    train_df, test_df = fixed_split(features)

    # Start timer
    start_time = time.time()

    mean_sale_results = mean_sale(features)
    # deepar_results = deepar_main(features,CLUSTER_NUMBER)
    # tft_results = tft_main(features,CLUSTER_NUMBER)
    # d_linear_results = dlinear_main(features)
    # deep_npts_results = deepnpts_main(features)
    # patch_tst_results = patchtst_main(features)
    # simple_feedforward_results = sff_main(features,CLUSTER_NUMBER)
    # wavenet_results = wavenet_main(features,CLUSTER_NUMBER)

    
    # Testing catboost models 
    # cb_results = catboost(features)
    # cb_results_sku = catboost_by_product(features)
    # xgb_results = xgboost(features)
    # xgb_results_sku = xgboost_by_product(features)
    # lgbm_results = lightgbm(features)
    # lgbm_results_sku = lightgbm_by_product(features)
    # lstm_results = lstm(features)

    # End timer
    end_time = time.time()
    duration = end_time - start_time
    print(f"Function took {duration:.2f} seconds to run")

    # test_df = pd.merge(test_df, cb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, cb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, xgb_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, xgb_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, lgbm_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, lgbm_results_sku, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    test_df = pd.merge(test_df, mean_sale_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial','cant_vta'], how='left')
    # test_df = pd.merge(test_df, deepar_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, tft_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, d_linear_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, deep_npts_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, patch_tst_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, simple_feedforward_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')
    # test_df = pd.merge(test_df, wavenet_results, on=['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial'], how='left')


    # summary_df = Metrics().create_summary_dataframe(test_df)
    # summary_df.to_csv(f'results/metrics_cluster_{CLUSTER_NUMBER}_{MODEL}_test.csv', index=False)

    # print(summary_df['best_rmse'].value_counts())
    # print(summary_df['best_mse'].value_counts())
    print(test_df)
    test_df.to_csv(f'results/inference/predictions_cluster_{CLUSTER_NUMBER}_{MODEL}.csv', index=False)

    # # Analysis 
    # plot_combinations(test_df, 10)



