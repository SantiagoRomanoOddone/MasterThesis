from models.catboost.main import analyze_catboost
from models.mean_sale.main import analyze_mean
from models.xgboost.main import analyze_xgboost
import numpy as np
import pandas as pd




if __name__ == '__main__':
    # Compare the results of the models for a specific cluster
    catboost_results = analyze_catboost(3)
    xgboost_results = analyze_xgboost(3)
    mean_results = analyze_mean(3)

    catboost_mean_mse = np.mean([result['mse'] for result in catboost_results])
    catboost_mean_rmse = np.mean([result['rmse'] for result in catboost_results])
    print(f"CatBoost Mean MSE: {catboost_mean_mse}")
    print(f"CatBoost Mean RMSE: {catboost_mean_rmse}")

    xgboost_mean_mse = np.mean([result['mse'] for result in xgboost_results])
    xgboost_mean_rmse = np.mean([result['rmse'] for result in xgboost_results])
    print(f"XGBoost Mean MSE: {xgboost_mean_mse}")
    print(f"XGBoost Mean RMSE: {xgboost_mean_rmse}")

    mean_mean_mse = np.mean([result['mse'] for result in mean_results])
    mean_mean_rmse = np.mean([result['rmse'] for result in mean_results])
    print(f"Mean Sale Mean MSE: {mean_mean_mse}")
    print(f"Mean Sale Mean RMSE: {mean_mean_rmse}")


    all_results = catboost_results + xgboost_results + mean_results

    best_models = {}

    # Iterate through all results and find the best model for each combination
    for result in all_results:
        key = (result['pdv_codigo'], result['codigo_barras_sku'])
        if key not in best_models or result['rmse'] < best_models[key]['rmse']:
            best_models[key] = {
                'pdv_codigo': result['pdv_codigo'],
                'codigo_barras_sku': result['codigo_barras_sku'],
                'mse': result['mse'],
                'rmse': result['rmse'],
                'best_model': result['model']
            }

    best_models_list = list(best_models.values())

    best_models_df = pd.DataFrame(best_models_list)
    print(best_models_df['best_model'].value_counts())



    # best_model cluster 1 
    # CatBoost Mean MSE: 5129869189.875176
    # CatBoost Mean RMSE: 3415.917703206119
    # XGBoost Mean MSE: 3584176817.65485
    # XGBoost Mean RMSE: 2611.38865951491
    # Mean Sale Mean MSE: 8874580619.746906
    # Mean Sale Mean RMSE: 10080.256686764185
    # XGBoost     5900
    # CatBoost    1154

    print("[END] Model Comparison")
