from metrics.metrics import Metrics
from train.transformations.onehot_encoding_pdv import onehot_encoding_pdv
from models.xgboost.xgboost_model import XGBoostRegressor, xgboost_bayesian_search
from train.splits.fixed_split import fixed_split
import pandas as pd
import numpy as np
import time

def xgboost(cluster_data):
    print('[START] XGBoost model')
    combinations = cluster_data[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()
    results = []

    for _, row in combinations.iterrows():
        pdv_codigo = row['pdv_codigo']
        codigo_barras_sku = row['codigo_barras_sku']
        print(f"Processing pdv_codigo: {pdv_codigo}, codigo_barras_sku: {codigo_barras_sku}")

        data = cluster_data[(cluster_data['codigo_barras_sku'] == codigo_barras_sku) & 
                           (cluster_data['pdv_codigo'] == pdv_codigo)]
        
        train_df, test_df = fixed_split(data)
        if train_df.empty or test_df.empty:
            continue

        if train_df['cant_vta'].nunique() == 1:
            print(f"Skipping due to equal train targets")
            continue

        # Create validation set (last 30 days of train)
        val_start = train_df['fecha_comercial'].max() - pd.Timedelta(days=30)
        val_df = train_df[train_df['fecha_comercial'] >= val_start]
        train_df = train_df[train_df['fecha_comercial'] < val_start]

        # Hyperparameter tuning
        best_trial = xgboost_bayesian_search(train_df, val_df, n_trials=10)
        
        # Train final model with best params
        model = XGBoostRegressor(**best_trial.params)
        model.fit(pd.concat([train_df, val_df]))  # Combine train and val for final training
        
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
        
        train_df, test_df = fixed_split(data)
        if train_df.empty or test_df.empty:
            continue
    
        if train_df['cant_vta'].nunique() == 1:
            print(f"Skipping codigo_barras_sku: {codigo_barras_sku} due to equal train targets")
            continue

        # Create validation set (last 30 days of train)
        val_start = train_df['fecha_comercial'].max() - pd.Timedelta(days=30)
        val_df = train_df[train_df['fecha_comercial'] >= val_start]
        train_df = train_df[train_df['fecha_comercial'] < val_start]

        # Hyperparameter tuning
        best_trial = xgboost_bayesian_search(train_df, val_df)
        
        # Train final model with best params
        model = XGBoostRegressor(**best_trial.params)
        model.fit(pd.concat([train_df, val_df]))  # Combine train and val for final training
        
        # Make predictions
        result = model.predict(test_df)
        results.append(result)

    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_xgb_sku'}, inplace=True)
    print('[END] XGBoost model')
    return final_results

if __name__ == '__main__':
   
    # Constants
    CLUSTER_NUMBER = 0
    FREQ = "D"
    PREDICTION_LENGTH = 30
    MODEL = 'xgboost'

    DATA_PATH = "/Users/santiagoromano/Documents/code/MasterThesis/features/processed/cleaned_features.parquet"

    features = pd.read_parquet(DATA_PATH)
    features = features[['pdv_codigo', 'fecha_comercial', 'codigo_barras_sku', 'cant_vta',
                        'month','day', 'day_of_week', 'week_of_year', 'day_of_year',
                        'cluster_sku']]
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster_sku"] == CLUSTER_NUMBER]

    # Filter BY number of stores associated with the SKU
    filter = features.groupby('codigo_barras_sku').agg({'pdv_codigo': 'nunique'}).reset_index().sort_values('pdv_codigo', ascending=False).head(10)['codigo_barras_sku'].tolist()
    features = features[features['codigo_barras_sku'].isin(filter)]
    
    train_df, test_df = fixed_split(features)

    # Start timer
    start_time = time.time()
    
    # Testing catboost models 
    # results = xgboost(features)
    results_sku = xgboost_by_product(features)

    # End timer
    end_time = time.time()
    duration = end_time - start_time
    print(f"Function took {duration:.2f} seconds to run")

    test_df = test_df.merge(results_sku, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')
    # test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo','fecha_comercial','cant_vta'], how='left')

    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())
    summary_df.to_csv(f'results/metrics_cluster_{CLUSTER_NUMBER}_{MODEL}.csv', index=False)

