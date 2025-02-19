
import pandas as pd
import numpy as np
from models.deep_learning.rnn.rnn_model import RNNRegressor, PyTorchRNNRegressor
from train.splits.fixed_split import fixed_split

def rnn(cluster_data):
    print('[START] RNN model')

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
        

        # Train the RNN model
        model = PyTorchRNNRegressor(input_size=len(features.columns) - 8, epochs=10)
        model.fit(train_df)
        result = model.predict(test_df)

        results.append(result)

    final_results = pd.concat(results, ignore_index=True)
    final_results.rename(columns={'cant_vta_pred': 'cant_vta_pred_cb_pdv_sku'}, inplace=True)
    print('[END] CatBoost model')
    return final_results



if __name__ == '__main__':
    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]

    train_df, test_df = fixed_split(features)


    results = rnn(features)


