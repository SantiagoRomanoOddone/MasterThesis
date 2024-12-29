import pandas as pd
import numpy as np
from metrics.metrics import Metrics

def analyze_mean(cluster_number):
    print('[START] Mean Sale model')
    # Load the data from the Parquet file
    features = pd.read_parquet('features/processed/features.parquet')
    features = features.sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)

    # Filter the specific cluster data
    cluster_data = features[features['cluster'] == cluster_number]

    # Get unique combinations of pdv_codigo and codigo_barras_sku
    combinations = cluster_data[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    # List to store the results
    results = []

    # Iterate over each combination of pdv_codigo and codigo_barras_sku
    for _, row in combinations.iterrows():
        pdv_codigo = row['pdv_codigo']
        codigo_barras_sku = row['codigo_barras_sku']
        print(f"Processing pdv_codigo: {pdv_codigo}, codigo_barras_sku: {codigo_barras_sku}")

        # Filter the data for the current combination
        data = cluster_data[(cluster_data['codigo_barras_sku'] == codigo_barras_sku) & (cluster_data['pdv_codigo'] == pdv_codigo)]

        # Split the data into training and testing sets
        split_date = '2024-11-10'
        train_df = data[data['fecha_comercial'] < split_date]
        test_df = data[data['fecha_comercial'] >= split_date]

        if train_df.empty or test_df.empty:
            continue

        # Prepare the target variable
        target = 'cant_vta'
        # Calculate the rolling mean using the latest 30 days of the training data
        train_df['rolling_mean_30'] = train_df[target].rolling(window=30, min_periods=1).mean()

        # Use the last value of the rolling mean from the training data as the prediction for the test set
        last_rolling_mean = train_df['rolling_mean_30'].iloc[-1]
        test_df['rolling_mean_30'] = last_rolling_mean

        y_pred_rolling_mean = test_df['rolling_mean_30']
        y_actual = test_df[target]

        # Evaluate the model using custom MSE and RMSE functions
        mse = Metrics.mean_squared_error(y_actual, y_pred_rolling_mean)
        rmse = Metrics.root_mean_squared_error(y_actual, y_pred_rolling_mean)

        # Store the results
        results.append({
            'pdv_codigo': pdv_codigo,
            'codigo_barras_sku': codigo_barras_sku,
            'mse': mse,
            'rmse': rmse,
            'model': 'CatBoost'
        })

    print('[END] Mean Sale model')
    return results

if __name__ == '__main__':
    # Call the function to analyze cluster 3
    results = analyze_mean(3)

    # Calculate the mean value of MSE and RMSE from the results list
    mean_mse = np.mean([result['mse'] for result in results])
    mean_rmse = np.mean([result['rmse'] for result in results])

    # Print the results
    print(f"Mean MSE: {mean_mse}") # 3810505572662.3716
    print(f"Mean RMSE: {mean_rmse}") # 842372.9273262176
