import pandas as pd
import numpy as np
import xgboost as xgb
from metrics.metrics import Metrics
from sklearn.metrics import mean_squared_error as mse_sklearn

def analyze_xgboost(cluster_number):
    print('[START] XGBoost model')
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

        # Prepare the features and target variable
        features = ['imp_vta', 'stock', 'year', 'month', 'day', 'day_of_week',
                    'is_weekend', 'quarter', 'week_of_year', 'day_of_year', 'is_month_start', 'is_month_end', 'is_first_week',
                    'is_last_week', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30', 'lag_1', 'lag_7',
                    'lag_30', 'diff_1', 'diff_7', 'diff_30']
        target = 'cant_vta'

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Train an XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model using custom MSE and RMSE functions
        mse = Metrics.mean_squared_error(y_test, y_pred)
        rmse = Metrics.root_mean_squared_error(y_test, y_pred)

        # Store the results
        results.append({
            'pdv_codigo': pdv_codigo,
            'codigo_barras_sku': codigo_barras_sku,
            'mse': mse,
            'rmse': rmse,
            'model': 'XGBoost'
        })
    print('[END] XGBoost model')

    return results


if __name__ == '__main__':
    # Call the function to analyze cluster 3
    results = analyze_xgboost(3)

    # Calculate the mean value of MSE and RMSE from the results list
    mean_mse = np.mean([result['mse'] for result in results])
    mean_rmse = np.mean([result['rmse'] for result in results])

    # Print the results
    print(f"Mean MSE: {mean_mse}")
    print(f"Mean RMSE: {mean_rmse}")

