import pandas as pd
from models.deep_learning.deepar.deepar_model import DeepARModel
from train.splits.fixed_split import fixed_split
from metrics.metrics import Metrics
import numpy as np
# from sklearn.preprocessing import StandardScaler

def deepar(cluster_data):
    # Realiza el split: train_df contiene la serie histórica y test_df el horizonte real (para evaluar)
    train_df, test_df = fixed_split(cluster_data)

    # Entrena el modelo con la parte histórica
    model = DeepARModel()
    model.train(train_df)

    # Para la predicción se utiliza la serie histórica (train_df)
    predictions, identifiers = model.predict(train_df)

    # Se construye un DataFrame con los pronósticos utilizando las fechas del horizonte real (test_df)
    results = []
    for item_id, pred_series in zip(identifiers, predictions):
        pdv_codigo, codigo_barras_sku = item_id.split('_')
        
        # Se obtiene la primera fecha del horizonte de pronóstico para cada SKU (de test_df)
        sku_mask = test_df['codigo_barras_sku'] == int(codigo_barras_sku)
        start_date = test_df.loc[sku_mask, 'fecha_comercial'].min()
        
        if start_date is pd.NaT:
            continue
        # Se crea un rango de fechas con longitud igual al horizonte (prediction_length)
        dates = pd.date_range(start=start_date, periods=len(pred_series), freq='D')

        df_pred = pd.DataFrame({
            'fecha_comercial': dates,
            'codigo_barras_sku': int(codigo_barras_sku),
            'pdv_codigo': int(pdv_codigo),
            'cant_vta_pred_deepar': np.maximum(0, pred_series)
        })
        results.append(df_pred)

    final_results = pd.concat(results, ignore_index=True)
    return final_results

if __name__ == '__main__':
    # Load data
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]

    # Split data
    train_df, test_df = fixed_split(features)

    # Train and evaluate DeepAR model
    results = deepar(features)

    # Merge predictions with test data
    test_df = test_df.merge(results, on=['codigo_barras_sku', 'pdv_codigo', 'fecha_comercial'], how='left')

    # Evaluate metrics
    summary_df = Metrics().create_summary_dataframe(test_df)

    print(summary_df['best_rmse'].value_counts())
    print(summary_df['best_mse'].value_counts())


