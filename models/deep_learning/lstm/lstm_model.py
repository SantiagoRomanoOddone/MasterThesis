import pandas as pd 
import torch
from copy import deepcopy as dc
from train.splits.fixed_split import fixed_split


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('fecha_comercial', inplace=True)

    for i in range(1, n_steps+1):
        df[f'cant_vta(t-{i})'] = df['cant_vta'].shift(i)

    df.dropna(inplace=True)

    return df




if __name__ == '__main__':

    cluster_number = 3

    features = pd.read_parquet('features/processed/features.parquet').sort_values(['pdv_codigo', 'codigo_barras_sku', 'fecha_comercial']).reset_index(drop=True)
    features = features[features['cluster'] == cluster_number]


    combinations = features[['pdv_codigo', 'codigo_barras_sku']].drop_duplicates()

    pdv_codigo = 1
    codigo_barras_sku = 7894900027013


    data = features[(features['codigo_barras_sku'] == codigo_barras_sku) & (features['pdv_codigo'] == pdv_codigo)]

    data = data[['fecha_comercial', 'cant_vta']]

    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    shifted_df

    # Split
    train_df, test_df = fixed_split(data)

