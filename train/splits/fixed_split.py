import pandas as pd

def fixed_split(data: pd.DataFrame):
    split_date = '2024-11-10'
    train_df = data[data['fecha_comercial'] < split_date]
    test_df = data[data['fecha_comercial'] >= split_date]

    return train_df, test_df