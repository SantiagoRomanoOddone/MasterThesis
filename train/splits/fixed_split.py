import pandas as pd

def fixed_split(split_date: str, data: pd.DataFrame):
    train_df = data[data['fecha_comercial'] < split_date]
    test_df = data[data['fecha_comercial'] >= split_date]

    return train_df, test_df