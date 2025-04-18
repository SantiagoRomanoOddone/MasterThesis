import pandas as pd

def fixed_split(data: pd.DataFrame, start_test: pd.Timestamp= None, end_test: pd.Timestamp= None):
    if start_test is None:
        start_test = pd.Timestamp("2024-11-01")
    if end_test is None:
        end_test = pd.Timestamp("2024-11-30")
    data = data[data['fecha_comercial'] <= end_test]
    
    test = data[data['fecha_comercial'] >= start_test]
    train = data[data['fecha_comercial'] < start_test]

    return train, test