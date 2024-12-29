import pandas as pd

def onehot_encoding_pdv(df):
    
    pdv = df['pdv_codigo']
    df = pd.get_dummies(df, prefix='pdv_codigo', columns=['pdv_codigo'], drop_first=False)
    df['pdv_codigo'] = pdv

    return df