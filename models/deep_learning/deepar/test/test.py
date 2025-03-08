import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from gluonts.torch import DeepAREstimator
from gluonts.mx.trainer import Trainer
import numpy as np

from gluonts.evaluation import Evaluator
from tqdm.autonotebook import tqdm
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False



if __name__ == '__main__':
    cluster_number = 3
    features = pd.read_parquet("/content/features.parquet")
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster"] == cluster_number]
    sku = 7894900027013


    filtered = features[
        (features["codigo_barras_sku"] == sku)
    ].copy()
    filtered = filtered[filtered['pdv_codigo']==1]
    filtered = filtered[filtered['fecha_comercial'] <= '2024-11-30']
    validation = filtered[filtered['fecha_comercial']>= '2024-11-01']
    filtered = filtered[filtered['fecha_comercial'] < '2024-11-01']
    filtered['fecha_comercial'].max(), filtered['fecha_comercial'].min()
    
    
    df = filtered.pivot(
        index="fecha_comercial",
        columns="pdv_codigo",
        values="cant_vta"
    )

    df.columns = [f"pdv_codigo_{col}" for col in df.columns]
    df_input = df.reset_index().rename(columns={"fecha_comercial": "date"}) 


    ts_code = df_input.columns[1:].astype('category').codes  


    df_values = df_input.iloc[:, 1:].astype(float)


    df_train = df_values.iloc[:-31, :].values  
    df_test = df_values.iloc[:, :].values  


    freq = "D"
    start_train = df_input["date"].iloc[0]  
    start_test = df_input["date"].iloc[-31]  


    mean_value = df_train.mean()
    std_value = df_train.std()
    df_train = (df_train - mean_value) / std_value
    df_test = (df_test - mean_value) / std_value  


    prediction_length = 31
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_layers=2,
        hidden_size=32,
        lr=0.001,
        weight_decay=1e-08,
        dropout_rate=0.1,
        patience=10,
        num_feat_dynamic_real=0,
        num_feat_static_cat=1,
        num_feat_static_real=0,
        cardinality=[len(np.unique(ts_code))],
        embedding_dimension=None,
        scaling=False,  
        default_scale=None,
        lags_seq=None,
        time_features=None,
        num_parallel_samples=100,
        batch_size=32,
        num_batches_per_epoch=50,
        imputation_method=None,
        trainer_kwargs={"max_epochs": 5},
        train_sampler=None,
        validation_sampler=None,
        nonnegative_pred_samples=False,
    )


    train_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_train,
            FieldName.FEAT_STATIC_CAT: [fsc]  
        }
        for target, fsc in zip(df_train.T, ts_code)
    ], freq=freq)

    test_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_test,  
            FieldName.FEAT_STATIC_CAT: [fsc]  
        }
        for target, fsc in zip(df_test.T, ts_code)
    ], freq=freq)


    predictor = estimator.train(training_data=train_ds)


    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )


    tss = list(tqdm(ts_it, total=len(df_test.T)))


    forecasts = list(tqdm(forecast_it, total=len(df_test.T)))


    real_values = (np.array([tss[i].iloc[-31:].values.flatten() for i in range(len(tss))]) * std_value) + mean_value


    predictions = np.array([forecasts[i].mean for i in range(len(forecasts))]) * std_value + mean_value


    results = pd.DataFrame({
        'date': pd.date_range(start=start_test, periods=prediction_length, freq=freq),
        'real_value': real_values.mean(axis=0),  
        'prediction': predictions.mean(axis=0)  
    })


    results