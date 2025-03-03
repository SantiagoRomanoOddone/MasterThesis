import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
np.bool = np.bool_
from tqdm.autonotebook import tqdm

from gluonts.torch import DeepAREstimator
from gluonts.mx.trainer import Trainer

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions


mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False





if __name__ == '__main__':
    cluster_number = 3
    features = pd.read_parquet('features/processed/features.parquet')
    features = features.sort_values(["pdv_codigo", "codigo_barras_sku", "fecha_comercial"]).reset_index(drop=True)
    features = features[features["cluster"] == cluster_number]

    all_skus = features["codigo_barras_sku"].unique()
    all_skus = all_skus[:1]

    for sku in all_skus:
        print(f"SKU: {sku}")
        filtered = features[
            (features["codigo_barras_sku"] == sku)
        ].copy()

        df = filtered.pivot(
        index="fecha_comercial",  
        columns="pdv_codigo",    
        values="cant_vta"        
        )
        df.columns = [f"pdv_codigo_{col}" for col in df.columns]

        df_input=df.reset_index(drop=True).T.reset_index()

        ts_code=df_input["index"].astype('category').cat.codes.values

        df_train=df_input.iloc[:,1:707].values
        df_test=df_input.iloc[:,707:].values

        freq = "D"  
        start_train = pd.Timestamp("2022-12-01")  
        start_test = pd.Timestamp("2014-11-10")
        prediction_length = 30 

        estimator = DeepAREstimator(
        freq="D",  # Frequency of the time series (e.g., "D" for daily)
        prediction_length=30,  # Prediction length
        context_length=672,  # Context length (optional, defaults to prediction_length)
        num_layers=2,  # Number of RNN layers
        hidden_size=32,  # Number of hidden units in each RNN layer
        lr=0.001,  # Learning rate
        weight_decay=1e-08,  # Weight decay for regularization
        dropout_rate=0.1,  # Dropout rate for regularization
        patience=10,  # Patience for early stopping
        num_feat_dynamic_real=0,  # Number of dynamic real features
        num_feat_static_cat=1,  # Number of static categorical features
        num_feat_static_real=0,  # Number of static real features
        cardinality=[len(np.unique(ts_code))],  # Number of unique categories
        embedding_dimension=None,  # Embedding dimension for categorical features
        scaling=True,  # Whether to scale the data
        default_scale=None,  # Default scale for scaling
        lags_seq=None,  # Custom lag sequence (optional)
        time_features=None,  # Custom time features (optional)
        num_parallel_samples=100,  # Number of parallel samples for prediction
        batch_size=32,  # Batch size for training
        num_batches_per_epoch=50,  # Number of batches per epoch
        imputation_method=None,  # Method for imputing missing values
        trainer_kwargs={"max_epochs": 5},  # Trainer configuration
        train_sampler=None,  # Custom train sampler (optional)
        validation_sampler=None,  # Custom validation sampler (optional)
        nonnegative_pred_samples=False,  # Whether to enforce non-negative predictions
        )



        train_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start_train,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for (target, fsc) in zip(df_train,
                                    ts_code.reshape(-1,1))
        ], freq=freq)

        test_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start_test,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for (target, fsc) in zip(df_test,
                                    ts_code.reshape(-1,1))
        ], freq=freq)
        
        predictor = estimator.train(training_data=train_ds)


        forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  
        predictor=predictor,  
        num_samples=100, 
        )
        print("Obtaining time series conditioning values ...")
        tss = list(tqdm(ts_it, total=len(df_test)))
        print("Obtaining time series predictions ...")
        forecasts = list(tqdm(forecast_it, total=len(df_test)))

        
        # Initialize an empty list to store all results
        all_results = []

        # Iterate over forecasts, ground truth time series, and pdv_codigo
        for forecast, ts, pdv_codigo in zip(forecasts, tss, df_input['index']):
            # Extract the SKU from the loop context
            sku = sku  # Defined in the outer loop
            
            # Extract prediction dates
            prediction_dates = pd.date_range(
                start=forecast.start_date.to_timestamp(),
                periods=prediction_length,
                freq=freq
            )
            
            # Extract median predictions
            pred_values = np.median(forecast.samples, axis=0)  # Median prediction
            
            # Store predictions in the results list
            for date, pred in zip(prediction_dates, pred_values):
                all_results.append({
                    "pdv_codigo": pdv_codigo,  # Add pdv_codigo from df_input["index"]
                    "codigo_barras_sku": sku,
                    "fecha_comercial": date,
                    "pred_deepar": pred
                })

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(all_results)

    print(results_df.head())

       
