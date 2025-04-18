import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from functools import partial
import json

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.verbose = 0
        self.excluded_columns = ['fecha_comercial', 'codigo_barras_sku', 'nombre_sku', 'imp_vta',
                                'cant_vta', 'stock','pdv_codigo', 'cluster']
        self.return_columns = ['fecha_comercial', 'codigo_barras_sku', 'pdv_codigo', 'cant_vta', 'cant_vta_pred']
        self.kwargs = kwargs

    def fit(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]
        target = 'cant_vta'

        X = data[features]
        y = data[target]

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            **self.kwargs
        )
        self.model.fit(X, y)

    def predict(self, data):
        features = [col for col in data.columns if col not in self.excluded_columns]

        X = data[features]
        y_pred = self.model.predict(X).astype(int)
        y_pred[y_pred < 0] = 0

        y_pred_df = pd.DataFrame(y_pred, columns=['cant_vta_pred'])
        y_pred_df.index = data.index

        df_pred = pd.merge(data, y_pred_df, left_index=True, right_index=True)
        return df_pred[self.return_columns]

def sample_hyperparameters(trial, hyperparameter_space):
    hyperparams = {}
    for param_name, param_config in hyperparameter_space.items():
        param_type = param_config['type']
        if param_type == 'categorical':
            hyperparams[param_name] = trial.suggest_categorical(param_name, param_config['values'])
        elif param_type == 'int':
            hyperparams[param_name] = trial.suggest_int(
                param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
            )
        elif param_type == 'float':
            hyperparams[param_name] = trial.suggest_float(
                param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
            )
    return hyperparams

def get_xgboost_hyperparameter_space():
    return {
        "n_estimators": {
            "type": "int",
            "low": 50,
            "high": 1000,
            "log": True
        },
        "max_depth": {
            "type": "int",
            "low": 3,
            "high": 15
        },
        "learning_rate": {
            "type": "float",
            "low": 0.001,
            "high": 0.3,
            "log": True
        },
        "gamma": {
            "type": "float",
            "low": 0,
            "high": 5
        },
        "min_child_weight": {
            "type": "int",
            "low": 1,
            "high": 10
        },
        "subsample": {
            "type": "float",
            "low": 0.5,
            "high": 1.0
        },
        "colsample_bytree": {
            "type": "float",
            "low": 0.5,
            "high": 1.0
        }
    }

def xgboost_bayesian_search(train_df, val_df, n_trials=10):
    hyperparameter_space = get_xgboost_hyperparameter_space()
    
    def objective(trial):
        params = sample_hyperparameters(trial, hyperparameter_space)
        params.update({
            'objective': 'reg:squarederror',
            'random_state': 42,
            'early_stopping_rounds': 10
        })
        
        features = [col for col in train_df.columns if col not in XGBoostRegressor().excluded_columns]
        X_train = train_df[features]
        y_train = train_df['cant_vta']
        X_val = val_df[features]
        y_val = val_df['cant_vta']
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        preds[preds < 0] = 0
        return np.sqrt(mean_squared_error(y_val, preds))
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_trial