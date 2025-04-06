import random
import pandas as pd
import numpy as np
from models.deep_learning.gluonts.functions import make_predictions, set_random_seed
import optuna
from optuna.samplers import TPESampler
from gluonts.evaluation import Evaluator
from functools import partial                                                   
from lightning.pytorch.callbacks import EarlyStopping, Callback
import json
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


class LossHistoryLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics.get('train_loss').item() 
                                 if trainer.callback_metrics.get('train_loss') is not None else None)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics.get('val_loss').item() 
                               if trainer.callback_metrics.get('val_loss') is not None else None)




def general_random_search(train_ds, val_ds, prediction_length,
                         model_class, hyperparameter_space, n_trials, fixed_params=None):
    """
    General random search for hyperparameter tuning
    
    Parameters:
    -----------
    train_ds : Training dataset
    val_ds : Validation dataset
    prediction_length : Forecast horizon
    model_class : The model estimator class (e.g. DeepAREstimator)
    hyperparameter_space : Dict of hyperparameter search spaces with types
    n_trials : Number of random trials
    fixed_params : Dict of fixed parameters for the model (optional)
    """
    set_random_seed(42)

    if fixed_params is None:
        fixed_params = {}
    
    def sample_parameter(param_config):
        """Helper function to sample a parameter based on its type"""
        if param_config['type'] == 'categorical':
            return random.choice(param_config['values'])
        elif param_config['type'] == 'float':
            if param_config.get('log', False):
                log_low = np.log(param_config['low'])
                log_high = np.log(param_config['high'])
                return np.exp(random.uniform(log_low, log_high))
            else:
                return random.uniform(param_config['low'], param_config['high'])
        elif param_config['type'] == 'int':
            return random.randint(param_config['low'], param_config['high'])
        else:
            raise ValueError(f"Unknown parameter type: {param_config['type']}")
    
    # Randomly sample N sets of hyperparameters
    random_hyperparameter_sets = [
        {key: sample_parameter(values) for key, values in hyperparameter_space.items()}
        for _ in range(n_trials)
    ]

    best_rmse = float("inf")
    best_hyperparams = None

    for hyperparams in random_hyperparameter_sets:
        print(f"\nTraining with hyperparams: {hyperparams}")

        # Combine fixed and searchable params
        all_params = {**fixed_params, **hyperparams}

        # Create estimator instance
        estimator = model_class(
            **all_params
        )
        
        # Train and predict
        predictor = estimator.train(training_data=train_ds,
                                    validation_data=val_ds)
        tss, forecasts = make_predictions(predictor=predictor, test_ds=val_ds)
        
        # Calculate RMSE
        predictions_mean = np.array([forecast.mean for forecast in forecasts])
        actuals = np.array([ts.iloc[-prediction_length:].values for ts in tss])
        actuals = actuals.reshape(predictions_mean.shape)
        
        if np.isnan(actuals).any():
            actuals = np.nan_to_num(actuals, nan=0.0)
        
        rmse = np.sqrt(np.mean((predictions_mean - actuals) ** 2))
        print(f"RMSE achieved: {rmse:.4f}")
        
        # Store best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_hyperparams = hyperparams

    print(f"\nBest RMSE: {best_rmse:.4f}")
    print(f"Best hyperparameters: {best_hyperparams}")
    return best_hyperparams


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


def prepare_callbacks(trainer_kwargs):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    loss_logger = LossHistoryLogger()

    if "callbacks" not in trainer_kwargs:
        trainer_kwargs["callbacks"] = []

    trainer_kwargs["callbacks"] = [
        cb for cb in trainer_kwargs["callbacks"] if not isinstance(cb, EarlyStopping)
    ]

    trainer_kwargs["callbacks"].extend([early_stopping, loss_logger])
    return trainer_kwargs, early_stopping, loss_logger


def get_actual_epochs(early_stopping, max_epochs):
    if early_stopping.stopped_epoch != 0:
        # return early_stopping.stopped_epoch - 9 # 10 epochs before the stopping point
        return early_stopping.stopped_epoch 
    return max_epochs


def trim_losses(val_losses, train_losses):
    best_epoch = int(np.argmin(val_losses))
    return val_losses[:best_epoch + 1], train_losses[:best_epoch + 1], best_epoch


def general_bayesian_search(train_ds, val_ds, prediction_length,
                            model_class, hyperparameter_space, n_trials, fixed_params=None):
    set_random_seed(42)
    fixed_params = fixed_params or {}

    def objective(trial, train_ds, val_ds, prediction_length, model_class, fixed_params):
        hyperparams = sample_hyperparameters(trial, hyperparameter_space)
        all_params = {**fixed_params, **hyperparams}

        all_params.setdefault("trainer_kwargs", {})
        trainer_kwargs, early_stopping, loss_logger = prepare_callbacks(all_params["trainer_kwargs"])
        all_params["trainer_kwargs"] = trainer_kwargs

        try:
            estimator = model_class(**all_params)
            predictor = estimator.train(training_data=train_ds, validation_data=val_ds)

            max_epochs = all_params["trainer_kwargs"].get("max_epochs")
            actual_epochs = get_actual_epochs(early_stopping, max_epochs)

            val_losses_trimmed, train_losses_trimmed, best_epoch = trim_losses(
                loss_logger.val_losses, loss_logger.train_losses
            )

            trial.set_user_attr("actual_epochs", actual_epochs)
            trial.set_user_attr("train_losses", train_losses_trimmed)
            trial.set_user_attr("val_losses", val_losses_trimmed)
            trial.set_user_attr("best_epoch", best_epoch)

            tss, forecasts = make_predictions(predictor=predictor, test_ds=val_ds)
            predictions_mean = np.array([forecast.mean for forecast in forecasts])
            actuals = np.array([ts.iloc[-prediction_length:].values for ts in tss])
            actuals = actuals.reshape(predictions_mean.shape)

            if np.isnan(actuals).any():
                actuals = np.nan_to_num(actuals, nan=0.0)

            return np.sqrt(np.mean((predictions_mean - actuals) ** 2))

        except Exception as e:
            print(f"Error with params {hyperparams}: {str(e)}")
            return float('inf')

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        partial(objective, train_ds=train_ds, val_ds=val_ds,
                prediction_length=prediction_length,
                model_class=model_class, fixed_params=fixed_params),
        n_trials=n_trials
    )

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"RMSE: {trial.value:.4f}")
    print(f"Best hyperparameters: {trial.params}")

    epochs = trial.user_attrs.get('actual_epochs', fixed_params.get('trainer_kwargs', {}).get('max_epochs', 40))
    return trial, epochs


def hyperparameter_search(train_ds, val_ds, prediction_length,
                         model_class, hyperparameter_space, n_trials, type,fixed_params=None):

    if type == 'random':
        return general_random_search(train_ds, val_ds, prediction_length,
                                    model_class, hyperparameter_space, n_trials, fixed_params)
    elif type == 'bayesian':
        return general_bayesian_search(train_ds, val_ds, prediction_length,
                                      model_class, hyperparameter_space, n_trials, fixed_params)
    else:
        raise ValueError(f"Unknown search type: {type}")
    

def save_best_hyperparameters(best_trial, best_epochs, sku, cluster_number, model):
    
    results = {
        'best_params': best_trial.params,
        'best_trial': best_trial.number,
        'best_epochs': best_epochs,
        'train_losses': best_trial.user_attrs.get('train_losses'),
        'val_losses': best_trial.user_attrs.get('val_losses')
    }
    with open(f'results/{model}/best_hyperparameters_sku_{sku}_cluster_{cluster_number}.json', 'w') as f:
        json.dump(results, f, indent=4)
