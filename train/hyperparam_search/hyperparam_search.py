import random
import pandas as pd
import numpy as np
from models.deep_learning.gluonts.functions import make_predictions, set_random_seed
import optuna
from optuna.samplers import TPESampler
from gluonts.evaluation import Evaluator
from functools import partial                                                   
from lightning.pytorch.callbacks import EarlyStopping
import json
import random
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)




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
def general_bayesian_search(train_ds, val_ds, prediction_length,
                          model_class, hyperparameter_space, n_trials, fixed_params=None):
    """
    Bayesian hyperparameter optimization using Optuna's TPE sampler
    
    Parameters:
    -----------
    train_ds : Training dataset
    val_ds : Validation dataset
    prediction_length : Forecast horizon
    model_class : The model estimator class (e.g. DeepAREstimator)
    hyperparameter_space : Dict with hyperparameter search spaces (with types)
    n_trials : Number of optimization trials
    fixed_params : Dict of fixed parameters for the model (optional)
    """
    set_random_seed(42)

    if fixed_params is None:
        fixed_params = {}
    
    # Define the objective function for Optuna
    def objective(trial, train_ds, val_ds, prediction_length, model_class, fixed_params):
        # Sample hyperparameters according to their types
        hyperparams = {}
        for param_name, param_config in hyperparameter_space.items():
            if param_config['type'] == 'categorical':
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name, param_config['values']
                )
            elif param_config['type'] == 'int':
                hyperparams[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'], 
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'float':
                hyperparams[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'], 
                    log=param_config.get('log', False)
                )
        
        # Combine fixed and searchable params
        all_params = {**fixed_params, **hyperparams}
        
        # Create fresh EarlyStopping for each trial
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            verbose=True
        )
        
        # Ensure trainer_kwargs exists and add our callback
        if "trainer_kwargs" not in all_params:
            all_params["trainer_kwargs"] = {}
        if "callbacks" not in all_params["trainer_kwargs"]:
            all_params["trainer_kwargs"]["callbacks"] = []
        
        # Clear any existing EarlyStopping callbacks
        all_params["trainer_kwargs"]["callbacks"] = [
            cb for cb in all_params["trainer_kwargs"]["callbacks"] 
            if not isinstance(cb, EarlyStopping)
        ]
        all_params["trainer_kwargs"]["callbacks"].append(early_stopping)
        
        try:
            # Create and train model
            estimator = model_class(**all_params)
            predictor = estimator.train(
                training_data=train_ds,
                validation_data=val_ds
            )
            
            # Get actual epochs trained
            actual_epochs = (
                early_stopping.stopped_epoch + 1 
                if early_stopping.stopped_epoch is not None
                else all_params.get("trainer_kwargs", {}).get("max_epochs", 20)
            )
            trial.set_user_attr("actual_epochs", actual_epochs)
            
            # Make predictions and calculate RMSE
            tss, forecasts = make_predictions(predictor=predictor, test_ds=val_ds)
            
            predictions_mean = np.array([forecast.mean for forecast in forecasts])
            actuals = np.array([ts.iloc[-prediction_length:].values for ts in tss])
            actuals = actuals.reshape(predictions_mean.shape)
            
            if np.isnan(actuals).any():
                actuals = np.nan_to_num(actuals, nan=0.0)
            
            rmse = np.sqrt(np.mean((predictions_mean - actuals) ** 2))
            return rmse
            
        except Exception as e:
            print(f"Error with params {hyperparams}: {str(e)}")
            return float('inf')
    
    # Create study with TPE sampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Run optimization
    study.optimize(
        partial(
            objective,
            train_ds=train_ds,
            val_ds=val_ds,
            prediction_length=prediction_length,
            model_class=model_class,
            fixed_params=fixed_params
        ),
        n_trials=n_trials
    )
    
    # Print and return results
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"RMSE: {trial.value:.4f}")
    print(f"Best hyperparameters: {trial.params}")
    return trial.params, trial.user_attrs.get('actual_epochs', fixed_params.get('trainer_kwargs', {}).get('max_epochs', 20))

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
    

def save_best_hyperparameters(best_params, best_epochs, sku, cluster_number, model):
    results = {
        'best_params': best_params,
        'best_epochs': best_epochs
    }
    # Save to JSON file
    with open(f'results/{model}/best_hyperparameters_sku_{sku}_cluster_{cluster_number}.json', 'w') as f:
        json.dump(results, f, indent=4)