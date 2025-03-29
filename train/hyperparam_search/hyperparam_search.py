import random
import pandas as pd
import numpy as np
from models.deep_learning.gluonts.functions import make_predictions
import optuna
from optuna.samplers import TPESampler
import numpy as np
from gluonts.evaluation import Evaluator
from functools import partial                                                   



# General random search for hyperparameter tuning
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
    hyperparameter_space : Dict of hyperparameter search ranges
    n_trials : Number of random trials
    fixed_params : Dict of fixed parameters for the model (optional)
    """
    if fixed_params is None:
        fixed_params = {}
    
    # Randomly sample N sets of hyperparameters
    random_hyperparameter_sets = [
        {key: random.choice(values) for key, values in hyperparameter_space.items()}
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
        predictor = estimator.train(training_data=train_ds)
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
        
        try:
            # Create and train model
            estimator = model_class(**all_params)
            predictor = estimator.train(training_data=train_ds)
            
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
            return float('inf')  # Return worst possible score if error occurs
    
    # Create study with TPE sampler
    sampler = TPESampler(seed=42)  # For reproducibility
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
    
    return trial.params