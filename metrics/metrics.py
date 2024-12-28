import numpy as np

class Metrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE).
        
        Parameters:
        y_true (array-like): Array of real values.
        y_pred (array-like): Array of predicted values.
        
        Returns:
        float: Mean Squared Error.
        """
        n = len(y_true)
        mse = np.sum((y_true - y_pred) ** 2) / n
        return mse

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Parameters:
        y_true (array-like): Array of real values.
        y_pred (array-like): Array of predicted values.
        
        Returns:
        float: Root Mean Squared Error.
        """
        mse = Metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse