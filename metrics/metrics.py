import numpy as np
import pandas as pd

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
    
    def calculate_metrics(self, group):
        y_true = group['cant_vta']
        metrics = {}

        # Identify all columns that start with 'cant_vta_pred_'
        pred_columns = [col for col in group.columns if col.startswith('cant_vta_pred_')]

        for col in pred_columns:
            mse = round(Metrics.mean_squared_error(y_true, group[col]),1)
            rmse = round(Metrics.root_mean_squared_error(y_true, group[col]),1)
            metrics[f'mse_{col}'] = mse
            metrics[f'rmse_{col}'] = rmse

        return pd.Series(metrics)

    def create_summary_dataframe(self, test_df):
        # TODO: ANALIZE WHAT TO DO WHEN THERE ARE NULLS IN THE REAL VALUES
        summary_df = test_df.groupby(['pdv_codigo', 'codigo_barras_sku']).apply(self.calculate_metrics).reset_index()

        rmse_columns = [col for col in summary_df.columns if col.startswith('rmse_')]
        mse_columns = [col for col in summary_df.columns if col.startswith('mse_')]

        
        summary_df['best_rmse'] = summary_df[rmse_columns].idxmin(axis=1)
        summary_df['best_mse'] = summary_df[mse_columns].idxmin(axis=1)

        return summary_df