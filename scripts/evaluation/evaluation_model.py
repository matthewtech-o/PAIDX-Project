import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(true_values, predictions):
    """
    Evaluate model performance using various metrics.

    Args:
        true_values (np.ndarray): Actual values.
        predictions (np.ndarray): Predicted values.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    rmse = mean_squared_error(true_values, predictions, squared=False)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}

def analyze_residuals(true_values, predictions):
    """
    Analyze residuals to assess prediction errors.

    Args:
        true_values (np.ndarray): Actual values.
        predictions (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Residuals (true_values - predictions).
    """
    residuals = true_values - predictions
    return residuals

def compare_models(true_values, arima_predictions, lstm_predictions, hybrid_predictions):
    """
    Compare the performance of ARIMA, LSTM, and Hybrid models.

    Args:
        true_values (np.ndarray): Actual values.
        arima_predictions (np.ndarray): Predictions from the ARIMA model.
        lstm_predictions (np.ndarray): Predictions from the LSTM model.
        hybrid_predictions (np.ndarray): Predictions from the Hybrid model.

    Returns:
        dict: A dictionary of metrics for each model.
    """
    metrics = {
        "ARIMA": evaluate_model(true_values, arima_predictions),
        "LSTM": evaluate_model(true_values, lstm_predictions),
        "Hybrid": evaluate_model(true_values, hybrid_predictions),
    }
    return metrics