# hybrid_model.py

import numpy as np
from models.arima_model import fit_arima
from models.lstm_model import build_lstm
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(residuals, features, sequence_length=10):
    """
    Prepare LSTM input sequences from residuals and additional features.

    Args:
        residuals (pd.Series): Residuals from the ARIMA model.
        features (np.ndarray): Additional features for the LSTM.
        sequence_length (int): Number of timesteps in each LSTM sequence.

    Returns:
        X (np.ndarray): Input features for LSTM.
        y (np.ndarray): Target variable for LSTM.
        residuals_scaler (MinMaxScaler): Scaler for residuals.
        features_scaler (MinMaxScaler): Scaler for features.
    """
    # Scale residuals
    residuals_scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = residuals_scaler.fit_transform(residuals.values.reshape(-1, 1))

    # Scale features
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = features_scaler.fit_transform(features)

    # Create LSTM sequences
    X, y = [], []
    for i in range(sequence_length, len(residuals_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(residuals_scaled[i])
    return np.array(X), np.array(y), residuals_scaler, features_scaler


def train_hybrid_model(data, target_column='water_level'):
    """
    Train the ARIMA-LSTM hybrid model.

    Args:
        data (pd.DataFrame): The dataset containing features and target.
        target_column (str): The target column name.

    Returns:
        arima_model (ARIMAResults): The trained ARIMA model.
        lstm_model (Sequential): The trained LSTM model.
        residuals_scaler (MinMaxScaler): Scaler for ARIMA residuals.
        features_scaler (MinMaxScaler): Scaler for features.
    """
    # Train ARIMA
    arima_model, residuals = fit_arima(data[target_column])

    # Prepare data for LSTM
    features = data.drop(columns=[target_column, 'datetime', 'flood_event']).values
    X, y, residuals_scaler, features_scaler = prepare_lstm_data(residuals, features)

    # Train LSTM
    lstm_model = build_lstm((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

    return arima_model, lstm_model, residuals_scaler, features_scaler


def hybrid_predict(arima_model, lstm_model, residuals_scaler, features_scaler, features, sequence_length=10):
    """
    Generate predictions using the hybrid ARIMA-LSTM model.

    Args:
        arima_model (ARIMAResults): The trained ARIMA model.
        lstm_model (Sequential): The trained LSTM model.
        residuals_scaler (MinMaxScaler): Scaler for ARIMA residuals.
        features_scaler (MinMaxScaler): Scaler for features.
        features (np.ndarray): Feature matrix for LSTM.
        sequence_length (int): Number of timesteps for LSTM sequences.

    Returns:
        hybrid_pred (np.ndarray): Final hybrid model predictions.
    """
    # ARIMA predictions
    arima_pred = arima_model.forecast(steps=len(features))

    # Scale features for LSTM
    features_scaled = features_scaler.transform(features)
    X_test = []
    for i in range(sequence_length, len(features_scaled)):
        X_test.append(features_scaled[i-sequence_length:i])
    X_test = np.array(X_test)

    # Handle NaN values in X_test
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)

    # Predict residuals with LSTM
    lstm_residuals = lstm_model.predict(X_test)
    lstm_residuals = lstm_residuals.reshape(-1, 1)  # Ensure shape is (n_samples, 1)

    # Ensure no NaN in LSTM predictions
    if np.isnan(lstm_residuals).any():
        lstm_residuals = np.nan_to_num(lstm_residuals, nan=0.0)

    lstm_residuals = residuals_scaler.inverse_transform(lstm_residuals)  # Use residuals scaler

    # Combine predictions
    hybrid_pred = arima_pred[:len(lstm_residuals)] + lstm_residuals.flatten()

    # Ensure no NaN in final predictions
    if np.isnan(hybrid_pred).any():
        hybrid_pred = np.nan_to_num(hybrid_pred, nan=np.nanmean(hybrid_pred))

    return hybrid_pred