# flood_hybrid_model.py

import sys
import os
import numpy as np
from flood_event_data_merger import main as fedm_main 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
sys.path.append(project_root)

from scripts.flood_event_data_merger import main as fedm_main

# --- ARIMA MODEL FUNCTIONS ---
def fit_arima(series, order=(5, 1, 0)):
    """Fit an ARIMA model and return the model and residuals."""
    arima_model = ARIMA(series, order=order)
    arima_fit = arima_model.fit()
    residuals = arima_fit.resid
    return arima_fit, residuals

# --- LSTM DATA PREPARATION FUNCTIONS ---
def prepare_lstm_data(residuals, features, sequence_length=10):
    """Prepare sequences for LSTM from residuals and features."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    features_scaled = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(sequence_length, len(residuals_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(residuals_scaled[i])
    return np.array(X), np.array(y), scaler

# --- LSTM MODEL FUNCTIONS ---
def build_lstm(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- HYBRID MODEL TRAINING FUNCTION ---
def train_hybrid_model(data, target_column='water_level'):
    """Train the ARIMA-LSTM hybrid model."""
    arima_model, residuals = fit_arima(data[target_column])
    features = data.drop(columns=[target_column, 'datetime', 'flood_event']).values
    X, y, scaler = prepare_lstm_data(residuals, features)
    
    lstm_model = build_lstm((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    
    return arima_model, lstm_model, scaler

# --- MAIN EXECUTION FUNCTION ---
def main():
    # File paths to datasets
    file_paths = {
        "Lagdo_Dam_waterlevel": "/path/to/Lagdo_Dam_waterlevel.xlsx",
        "Lagdo_Dam_SurfaceArea": "/path/to/Lagdo_Dam_SurfaceArea.xlsx",
        "Lagdo_Dam_VolumeVariation": "/path/to/Lagdo_Dam_VolumeVariation.xlsx",
        "Lagdo_Dam_Flood_Data": "/path/to/Lagdo_Dam_Flood_Data.xlsx"
    }

    # Load and preprocess data (reuse previously defined functions if needed)
    # Assuming a function to load and merge all datasets is implemented:
    merged_data = fedm_main(file_paths)  # Not included, would use functions previously discussed
    
    # Train the hybrid model
    arima_model, lstm_model, scaler = train_hybrid_model(merged_data)

    # Save or evaluate models
    print("Training complete.")