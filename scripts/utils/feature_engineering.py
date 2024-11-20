# feature_engineering.py

import pandas as pd

def add_lagged_features(data, target_column, lags=3):
    """
    Add lagged features for a target variable.

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): Column to create lagged features for.
        lags (int): Number of lagged periods to include.

    Returns:
        pd.DataFrame: Dataset with lagged features.
    """
    for lag in range(1, lags + 1):
        data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)
    return data

def add_seasonality_indicators(data, datetime_column='datetime'):
    """
    Add seasonal indicators like month and year.

    Args:
        data (pd.DataFrame): Input dataset.
        datetime_column (str): Name of the datetime column.

    Returns:
        pd.DataFrame: Dataset with seasonality indicators.
    """
    data['month'] = data[datetime_column].dt.month
    data['year'] = data[datetime_column].dt.year
    return data

def preprocess_features(data, target_column, datetime_column='datetime', lags=3, fill_method='bfill'):
    """
    Preprocess features by adding lagged variables and seasonal indicators, with optional NaN handling.

    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): Column to create lagged features for.
        datetime_column (str): Name of the datetime column.
        lags (int): Number of lagged periods to include.
        fill_method (str): Method to handle NaN values ('bfill', 'ffill', or 'dropna').

    Returns:
        pd.DataFrame: Processed dataset with engineered features.
    """
    # Add lagged features
    data = add_lagged_features(data, target_column, lags)

    # Add seasonal indicators
    data = add_seasonality_indicators(data, datetime_column)

    # Handle NaN values based on the chosen fill method
    if fill_method == 'bfill':
        data.bfill(inplace=True)  # Backfill missing values
    elif fill_method == 'ffill':
        data.ffill(inplace=True)  # Forward fill missing values
    elif fill_method == 'dropna':
        data.dropna(inplace=True)  # Drop rows with NaN values
    else:
        raise ValueError("Invalid fill_method. Choose 'bfill', 'ffill', or 'dropna'.")

    return data