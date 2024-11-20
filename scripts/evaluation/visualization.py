import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(true_values, arima_predictions, lstm_predictions, hybrid_predictions):
    """
    Plot actual vs. predicted values for ARIMA, LSTM, and Hybrid models.

    Args:
        true_values (np.ndarray): Actual values.
        arima_predictions (np.ndarray): Predictions from the ARIMA model.
        lstm_predictions (np.ndarray): Predictions from the LSTM model.
        hybrid_predictions (np.ndarray): Predictions from the Hybrid model.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="Actual", color="black", linewidth=2)
    plt.plot(arima_predictions, label="ARIMA Predictions", linestyle="--", color="blue")
    plt.plot(lstm_predictions, label="LSTM Predictions", linestyle="--", color="green")
    plt.plot(hybrid_predictions, label="Hybrid Predictions", linestyle="--", color="red")
    plt.legend()
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.grid()
    plt.show()

def plot_residuals(residuals, model_name="Hybrid Model"):
    """
    Plot residuals distribution and time series.

    Args:
        residuals (np.ndarray): Residuals (true_values - predictions).
        model_name (str): Name of the model (e.g., Hybrid, ARIMA, LSTM).
    """
    # Histogram of Residuals
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7, color="purple", edgecolor="black")
    plt.title(f"Residuals Distribution ({model_name})")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    # Residuals Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, color="purple", linestyle="--")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Residuals Over Time ({model_name})")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.grid()
    plt.show()

def plot_feature_relationships(data, feature_name, target_name="water_level"):
    """
    Plot the relationship between a feature and the target variable.

    Args:
        data (pd.DataFrame): Data containing the features and target.
        feature_name (str): The feature to analyze.
        target_name (str): The target variable name (default: "water_level").
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[feature_name], data[target_name], alpha=0.7, color="blue")
    plt.title(f"Relationship Between {feature_name} and {target_name}")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.grid()
    plt.show()