# main.py
import numpy as np
from utils.data_preprocessing import load_dataset, clean_flood_occurrences, merge_hydrological_data, add_flood_event_indicator
from utils.feature_engineering import preprocess_features
from models.hybrid_model import train_hybrid_model, hybrid_predict
from evaluation.evaluation_model import evaluate_model, analyze_residuals, compare_models
from evaluation.visualization import plot_predictions, plot_residuals, plot_feature_relationships

def main():
    # Paths to datasets
    file_paths = {
        "Lagdo_Dam_waterlevel": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_waterlevel_dahiti.xlsx",
        "Lagdo_Dam_SurfaceArea": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_SurfaceArea_Dahiti.xlsx",
        "Lagdo_Dam_VolumeVariation": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_VolumeVariation_Dahiti.xlsx",
        "Lagdo_Dam_Flood_Data": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_Flood_Data between 1982 to 2024.xlsx"
    }

    # Load datasets
    water_levels = load_dataset(file_paths["Lagdo_Dam_waterlevel"], parse_dates=["datetime"])
    surface_area = load_dataset(file_paths["Lagdo_Dam_SurfaceArea"], parse_dates=["datetime"])
    volume_variation = load_dataset(file_paths["Lagdo_Dam_VolumeVariation"], parse_dates=["datetime"])
    flood_occurrences = load_dataset(file_paths["Lagdo_Dam_Flood_Data"])

    # Preprocess data
    flood_occurrences = clean_flood_occurrences(flood_occurrences)
    hydrological_data = merge_hydrological_data(water_levels, surface_area, volume_variation)
    hydrological_data = add_flood_event_indicator(hydrological_data, flood_occurrences)

    # Feature engineering
    processed_data = preprocess_features(hydrological_data, target_column='water_level')

    # Train hybrid model
    arima_model, lstm_model, residuals_scaler, features_scaler = train_hybrid_model(processed_data)

    # Make predictions
    features = processed_data.drop(columns=['water_level', 'datetime', 'flood_event']).values
    true_values = processed_data["water_level"].values
    hybrid_predictions = hybrid_predict(arima_model, lstm_model, residuals_scaler, features_scaler, features)

    # Evaluate Models
    arima_predictions = arima_model.forecast(steps=len(features))
    # Reshape features for LSTM input
    features_scaled = features_scaler.transform(features)
    sequence_length = 10  # Replace with the sequence length used during training

    X_lstm = []
    for i in range(sequence_length, len(features_scaled)):
        X_lstm.append(features_scaled[i-sequence_length:i])

    X_lstm = np.array(X_lstm)  # Convert to numpy array

    # Make predictions with the LSTM model
    lstm_predictions = lstm_model.predict(X_lstm).flatten()
    # lstm_predictions = lstm_model.predict(features_scaler.transform(features)).flatten()

    # Align true_values and predictions
    min_samples = min(len(true_values), len(arima_predictions), len(lstm_predictions), len(hybrid_predictions))
    true_values = true_values[-min_samples:]
    arima_predictions = arima_predictions[-min_samples:]
    lstm_predictions = lstm_predictions[-min_samples:]
    hybrid_predictions = hybrid_predictions[-min_samples:]


    def remove_nan(true_values, *predictions):
        """
        Removes NaN values from true_values and predictions.

        Args:
            true_values (array-like): Array of true values.
            *predictions (array-like): Arrays of predictions.

        Returns:
            tuple: Cleaned true_values and predictions.
        """
        # Convert to numpy arrays
        true_values = np.array(true_values)
        predictions = [np.array(pred) for pred in predictions]

        # Identify valid indices
        valid_indices = ~np.isnan(true_values)
        for prediction in predictions:
            valid_indices &= ~np.isnan(prediction)

        # Filter out NaN values
        return (true_values[valid_indices],) + tuple(pred[valid_indices] for pred in predictions)

    # Remove NaN values from true_values and predictions
    true_values, arima_predictions, lstm_predictions, hybrid_predictions = remove_nan(
    true_values, arima_predictions, lstm_predictions, hybrid_predictions
)

    metrics = compare_models(true_values, arima_predictions, lstm_predictions, hybrid_predictions)
    print("Model Evaluation Metrics:")
    print(metrics)

    # Analyze Residuals
    hybrid_residuals = analyze_residuals(true_values, hybrid_predictions)

    # Visualize Results
    plot_predictions(true_values, arima_predictions, lstm_predictions, hybrid_predictions)
    plot_residuals(hybrid_residuals, model_name="Hybrid Model")
    plot_feature_relationships(hydrological_data, "surface_area", target_name="water_level")

if __name__ == "__main__":
    main()