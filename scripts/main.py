# main.py
from utils.data_preprocessing import load_dataset, clean_flood_occurrences, merge_hydrological_data, add_flood_event_indicator
from utils.feature_engineering import preprocess_features
from models.hybrid_model import train_hybrid_model, hybrid_predict

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

# # Train hybrid model
# arima_model, lstm_model, scaler = train_hybrid_model(processed_data)

# Unpack the four returned values
arima_model, lstm_model, residuals_scaler, features_scaler = train_hybrid_model(processed_data)

# Make predictions
features = processed_data.drop(columns=['water_level', 'datetime', 'flood_event']).values
# predictions = hybrid_predict(arima_model, lstm_model, scaler, features)
predictions = hybrid_predict(
    arima_model=arima_model,
    lstm_model=lstm_model,
    residuals_scaler=residuals_scaler,
    features_scaler=features_scaler,
    features=features
)
print(predictions)