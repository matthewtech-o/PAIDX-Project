import pandas as pd

def load_dataset(file_path, parse_dates=None):
    """Load a dataset from a file."""
    return pd.read_excel(file_path, parse_dates=parse_dates)

def clean_flood_occurrences(flood_data):
    """Clean and prepare flood occurrence data."""
    # Remove rows with non-numeric years
    flood_data_cleaned = flood_data.loc[pd.to_numeric(flood_data['Year'], errors='coerce').notnull()].copy()

    # Use .loc for explicit assignments
    flood_data_cleaned.loc[:, 'Year'] = flood_data_cleaned['Year'].astype(int)

    
    # Handle 'Month(s) of Occurrence' with text months or missing values
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Replace with explicit downcasting to avoid FutureWarning
    flood_data_cleaned.loc[:, 'Month(s) of Occurrence'] = (
        flood_data_cleaned['Month(s) of Occurrence']
        .replace(month_mapping)
        .infer_objects()
    )

    flood_data_cleaned.loc[:, 'Month(s) of Occurrence'] = (
        flood_data_cleaned['Month(s) of Occurrence']
        .fillna(1)
        .infer_objects()
    )
    
    # Create datetime safely using .loc
    flood_data_cleaned.loc[:, 'datetime'] = pd.to_datetime(
        flood_data_cleaned['Year'].astype(str) + '-' + flood_data_cleaned['Month(s) of Occurrence'].astype(str) + '-01',
        errors='coerce'
    )
    return flood_data_cleaned

def merge_hydrological_data(water_levels, surface_area, volume_variation):
    """Merge hydrological datasets."""
    hydrological_data = water_levels.merge(surface_area, on="datetime", how="outer")
    hydrological_data = hydrological_data.merge(volume_variation, on="datetime", how="outer")
    return hydrological_data

def add_flood_event_indicator(hydrological_data, flood_occurrences_cleaned):
    """Add flood event indicator to hydrological data."""
    hydrological_data['flood_event'] = hydrological_data['datetime'].isin(
        flood_occurrences_cleaned['datetime']
    ).astype(int)
    return hydrological_data

def add_engineered_features(data):
    """Add lagged and seasonal features to the dataset."""
    # Add lagged features (e.g., previous water levels, surface area, volume)
    data['water_level_lag1'] = data['water_level'].shift(1)
    data['surface_area_lag1'] = data['surface_area'].shift(1)
    data['volume_variation_lag1'] = data['volume_variation'].shift(1)
    
    # Add seasonal indicators
    data['month'] = data['datetime'].dt.month
    data['year'] = data['datetime'].dt.year
    
    # Fill NaN values introduced by lagging
    data.bfill(inplace=True)
    
    return data

def main(file_paths):
    """Main function to load, clean, merge, and process datasets."""
    # Load datasets
    water_levels = load_dataset(file_paths["Lagdo_Dam_waterlevel"], parse_dates=["datetime"])
    surface_area = load_dataset(file_paths["Lagdo_Dam_SurfaceArea"], parse_dates=["datetime"])
    volume_variation = load_dataset(file_paths["Lagdo_Dam_VolumeVariation"], parse_dates=["datetime"])
    flood_occurrences = load_dataset(file_paths["Lagdo_Dam_Flood_Data"])

    # Clean flood occurrence data
    flood_occurrences_cleaned = clean_flood_occurrences(flood_occurrences)

    # Merge hydrological data
    hydrological_data = merge_hydrological_data(water_levels, surface_area, volume_variation)

    # Add flood event indicator
    merged_data = add_flood_event_indicator(hydrological_data, flood_occurrences_cleaned)

    # Add engineered features
    final_data = add_engineered_features(merged_data)

    return final_data


# File paths dictionary
file_paths = {
    "Lagdo_Dam_waterlevel": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_waterlevel_dahiti.xlsx",
    "Lagdo_Dam_SurfaceArea": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_SurfaceArea_Dahiti.xlsx",
    "Lagdo_Dam_VolumeVariation": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_VolumeVariation_Dahiti.xlsx",
    "Lagdo_Dam_Flood_Data": "/Users/matthewoladiran/Downloads/PAIDX-Project/data/raw/Lagdo_Dam_Flood_Data between 1982 to 2024.xlsx"
}

# Execute the main function
final_merged_data = main(file_paths)

# Display the processed dataset with new features
print(final_merged_data.head())  # Display a sample of the result