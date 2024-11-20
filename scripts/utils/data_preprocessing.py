import pandas as pd

def load_dataset(file_path, parse_dates=None):
    """
    Load a dataset from a file path.

    Args:
        file_path (str): Path to the data file.
        parse_dates (list): Columns to parse as datetime.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_excel(file_path, parse_dates=parse_dates)

def clean_flood_occurrences(flood_data):
    """
    Clean and prepare flood occurrence data.

    Args:
        flood_data (pd.DataFrame): Raw flood occurrence dataset.

    Returns:
        pd.DataFrame: Cleaned flood occurrence dataset.
    """
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
    """
    Merge hydrological datasets.

    Args:
        water_levels (pd.DataFrame): Water level dataset.
        surface_area (pd.DataFrame): Surface area dataset.
        volume_variation (pd.DataFrame): Volume variation dataset.

    Returns:
        pd.DataFrame: Merged hydrological dataset.
    """
    hydrological_data = water_levels.merge(surface_area, on="datetime", how="outer")
    hydrological_data = hydrological_data.merge(volume_variation, on="datetime", how="outer")
    return hydrological_data

def add_flood_event_indicator(hydrological_data, flood_occurrences_cleaned):
    """
    Add flood event indicator to hydrological data.

    Args:
        hydrological_data (pd.DataFrame): Hydrological dataset.
        flood_occurrences (pd.DataFrame): Flood occurrence data.

    Returns:
        pd.DataFrame: Hydrological data with flood event indicator.
    """
    hydrological_data['flood_event'] = hydrological_data['datetime'].isin(
        flood_occurrences_cleaned['datetime']
    ).astype(int)
    return hydrological_data