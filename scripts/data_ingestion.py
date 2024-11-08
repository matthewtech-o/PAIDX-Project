import pandas as pd

def load_data():
    economic_data = pd.read_csv('data/Economic_Impact_Lagdo_Dataset.csv')
    flood_risk_data = pd.read_csv('data/Flood_Risk_Lagdo_Dataset.csv')
    hydrological_data = pd.read_csv('data/Hydrological_Lagdo_Dataset.csv')
    return economic_data, flood_risk_data, hydrological_data