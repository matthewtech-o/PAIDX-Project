import pandas as pd

def feature_engineering(hydrological_data, economic_data):
    hydrological_data['Lagdo_Water_Release_Lag'] = hydrological_data['Lagdo_Water_Release_m3_s'].shift(1)
    economic_data['Economic_Stress_Index'] = (
        economic_data['Lagdo_Average_Household_Income'] * economic_data['Employment_Rate_%']
    ) / 100
    return hydrological_data, economic_data