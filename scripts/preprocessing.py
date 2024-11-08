import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(economic_data, flood_risk_data, hydrological_data):
    economic_data.fillna(method='ffill', inplace=True)
    flood_risk_data.fillna(method='bfill', inplace=True)
    hydrological_data.dropna(inplace=True)

    scaler = MinMaxScaler()
    economic_data[['Lagdo_Average_Household_Income']] = scaler.fit_transform(
        economic_data[['Lagdo_Average_Household_Income']])
    return economic_data, flood_risk_data, hydrological_data