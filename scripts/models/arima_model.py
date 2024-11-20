# arima_model.py

from statsmodels.tsa.arima.model import ARIMA

def fit_arima(series, order=(5, 1, 0)):
    """
    Fit an ARIMA model to the time series data.

    Args:
        series (pd.Series): The target time series.
        order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
        arima_fit (ARIMAResults): The fitted ARIMA model.
        residuals (pd.Series): Residuals of the model.
    """
    arima_model = ARIMA(series, order=order)
    arima_fit = arima_model.fit()
    residuals = arima_fit.resid
    return arima_fit, residuals