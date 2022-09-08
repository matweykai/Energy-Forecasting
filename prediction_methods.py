from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy import stats
from scipy.special import inv_boxcox


def get_ARMA_prediction(data: list[float], p: int, d: int, q: int, forecast_len: int) -> list[float]:
    """Trains ARMA model with specified parameters and makes prediction"""

    # Stabilizing dispersion
    model_data, lmb = stats.boxcox(data)

    # TODO: Add logging on training
    arma_model = ARIMA(model_data, order=(p, d, q)).fit()

    # Making prediction
    pred = arma_model.predict(len(data), len(data) + forecast_len)

    # TODO: Add logging after prediction
    # Inverting box-cox transform
    result = inv_boxcox(pred, lmb)

    return result


def get_exp_autoreg_prediction(data: list[float], forecast_len: int) -> list[float]:
    """Trains exponential autoregressive model and makes prediction for the specified forecast len"""
    model = SimpleExpSmoothing(data)

    model = model.fit(smoothing_level=0.6)

    result = model.predict(len(data), len(data) + forecast_len)

    return result
