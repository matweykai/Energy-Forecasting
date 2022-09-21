from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np


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


def get_regr_prediction(data: list[float], forecast_len: int) -> list[float]:
    """Trains Linear regression model and makes prediction"""
    def get_features(t_data: np.array) -> np.array:
        """Extract features from data"""
        result = list()

        # Previous value
        result.append(t_data[-1])
        # Penultimate value
        result.append(t_data[-2])
        # Average for 3 days
        result.append(np.mean(t_data[-3:]))
        # Average for 5 days
        result.append(np.mean(t_data[-5:]))

        return np.array(result)

    train_len = 5
    # Forming train data
    X_train = np.ndarray((len(data) - train_len, train_len), dtype=float)
    y_train = np.ndarray((len(data) - train_len, 1), dtype=float)

    for ind in range(train_len, len(data)):
        X_train[ind - train_len, :] = np.array(data[ind - train_len: ind]).reshape((1, -1))
        y_train[ind - train_len] = data[ind]

    # Training model
    model = make_pipeline(FunctionTransformer(lambda x: np.array([get_features(row) for row in x])),
                          LinearRegression()).fit(X_train, y_train)

    # Prediction
    for item in range(forecast_len):
        t_data = np.array(data[-train_len:]).reshape((1, -1))
        data.append(model.predict(t_data)[0, 0])

    return data[-forecast_len:]
