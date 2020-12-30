from typing import Union

import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics


class TimeSeriesPrecision(BaseTimeSeriesMetrics):
    """ This class calculates precision for time series"""

    def __init__(self, alpha=0.0, cardinality="one", bias="flat"):
        """Constructor

        Args:
            alpha (float, optional): 0 <= alpha <= 1. Defaults to 0.0.
            cardinality (str, optional): ["one", "reciprocal", "udf_gamma"]. Defaults to "one".
            bias (str, optional): ["flat", "front", "middle", "back"]. Defaults to "flat".
        """

        assert (alpha >= 0) & (alpha <= 1)
        assert cardinality in ["one", "reciprocal", "udf_gamma"]
        assert bias in ["flat", "front", "middle", "back"]

        self.alpha = alpha
        self.cardinality = cardinality
        self.bias = bias

    def score(self, real: Union[np.ndarray, list], pred: Union[np.ndarray, list]) -> float:
        """Computing precision score

        Args:
            real (np.ndarray or list):
                One-dimensional array of correct answers with values of 1 or 0.
            pred (np.ndarray or list):
                One-dimensional array of predicted answers with values of 1 or 0.

        Returns:
            float: precision
        """

        assert isinstance(real, np.ndarray) or isinstance(real, list)
        assert isinstance(pred, np.ndarray) or isinstance(pred, list)

        if not isinstance(real, np.ndarray):
            real = np.array(real)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        real_anomalies, predicted_anomalies = self._prepare_data(real, pred)
        precision = self._update_precision(real_anomalies, predicted_anomalies)

        return precision

    def _update_precision(self, real_anomalies: np.ndarray, predicted_anomalies: np.ndarray) -> float:
        """Update precision

        Args:
            real_anomalies (np.ndarray):
                2-dimensional array of the first and last indexes of each real anomaly range.
                e.g. np.array([[1933,  1953],[1958,  2000], ...])
            predicted_anomalies (np.ndarray):
                2-dimensional array of the first and last indexes of each predicted anomaly range.
                e.g. np.array([[1933,  1953],[1958,  2000], ...])

        Returns:
            float: precision
        """
        precision = 0
        if len(predicted_anomalies) == 0:
            return 0
        for i in range(len(predicted_anomalies)):
            range_p = predicted_anomalies[i, :]
            omega_reward = 0
            overlap_count = [0]
            for j in range(len(real_anomalies)):
                range_r = real_anomalies[j, :]
                omega_reward += self._compute_omega_reward(range_p, range_r, overlap_count)
            overlap_reward = self._gamma_function(overlap_count) * omega_reward
            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            precision += self.alpha * existence_reward + (1 - self.alpha) * overlap_reward
        precision /= len(predicted_anomalies)
        return precision
