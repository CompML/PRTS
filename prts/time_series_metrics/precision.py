from typing import Any

import numpy as np

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TimeSeriesPrecision(InterfaceTimeSeriesMetrics):
    """ This class calculates precision for time series """

    def __init__(self, beta=1.0, alpha=0.0, cardinality="one", bias="flat"):
        """Constructor

        Args:
            beta (float, optional): 0 < beta. Defaults to 1.0.
            alpha (float, optional): 0 <= alpha <= 1. Defaults to 0.0.
            cardinality (str, optional): ["one", "reciprocal", "udf_gamma"]. Defaults to "one".
            bias (str, optional): ["flat", "front", "middle", "back"]. Defaults to "flat".
        """

        assert (alpha >= 0) & (alpha <= 1)
        assert beta > 0
        assert cardinality in ["one", "reciprocal", "udf_gamma"]
        assert bias in ["flat", "front", "middle", "back"]

        self.beta = beta
        self.alpha = alpha
        self.cardinality = cardinality
        self.bias = bias

    def score(self, real: np.ndarray, pred: np.ndarray) -> Any:
        """

        Args:
            real:
            pred:

        Returns:

        """

        assert isinstance(real, np.ndarray)
        assert isinstance(pred, np.ndarray)
        real_anomalies, predicted_anomalies = self._prepare_data(real, pred)
        precision = self._update_precision(real_anomalies, predicted_anomalies)

        return precision

    def _update_precision(self, real_anomalies, predicted_anomalies):
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
