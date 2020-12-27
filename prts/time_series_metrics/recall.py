from typing import Any

import numpy as np

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TimeSeriesRecall(InterfaceTimeSeriesMetrics):
    """ This class calculates recall for time series """

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
        recall = self._update_recall(real_anomalies, predicted_anomalies)

        return recall

    def _update_recall(self, real_anomalies, predicted_anomalies):
        recall = 0
        if len(real_anomalies) == 0:
            return 0
        for i in range(len(real_anomalies)):
            omega_reward = 0
            overlap_count = [0]
            range_r = real_anomalies[i, :]
            for j in range(len(predicted_anomalies)):
                range_p = predicted_anomalies[j, :]
                omega_reward += self._compute_omega_reward(range_r, range_p, overlap_count)
            overlap_reward = self._gamma_function(overlap_count) * omega_reward

            if overlap_count[0] > 0:
                existence_reward = 1
            else:
                existence_reward = 0

            recall += self.alpha * existence_reward + (1 - self.alpha) * overlap_reward
        recall /= len(real_anomalies)
        return recall
