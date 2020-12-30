from typing import Union

import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics


class TimeSeriesRecall(BaseTimeSeriesMetrics):
    """ This class calculates recall for time series """

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
        """Computing recall score

        Args:
            real (np.ndarray or list):
                One-dimensional array of correct answers with values of 1 or 0.
            pred (np.ndarray or list):
                One-dimensional array of predicted answers with values of 1 or 0.
        Returns:
            float: recall
        """

        assert isinstance(real, np.ndarray) or isinstance(real, list)
        assert isinstance(pred, np.ndarray) or isinstance(pred, list)

        if not isinstance(real, np.ndarray):
            real = np.array(real)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        real_anomalies, predicted_anomalies = self._prepare_data(real, pred)
        recall = self._update_recall(real_anomalies, predicted_anomalies)

        return recall

    def _update_recall(self, real_anomalies: np.ndarray, predicted_anomalies: np.ndarray) -> float:
        """Update recall
        Args:
            real_anomalies (np.ndarray):
                2-dimensional array of the first and last indexes of each real anomaly range.
                e.g. np.array([[1933,  1953],[1958,  2000], ...])
            predicted_anomalies (np.ndarray):
                2-dimensional array of the first and last indexes of each predicted anomaly range.
                e.g. np.array([[1933,  1953],[1958,  2000], ...])
        Returns:
            float: recall
        """

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
