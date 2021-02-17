from typing import Any

import numpy as np


class BaseTimeSeriesMetrics:
    """Base class for time series metrics """

    def score(self, real: np.ndarray, pred: np.ndarray) -> Any:
        """

        Args:
            real:
            pred:

        Returns:

        """
        ...

    def _udf_gamma(self):
        """The function of the user-defined gamma.

        Returns:
            float: the value of the user-defined gamma
        """

        return 1.0

    def _gamma_select(self, gamma: str, overlap: int) -> float:
        """The function of selecting the gamma value according to the parameters.

        Args:
            gamma: str
                - 'one': the value 1
                - 'reciprocal';: a reciprocal of the overlap
                - 'udf_gamma': user defined gamma
            overlap: int
                overlap between real and pred

        Returns:
            float: the selected gamma value
        """
        assert type(overlap) == int, TypeError("")

        if gamma == "one":
            return 1.0
        elif gamma == "reciprocal":
            if overlap > 1:
                return 1.0 / overlap
            else:
                return 1.0
        elif gamma == "udf_gamma":
            if overlap > 1:
                return 1.0 / self._udf_gamma()
            else:
                return 1.0
        else:
            raise ValueError(f"Expected one of one, reciprocal, udf_gamma. gamma type string: {gamma}")

    def _gamma_function(self, overlap_count):
        overlap = overlap_count[0]
        return self._gamma_select(self.cardinality, overlap)

    def _compute_omega_reward(self, r1, r2, overlap_count):
        if r1[1] < r2[0] or r1[0] > r2[1]:
            return 0
        else:
            overlap_count[0] += 1
            overlap = np.zeros(r1.shape)
            overlap[0] = max(r1[0], r2[0])
            overlap[1] = min(r1[1], r2[1])
            return self._omega_function(r1, overlap)

    def _omega_function(self, rrange, overlap):
        anomaly_length = rrange[1] - rrange[0] + 1
        my_positional_bias = 0
        max_positional_bias = 0
        temp_bias = 0
        for i in range(1, anomaly_length + 1):
            temp_bias = self._delta_function(i, anomaly_length)
            max_positional_bias += temp_bias
            j = rrange[0] + i - 1
            if j >= overlap[0] and j <= overlap[1]:
                my_positional_bias += temp_bias
        if max_positional_bias > 0:
            res = my_positional_bias / max_positional_bias
            return res
        else:
            return 0

    def _delta_function(self, t, anomaly_length):
        return self._delta_select(self.bias, t, anomaly_length)

    def _delta_select(self, delta, t, anomaly_length):
        if delta == "flat":
            return 1.0
        elif delta == "front":
            return float(anomaly_length - t + 1.0)
        elif delta == "middle":
            if t <= anomaly_length / 2.0:
                return float(t)
            else:
                return float(anomaly_length - t + 1.0)
        elif delta == "back":
            return float(t)
        elif delta == "udf_delta":
            return self._udf_delta(t, anomaly_length)
        else:
            raise Exception("Invalid positional bias value")

    def _udf_delta(self):
        """
        user defined delta function
        """

        return 1.0

    def _shift(self, arr, num, fill_value=np.nan):
        arr = np.roll(arr, num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr

    def _prepare_data(self, values_real, values_pred):

        assert len(values_real) == len(values_pred)
        assert np.allclose(np.unique(values_real), np.array([0, 1])) or np.allclose(
            np.unique(values_real), np.array([1])
        )
        assert np.allclose(np.unique(values_pred), np.array([0, 1])) or np.allclose(
            np.unique(values_pred), np.array([1])
        )

        predicted_anomalies_ = np.argwhere(values_pred == 1).ravel()
        predicted_anomalies_shift_forward = self._shift(predicted_anomalies_, 1, fill_value=predicted_anomalies_[0])
        predicted_anomalies_shift_backward = self._shift(predicted_anomalies_, -1, fill_value=predicted_anomalies_[-1])
        predicted_anomalies_start = np.argwhere(
            (predicted_anomalies_shift_forward - predicted_anomalies_) != -1
        ).ravel()
        predicted_anomalies_finish = np.argwhere(
            (predicted_anomalies_ - predicted_anomalies_shift_backward) != -1
        ).ravel()
        predicted_anomalies = np.hstack(
            [
                predicted_anomalies_[predicted_anomalies_start].reshape(-1, 1),
                predicted_anomalies_[predicted_anomalies_finish].reshape(-1, 1),
            ]
        )

        real_anomalies_ = np.argwhere(values_real == 1).ravel()
        real_anomalies_shift_forward = self._shift(real_anomalies_, 1, fill_value=real_anomalies_[0])
        real_anomalies_shift_backward = self._shift(real_anomalies_, -1, fill_value=real_anomalies_[-1])
        real_anomalies_start = np.argwhere((real_anomalies_shift_forward - real_anomalies_) != -1).ravel()
        real_anomalies_finish = np.argwhere((real_anomalies_ - real_anomalies_shift_backward) != -1).ravel()
        real_anomalies = np.hstack(
            [
                real_anomalies_[real_anomalies_start].reshape(-1, 1),
                real_anomalies_[real_anomalies_finish].reshape(-1, 1),
            ]
        )

        return real_anomalies, predicted_anomalies
