import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.precision import TimeSeriesPrecision
from prts.time_series_metrics.recall import TimeSeriesRecall


class TimeSeriesFScore(BaseTimeSeriesMetrics):
    """ This class calculates f-score for time series"""

    def __init__(self, beta=1.0, p_alpha=0.0, r_alpha=0.0, cardinality="one", p_bias="flat", r_bias="flat"):
        """Constructor

        Args:
            beta (float, optional): determines the weight of recall in the combined score.. Defaults to 1.0.
            p_alpha (float, optional): alpha of precision, 0<=alpha_p<=1. Defaults to 0.0.
            r_alpha (float, optional): alpha of recall, 0<=alpha<=1. Defaults to 0.0.
            cardinality (str, optional): ["one", "reciprocal", "udf_gamma"]. Defaults to "one".
            p_bias (str, optional): bias of precision, ["flat", "front", "middle", "back"]. Defaults to "flat".
            r_bias (str, optional): bias of recall, ["flat", "front", "middle", "back"]. Defaults to "flat".
        """

        assert beta >= 0

        self.beta = beta
        self.p_alpha = p_alpha
        self.r_alpha = r_alpha
        self.cardinality = cardinality
        self.p_bias = p_bias
        self.r_bias = r_bias

    def score(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Computing fbeta score

        Args:
            real (np.ndarray):
                One-dimensional array of correct answers with values of 1 or 0.
            pred (np.ndarray):
                One-dimensional array of predicted answers with values of 1 or 0.

        Returns:
            float: fbeta
        """

        assert isinstance(real, np.ndarray) or isinstance(real, list)
        assert isinstance(pred, np.ndarray) or isinstance(pred, list)

        if not isinstance(real, np.ndarray):
            real = np.array(real)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        precision = TimeSeriesPrecision(self.p_alpha, self.cardinality, self.p_bias).score(real, pred)
        recall = TimeSeriesRecall(self.r_alpha, self.cardinality, self.r_bias).score(real, pred)

        if precision + recall != 0:
            f_beta = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall)
        else:
            f_beta = 0

        return f_beta
