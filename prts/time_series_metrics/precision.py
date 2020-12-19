from typing import Any

import numpy as np

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TimeSeriesPrecision(InterfaceTimeSeriesMetrics):
    """ This class calculates precision for time series """

    def __init__(self, beta=1.0, alpha=0.0, cardinality="one", bias="flat"):
        """ Constructor

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
        # TODO: impl
