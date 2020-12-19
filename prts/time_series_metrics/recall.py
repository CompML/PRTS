from typing import Any

import numpy as np

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TimeSeriesRecall(InterfaceTimeSeriesMetrics):
    """ This class calculates recall for time series """

    def score(self, real: np.ndarray, pred: np.ndarray) -> Any:
        """

        Args:
            real:
            pred:

        Returns:

        """
        # TODO: impl
