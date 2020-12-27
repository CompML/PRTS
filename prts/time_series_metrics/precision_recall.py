from typing import Any

import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics


class TimeSeriesPrecisionRecall(BaseTimeSeriesMetrics):
    """ This class calculates precision and recall for time series """

    def score(self, real: np.ndarray, pred: np.ndarray) -> Any:
        """

        Args:
            real:
            pred:

        Returns:

        """
        # TODO: impl
