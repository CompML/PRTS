from typing import Any

import numpy as np


class InterfaceTimeSeriesMetrics:
    """ This class is interface for time series metrics """

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