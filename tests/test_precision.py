import unittest

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics
from prts.time_series_metrics.precision import TimeSeriesPrecision


class TestPrecision(unittest.TestCase):
    def test_PrecisionClass_inherited_InterfaceTimeSeriesMetrics(self):
        """ InterfaceTimeSeriesMetircsを継承しているかチェック
        """
        obj = TimeSeriesPrecision()
        self.assertTrue(
            isinstance(obj, InterfaceTimeSeriesMetrics)
        )
