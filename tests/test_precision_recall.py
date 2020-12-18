import unittest

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics
from prts.time_series_metrics.precision_recall import TimeSeriesPrecisionRecall


class TestPrecision(unittest.TestCase):
    def test_PrecisionRecallClass_inherited_InterfaceTimeSeriesMetrics(self):
        """ InterfaceTimeSeriesMetircsを継承しているかチェック
        """
        obj = TimeSeriesPrecisionRecall()
        self.assertTrue(
            isinstance(obj, InterfaceTimeSeriesMetrics)
        )


if __name__ == '__main__':
    unittest.main()
