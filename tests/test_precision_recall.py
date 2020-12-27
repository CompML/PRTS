import unittest

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.precision_recall import TimeSeriesPrecisionRecall


class TestPrecision(unittest.TestCase):
    def test_PrecisionRecallClass_inherited_BaseTimeSeriesMetrics(self):
        """ BaseTimeSeriesMetircsを継承しているかチェック
        """
        obj = TimeSeriesPrecisionRecall()
        self.assertTrue(
            isinstance(obj, BaseTimeSeriesMetrics)
        )


if __name__ == '__main__':
    unittest.main()
