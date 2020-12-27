from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.recall import TimeSeriesRecall

import unittest

class TestRecall(unittest.TestCase):
    def test_recall_inherited_BaseTimeSeriesMetrics(self):
        """ BaseTimeSeriesMetircsを継承しているかチェック
        """
        recall = TimeSeriesRecall()
        self.assertTrue(
            isinstance(recall, BaseTimeSeriesMetrics)
        )


