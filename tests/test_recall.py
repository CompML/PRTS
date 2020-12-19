from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics
from prts.time_series_metrics.recall import TimeSeriesRecall

import unittest

class TestRecall(unittest.TestCase):
    def test_recall_inherited_InterfaceTimeSeriesMetrics(self):
        """ InterfaceTimeSeriesMetircsを継承しているかチェック
        """
        recall = TimeSeriesRecall()
        self.assertTrue(
            isinstance(recall, InterfaceTimeSeriesMetrics)
        )


