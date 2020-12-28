import unittest
import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.recall import TimeSeriesRecall
from prts import ts_recall


class TestRecall(unittest.TestCase):
    def test_recall_inherited_BaseTimeSeriesMetrics(self):
        """ BaseTimeSeriesMetircsを継承しているかチェック
        """
        recall = TimeSeriesRecall()
        self.assertTrue(
            isinstance(recall, BaseTimeSeriesMetrics)
        )

    def test_recall_function(self):
        """Teest of ts_recall function.
        """

        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([1, 1, 1, 1, 0])

        score = ts_recall(real, pred)
        self.assertEqual(score, 1.0)

    def test_recall_function_with_list(self):
        """Teest of ts_recall function with list type arguments.
        """

        real = [1, 1, 0, 0, 0]
        pred = [1, 1, 1, 1, 0]

        score = ts_recall(real, pred)
        self.assertEqual(score, 1.0)
