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

    def test_RecallClass_init(self):
        """ Test of init function.
        """

        test_case_1 = {'alpha': 0.0, 'cardinality': 'one', 'bias': 'flat'}
        test_case_2 = {'alpha': 0.0, 'cardinality': 'one', 'bias': None}
        test_case_3 = {'alpha': 10.0, 'cardinality': 'one', 'bias': 'flat'}

        # test of the normal call
        obj = TimeSeriesRecall(**test_case_1)
        self.assertEqual(obj.alpha, test_case_1['alpha'])
        self.assertEqual(obj.cardinality, test_case_1['cardinality'])
        self.assertEqual(obj.bias, test_case_1['bias'])

        # test of the invalid bias
        with self.assertRaises(Exception):
            obj = TimeSeriesRecall(**test_case_2)

        # test of the invalid alpha
        with self.assertRaises(Exception):
            obj = TimeSeriesRecall(**test_case_3)

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
