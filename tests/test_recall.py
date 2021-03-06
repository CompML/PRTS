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

    def test_RecallClass_score(self):
        """Test of score function.
        """

        # test normal case
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        obj = TimeSeriesRecall()

        score = obj.score(real, pred)
        self.assertEqual(score, 0.5)

        # test invalid inputs
        real = None
        pred = np.array([0, 1, 0, 0, 0])
        with self.assertRaises(Exception):
            score = obj.score(real, pred)

        real = np.array([1, 1, 0, 0, 0])
        pred = None
        with self.assertRaises(Exception):
            score = obj.score(real, pred)

    def test_RecallClass_update_recall(self):
        """Test of _update_recall function.
        """

        # test of the normal case
        real = np.array([0, 1, 0, 0, 0])
        pred = np.array([1, 1, 0, 0, 0])

        obj = TimeSeriesRecall()
        real_anomalies, predicted_anomalies = obj._prepare_data(real, pred)

        score = obj._update_recall(real_anomalies, predicted_anomalies)
        self.assertEqual(score, 1.0)

        # test of the empty case
        empty_real = np.array([])
        empty_pred = np.array([])

        score = obj._update_recall(empty_real, empty_pred)
        self.assertEqual(score, 0.0)

    def test_recall_function(self):
        """Test of ts_recall function.
        """

        # test case1
        real = np.array([1, 0, 0, 0, 0])
        pred = np.array([1, 1, 0, 0, 0])

        score = ts_recall(real, pred)
        self.assertEqual(score, 1.0)

        # test case2
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 0, 1, 1, 1])

        score = ts_recall(real, pred)
        self.assertEqual(score, 0.0)

    def test_recall_function_with_list(self):
        """Teest of ts_recall function with list type arguments.
        """

        real = [1, 1, 0, 0, 0]
        pred = [1, 1, 1, 1, 0]

        score = ts_recall(real, pred)
        self.assertEqual(score, 1.0)
