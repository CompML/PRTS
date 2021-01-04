import unittest
import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.fscore import TimeSeriesFScore
from prts import ts_fscore


class TestFScore(unittest.TestCase):
    def test_FScoreClass_inherited_BaseTimeSeriesMetrics(self):
        """ Check if it inherits BaseTimeSeriesMetrics
        """
        obj = TimeSeriesFScore()
        self.assertTrue(
            isinstance(obj, BaseTimeSeriesMetrics)
        )

    def test_FscoreClass_init(self):
        """ Test of init function
        """

        test_case_1 = {'beta': 1.0, 'p_alpha': 0.0, 'r_alpha': 0.0}
        test_case_2 = {'beta': -1.0, 'p_alpha': 0.0, 'r_alpha': 0.0}

        # test of normal call
        obj = TimeSeriesFScore(**test_case_1)
        self.assertEqual(obj.beta, test_case_1['beta'])
        self.assertEqual(obj.p_alpha, test_case_1['p_alpha'])
        self.assertEqual(obj.r_alpha, test_case_1['r_alpha'])

        # test of the invalid beta
        with self.assertRaises(Exception):
            obj = TimeSeriesFScore(**test_case_2)

    def test_fscore_function(self):
        """Teest of ts_fscore function.
        """

        real = np.array([1, 1, 1, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        score = ts_fscore(real, pred)
        self.assertEqual(score, 0.5)

    def test_fscore_function_with_list(self):
        """Teest of ts_fscore function with list type arguments.
        """

        real = [1, 1, 1, 0, 0]
        pred = [0, 1, 0, 0, 0]

        score = ts_fscore(real, pred)
        self.assertEqual(score, 0.5)
