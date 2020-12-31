import unittest
import numpy as np

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.precision import TimeSeriesPrecision
from prts import ts_precision


class TestPrecision(unittest.TestCase):
    def test_PrecisionClass_inherited_BaseTimeSeriesMetrics(self):
        """ Check if it inherits BaseTimeSeriesMetircs.
        """
        obj = TimeSeriesPrecision()
        self.assertTrue(
            isinstance(obj, BaseTimeSeriesMetrics)
        )

    def test_PrecisionClass_init(self):
        """ Test of init function.
        """

        test_case_1 = {'alpha': 0.0, 'cardinality': 'one', 'bias': 'flat'}
        test_case_2 = {'alpha': 0.0, 'cardinality': 'one', 'bias': None}
        test_case_3 = {'alpha': 10.0, 'cardinality': 'one', 'bias': 'flat'}

        # test of the normal call
        obj = TimeSeriesPrecision(**test_case_1)
        self.assertEqual(obj.alpha, test_case_1['alpha'])
        self.assertEqual(obj.cardinality, test_case_1['cardinality'])
        self.assertEqual(obj.bias, test_case_1['bias'])

        # test of the invalid bias
        with self.assertRaises(Exception):
            obj = TimeSeriesPrecision(**test_case_2)

        # test of the invalid alpha
        with self.assertRaises(Exception):
            obj = TimeSeriesPrecision(**test_case_3)

    def test_PrecisionClass_score(self):
        """Test of score function.
        """

        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        obj = TimeSeriesPrecision()

        score = obj.score(real, pred)
        self.assertEqual(score, 1.0)


    def test_precision_function(self):
        """Teest of ts_precision function.
        """

        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        score = ts_precision(real, pred)
        self.assertEqual(score, 1.0)

    def test_precision_function_with_list(self):
        """Teet of ts_precision function with list type arguments.
        """

        real = [1, 1, 0, 0, 0]
        pred = [0, 1, 0, 0, 0]

        score = ts_precision(real, pred)
        self.assertEqual(score, 1.0)

    def test_precision_function_with_invalid_alpha(self):
        """Test of ts_precision function with invalid alpha
        """

        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        with self.assertRaises(Exception):
            ts_precision(real, pred, alpha=10)

        with self.assertRaises(Exception):
            ts_precision(real, pred, alpha=-1)

    def test_precision_function_with_invalid_bias(self):
        """Test of ts_precision function with invalid bias
        """

        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        with self.assertRaises(Exception):
            ts_precision(real, pred, bias=None)

        with self.assertRaises(Exception):
            ts_precision(real, pred, bias="Invalid")