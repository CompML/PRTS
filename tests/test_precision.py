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

        # test normal case
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        obj = TimeSeriesPrecision()

        score = obj.score(real, pred)
        self.assertEqual(score, 1.0)

        # test invalid inputs
        real = None
        pred = np.array([0, 1, 0, 0, 0])
        with self.assertRaises(Exception):
            score = obj.score(real, pred)

        real = np.array([1, 1, 0, 0, 0])
        pred = None
        with self.assertRaises(Exception):
            score = obj.score(real, pred)

    def test_PrecisionClass_update_precision(self):
        """Test of _update_precision function.
        """

        # test of the normal case
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        obj = TimeSeriesPrecision()
        real_anomalies, predicted_anomalies = obj._prepare_data(real, pred)

        score = obj._update_precision(real_anomalies, predicted_anomalies)
        self.assertEqual(score, 1.0)

        # test of the empty case
        empty_real = np.array([])
        empty_pred = np.array([])

        score = obj._update_precision(empty_real, empty_pred)
        self.assertEqual(score, 0.0)

    def test_precision_function(self):
        """Teest of ts_precision function.
        """

        # test case1
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])

        score = ts_precision(real, pred)
        self.assertEqual(score, 1.0)

        # test case2
        real = np.array([1, 1, 0, 0, 0])
        pred = np.array([0, 0, 1, 1, 1])

        score = ts_precision(real, pred)
        self.assertEqual(score, 0.0)

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

    def test_precision_function_with_all_zeros(self):
        """Test of ts_precision function with all zero values
        """

        real = np.array([0, 0, 0, 0, 0])
        pred = np.array([0, 0, 0, 0, 0])

        with self.assertRaises(Exception):
            ts_precision(real, pred)

    def test_precision_function_with_all_ones(self):
        """Test of ts_precision function with all zero values
        """

        real = np.array([1, 1, 1, 1, 1])
        pred = np.array([1, 1, 1, 1, 1])

        self.assertEqual(ts_precision(real, pred), 1.0)

