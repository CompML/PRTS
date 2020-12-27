import unittest

from prts.base.time_series_metrics import BaseTimeSeriesMetrics
from prts.time_series_metrics.precision import TimeSeriesPrecision


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

        test_case_1 = {'beta': 1.0, 'alpha': 0.0, 'cardinality': 'one', 'bias': 'flat'}
        test_case_2 = {'beta': 1.0, 'alpha': 0.0, 'cardinality': 'one', 'bias': None}
        test_case_3 = {'beta': -1, 'alpha': 0.0, 'cardinality': 'one', 'bias': 'flat'}
        test_case_4 = {'beta': 1.0, 'alpha': 10.0, 'cardinality': 'one', 'bias': 'flat'}

        # test of the normal call
        obj = TimeSeriesPrecision(**test_case_1)
        self.assertEqual(obj.beta, test_case_1['beta'])
        self.assertEqual(obj.alpha, test_case_1['alpha'])
        self.assertEqual(obj.cardinality, test_case_1['cardinality'])
        self.assertEqual(obj.bias, test_case_1['bias'])

        # test of the invalid bias
        with self.assertRaises(Exception):
            obj = TimeSeriesPrecision(**test_case_2)

        # test of the invalid beta
        with self.assertRaises(Exception):
            obj = TimeSeriesPrecision(**test_case_3)

        # test of the invalid alpha
        with self.assertRaises(Exception):
            obj = TimeSeriesPrecision(**test_case_4)
