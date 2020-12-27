import unittest

from prts.base.time_series_metrics import BaseTimeSeriesMetrics


class TestPrecision(unittest.TestCase):
    def test_BaseTimeSeriesMetricsClass_udf_gamma(self):
        """ Check the value of udf_gamma.
        """
        obj = BaseTimeSeriesMetrics()
        self.assertEqual(obj._udf_gamma(), 1.0)

    def test_BaseTimeSeriesMetricsClass_gamma_select(self):
        """ Check the value of gamma_select.
        """
        obj = BaseTimeSeriesMetrics()
        self.assertEqual(obj._gamma_select("one", 1), 1.0)
        self.assertEqual(obj._gamma_select("reciprocal", 2), 0.5)
        self.assertEqual(obj._gamma_select("udf_gamma", 2), 1.0)
        self.assertEqual(obj._gamma_select("udf_gamma", 1), 1.0)
        with self.assertRaises(ValueError):
            _ = obj._gamma_select("two", 1)

if __name__ == '__main__':
    unittest.main()
