import unittest

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TestPrecision(unittest.TestCase):
    def test_InterfaceTimeSeriesMetricsClass_udf_gamma(self):
        """ Check the value of udf_gamma.
        """
        obj = InterfaceTimeSeriesMetrics()
        self.assertEqual(obj._udf_gamma(), 1.0)

    def test_InterfaceTimeSeriesMetricsClass_gamma_select(self):
        """ Check the value of gamma_select.
        """
        obj = InterfaceTimeSeriesMetrics()
        self.assertEqual(obj._gamma_select("one", 1), 1.0)
        self.assertEqual(obj._gamma_select("reciprocal", 2), 0.5)
        self.assertEqual(obj._gamma_select("udf_gamma", 2), 1.0)
        self.assertEqual(obj._gamma_select("udf_gamma", 1), 1.0)

        # TODO: assertRaisesだとValueErrorのチェックができなさそう?
        # with self.assertRaises(ValueError):
            # _ = obj._gamma_select("two", 1)

if __name__ == '__main__':
    unittest.main()
