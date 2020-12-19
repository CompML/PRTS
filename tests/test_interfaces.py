import unittest

from prts.interfaces.time_series_metrics import InterfaceTimeSeriesMetrics


class TestPrecision(unittest.TestCase):
    def test_InterfaceTimeSeriesMetricsClass_udf_gamma(self):
        """ Check the value of udf_gamma.
        """
        obj = InterfaceTimeSeriesMetrics()
        self.assertEqual(obj._udf_gamma(), 1.0)


if __name__ == '__main__':
    unittest.main()
