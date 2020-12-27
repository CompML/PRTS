from prts.time_series_metrics.precision import TimeSeriesPrecision
from prts.time_series_metrics.recall import TimeSeriesRecall


def ts_precision(real, pred, beta=1.0, alpha=0.0, cardinality="one", bias="flat"):
    precision = TimeSeriesPrecision(beta, alpha, cardinality, bias)
    return precision.score(real, pred)


def ts_recall(real, pred, beta=1.0, alpha=0.0, cardinality="one", bias="flat"):
    recall = TimeSeriesRecall(beta, alpha, cardinality, bias)
    return recall.score(real, pred)
