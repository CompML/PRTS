from prts.time_series_metrics.fscore import TimeSeriesFScore
from prts.time_series_metrics.precision import TimeSeriesPrecision
from prts.time_series_metrics.recall import TimeSeriesRecall


def ts_precision(real, pred, alpha=0.0, cardinality="one", bias="flat"):
    """Compute the time series precision.

    The time series precision is the average of "Precision_Ti", where "Precision_Ti" is
    the precision score of each predicted anomaly range.
    "Precision_Ti" for a single predicted anomaly range is calculated by the following formula.
        Precision_Ti = α x ExistenceReward + (1 - α) x OverlapReward , where 0 ≤ α ≤ 1
    α represents the relative importance of rewarding existence, whereas
    (1 − α) represents the relative importance of rewarding size, position, and cardinality.

    "ExistenceReward" is 1 if a real anomaly range has overlap with even a single point of
    the predicted anomaly range, 0 otherwise.
    Note: For prediction, there is no need for an existence reward, since precision by definition
          emphasizes prediction quality, and existence by itself is too low a bar for judging
          the quality of a prediction (i.e., α = 0).

    "OverlapReward" is calculated by the following formula.
        OverlapReward = CardinalityFactor x Sum of ω
    "CardinalityFactor" is 1 if the predicted anomaly range overlaps with only one real anomaly range.
    Otherwise it receives 0 ≤ γ() ≤ 1 defined by the application.
    "CardinalityFactor" serves as a scaling factor for the rewards "ω"s, which is earned from overlap
    size and position.
    In determing "ω", we consider the size of the correctly predicted portion of an predicted anomaly
    range and the relative position of the correctly predicted portion of an predicted anomaly range.

    Args:
        real: np.ndarray
            One-dimensional array of correct answers with values of 1 or 0.
        pred: np.ndarray
            One-dimensional array of predicted answers with values of 1 or 0.
        alpha: float, default=0.0
            Relative importance of existence reward. 0 ≤ alpha ≤ 1.
        cardinality: string, default="one"
            Cardinality type. This should be "one", "reciprocal" or "udf_gamma".
        bias: string, default="flat"
            Positional bias. This should be "flat", "front", "middle", or "back"

    Returns:
        float: precision.score
    """
    precision = TimeSeriesPrecision(alpha, cardinality, bias)
    return precision.score(real, pred)


def ts_recall(real, pred, alpha=0.0, cardinality="one", bias="flat"):
    """Compute the time series recall.

    The time series recall is the average of "Recall_Ti", where "Recall_Ti" is
    the recall score of each real anomaly range.
    "Recall_Ti" for a single real anomaly range is calculated by the following formula.
        Recall_Ti = α x ExistenceReward + (1 - α) x OverlapReward , where 0 ≤ α ≤ 1
    α represents the relative importance of rewarding existence, whereas
    (1 − α) represents the relative importance of rewarding size, position, and cardinality.

    "ExistenceReward" is 1 if a prediction captures even a single point of the real anomaly range, 0 otherwise.

    "OverlapReward" is calculated by the following formula.
        OverlapReward = CardinalityFactor x Sum of ω
    "CardinalityFactor" is 1 if the real anomaly range overlaps with only one predicted anomaly range.
    Otherwise it receives 0 ≤ γ() ≤ 1 defined by the application.
    "CardinalityFactor" serves as a scaling factor for the rewards "ω"s, which is earned from overlap
    size and position.
    In determing "ω", we consider the size of the correctly predicted portion of the real anomaly range
    and the relative
    position of the correctly predicted portion of the real anomaly range.

    Args:
        real: np.ndarray
            One-dimensional array of correct answers with values of 1 or 0.
        pred: np.ndarray
            One-dimensional array of predicted answers with values of 1 or 0.
        alpha: float, default=0.0
            Relative importance of existence reward. 0 ≤ alpha ≤ 1.
        cardinality: string, default="one"
            Cardinality type. This should be "one", "reciprocal" or "udf_gamma".
        bias: string, default="flat"
            Positional bias. This should be "flat", "front", "middle", or "back"

    Returns:
        float: recall.score
    """
    recall = TimeSeriesRecall(alpha, cardinality, bias)
    return recall.score(real, pred)


def ts_fscore(real, pred, beta=1.0, p_alpha=0.0, r_alpha=0.0, cardinality="one", p_bias="flat", r_bias="flat"):
    """Compute the time series f-score

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.
    The beta parameter determines the weight of recall in the combined score.
    beta < 1 lends more weight to precision, while beta > 1 favors recall
    (beta -> 0 considers only precision, beta -> +inf only recall).

    Args:
        real: np.ndarray
            One-dimensional array of correct answers with values of 1 or 0.
        pred: np.ndarray
            One-dimensional array of predicted answers with values of 1 or 0.
        p_alpha: float, default=0.0
            Relative importance of existence reward for precision. 0 ≤ alpha ≤ 1.
        r_alpha: float, default=0.0
            Relative importance of existence reward for recall. 0 ≤ alpha ≤ 1.
        cardinality: string, default="one"
            Cardinality type. This should be "one", "reciprocal" or "udf_gamma".
        p_bias: string, default="flat"
            Positional bias for precision. This should be "flat", "front", "middle", or "back"
        r_bias: string, default="flat"
            Positional bias for recall. This should be "flat", "front", "middle", or "back"

    Returns:
        float: f.score
    """

    fscore = TimeSeriesFScore(beta, p_alpha, r_alpha, cardinality, p_bias, r_bias)
    return fscore.score(real, pred)
