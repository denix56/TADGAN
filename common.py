import numpy as np
from sklearn import metrics


def _overlap(expected, observed):
    first = expected[0] - observed[1]
    second = expected[1] - observed[0]
    return first * second < 0


def _any_overlap(part, intervals):
    for interval in intervals:
        if _overlap(part, interval):
            return 1

    return 0


def _weighted_segment(expected, observed, _partition, start=None, end=None):
    expected, observed, weights = _partition(expected, observed, start, end)

    return metrics.confusion_matrix(
        expected, observed, sample_weight=weights, labels=[0, 1]).ravel()


def _accuracy(expected, observed, data, start, end, cm):
    tn, fp, fn, tp = cm(expected, observed, data, start, end)

    if tn is None:
        raise ValueError("Cannot obtain accuracy score for overlap segment method.")

    return (tp + tn) / (tn + fp + fn + tp)


def _precision(expected, observed, data, start, end, cm):
    tn, fp, fn, tp = cm(expected, observed, data, start, end)

    try:
        return tp / (tp + fp)

    except ZeroDivisionError as ex:
        print(
            'Evaluation exception {} (tp {}/ fp {}).'.format(ex, tp, fp))

        return 0


def _recall(expected, observed, data, start, end, cm):
    tn, fp, fn, tp = cm(expected, observed, data, start, end)

    try:
        return tp / (tp + fn)

    except ZeroDivisionError as ex:
        print(
            'Evaluation exception {} (tp {}/ fn {}).'.format(ex, tp, fn))

        return 0


def _f1_score(expected, observed, data, start, end, cm, beta):
    precision = _precision(expected, observed, data, start, end, cm)
    recall = _recall(expected, observed, data, start, end, cm)

    try:
        return (1+beta**2) * (precision * recall) / (beta**2 * precision + recall)

    except ZeroDivisionError:
        print(
            'Invalid value encountered for precision {}/ recall {}.'.format(precision, recall))

        return 0
