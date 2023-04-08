import math
from kaitorch.utils import unwrap
from kaitorch.core import Scalar
import numpy as np

__all__ = ['mse', 'binary_crossentropy', 'categorical_crossentropy']


def mse():
    return MeanSquaredError()


def binary_crossentropy():
    return BinaryCrossEntropy()


def categorical_crossentropy():
    return CategoricalCrossEntropy()


class MeanSquaredError:

    def __init__(self):
        pass

    def __call__(self, y: list, y_pred: list):

        pred_length = len(y)
        squared_error = sum(
            (y_record - y_out)**2 for y_record, y_out in zip(y, y_pred)
        )
        mean_squared_error = squared_error/pred_length

        return mean_squared_error

    def __repr__(self):
        return 'MeanSquaredError()'


class BinaryCrossEntropy:
    def __init__(self):
        pass

    def __call__(self, y: list, y_pred: list):

        pred_length = len(y_pred)

        term_0 = sum(
            (1-y_) * (1-y_p+1e-8).log() for y_, y_p in zip(y, y_pred)
        )
        term_1 = sum(
            (y_) * (y_p+1e-8).log() for y_, y_p in zip(y, y_pred)
        )
        binary_crossentropy = -1 * (term_0 + term_1) / pred_length

        return binary_crossentropy

    def __repr__(self):
        return 'BinaryCrossEntropy()'


class CategoricalCrossEntropy:
    def __init__(self):
        pass

    def __call__(self, y:list, y_pred: list):

        pred_length = len(y)

        term_sums = 0
        for i in pred_length:
            term_sums = sum(y_ * math.log(y_p.data) for y_, y_p in zip(y, y_pred))

        categorical_crossentropy = -1 * term_sums / pred_length

        return unwrap(categorical_crossentropy)

    def __repr__(self):
        return 'CategoricalCrossEntropy()'
