# from kaitorch.core import Scalar


__all__ = ['mse', 'binary_crossentropy', 'categorical_crossentropy']


def mse():
    return MeanSquaredError()


def binary_crossentropy():
    return BinaryCrossentropy()


def categorical_crossentropy():
    return CategoricalCrossentropy()


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


class BinaryCrossentropy:
    def __init__(self):
        pass

    def __call__(self, y, y_pred):

        loss = 0.0
        for i in range(len(y)):
            if y[i] == 1:
                loss += -(y_pred[i]).log()
            elif y[i] == 0:
                loss += -(1 - y_pred[i]).log()
        binary_crossentropy_loss = loss / len(y)
        return binary_crossentropy_loss

    def __repr__(self):
        return 'BinaryCrossentropy()'


class CategoricalCrossentropy:
    def __init__(self):
        pass

    def __call__(self, y, y_pred):
        loss = 0.0
        for i in range(len(y)):
            for j in range(len(y[i])):
                if y[i][j] == 1:
                    loss += -(y_pred[i][j]).log()
                elif y[i][j] == 0:
                    loss += -(1 - y_pred[i][j]).log()
        categorical_crossentropy_loss = loss / len(y)
        return categorical_crossentropy_loss

    def __repr__(self):
        return 'CategoricalCrossEntropy()'
