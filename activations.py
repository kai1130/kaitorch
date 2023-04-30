__all__ = ['sigmoid', 'tanh', 'ReLU', 'LeakyReLU', 'ELU', 'swish', 'softmax']

import warnings

import kaitorch.functional as F
from kaitorch.core import Scalar


class sigmoid:

    def __init__(self):
        pass

    def __repr__(self):
        return 'sigmoid'

    def __call__(self, scalar):

        def _forward():
            y = F.sigmoid(scalar.data)
            return Scalar(y, (scalar, ), 'sigmoid')
        out = _forward()

        def _backward():
            scalar.grad += F.d_sigmoid(out.data) * out.grad
        out._backward = _backward

        return out


class tanh:

    def __init__(self):
        pass

    def __repr__(self):
        return 'tanh'

    def __call__(self, scalar):

        def _forward():
            y = F.tanh(scalar.data)
            return Scalar(y, (scalar, ), 'tanh')
        out = _forward()

        def _backward():
            scalar.grad += F.d_tanh(out.data) * out.grad
        out._backward = _backward

        return out


class swish:

    def __init__(self, beta=None):

        self.beta = beta
        if beta is None:
            self.beta = 1
            warnings.warn('Parameter {beta} not specified, using default value 1')

    def __repr__(self):
        return f'swish(beta={self.beta})'

    def __call__(self, scalar):

        def _forward():
            y = F.swish(scalar.data, self.beta)
            return Scalar(y, (scalar, ), 'swish')
        out = _forward()

        def _backward():
            scalar.grad += F.d_swish(out.data, self.beta) * out.grad
        out._backward = _backward

        return out


class ReLU:

    def __init__(self):
        pass

    def __repr__(self):
        return 'ReLU'

    def __call__(self, scalar):

        def _forward():
            y = F.ReLU(scalar.data)
            return Scalar(y, (scalar, ), 'ReLU')
        out = _forward()

        def _backward():
            scalar.grad += F.d_ReLU(out.data) * out.grad
        out._backward = _backward

        return out


class LeakyReLU:

    def __init__(self, alpha=None):

        self.alpha = alpha
        if alpha is None:
            self.alpha = 0.01
            warnings.warn('Parameter {alpha} not specified, using default value 0.01')

    def __repr__(self):
        return f'LeakyReLU(alpha={self.alpha})'

    def __call__(self, scalar):

        def _forward():
            y = F.LeakyReLU(scalar.data, self.alpha)
            return Scalar(y, (scalar, ), 'LeakyReLU')
        out = _forward()

        def _backward():
            scalar.grad += F.d_LeakyReLU(out.data, self.alpha) * out.grad
        out._backward = _backward

        return out


class ELU:

    def __init__(self, alpha=None):

        self.alpha = alpha
        if alpha is None:
            self.alpha = 0.01
            warnings.warn('Parameter {alpha} not specified, using default value 0.01')

    def __repr__(self):
        return f'ELU(alpha={self.alpha})'

    def __call__(self, scalar):

        def _forward():
            y = F.ELU(scalar.data, self.alpha)
            return Scalar(y, (scalar, ), 'ELU')
        out = _forward()

        def _backward():
            scalar.grad += F.d_ELU(out.data, self.alpha) * out.grad
        out._backward = _backward

        return out


def softmax(ins: list):

    exps = [n.exp() for n in ins]
    sums = sum([n.data for n in exps])
    outs = [n/sums for n in exps]
    return outs
