__all__ = ['sigmoid', 'tanh', 'ReLU', 'LeakyReLU', 'ELU', 'swish', 'softmax']

import math
import warnings

import kaitorch.functional as F
from kaitorch.core import Scalar


def sigmoid(self):

    def _forward():
        y = F.sigmoid(self.data)
        return Scalar(y, (self, ), 'sigmoid')
    out = _forward()

    def _backward():
        self.grad += F.d_sigmoid(out.data) * out.grad
    out._backward = _backward

    return out


def tanh(self):

    def _forward():
        y = F.tanh(self.data)
        return Scalar(y, (self, ), 'tanh')
    out = _forward()

    def _backward():
        self.grad += F.d_tanh(out.data) * out.grad
    out._backward = _backward

    return out


def swish(self, beta=None):

    if beta is None:
        beta = 1
        warnings.warn('Parameter {beta} not specified, using default value 1')

    def _forward():
        y = F.swish(self.data, beta)
        return Scalar(y, (self, ), 'swish')
    out = _forward()

    def _backward():
        self.grad += F.d_swish(out.data, beta) * out.grad
    out._backward = _backward

    return out


def ReLU(self):

    def _forward():
        y = F.ReLU(self.data)
        return Scalar(y, (self, ), 'ReLU')
    out = _forward()

    def _backward():
        self.grad += F.d_ReLU(out.data) * out.grad
    out._backward = _backward

    return out


def LeakyReLU(self, alpha=None):

    if alpha is None:
        alpha = 0.01
        warnings.warn(
            'Parameter {alpha} not specified, using default value 0.01'
        )

    def _forward():
        y = F.LeakyReLU(self.data, alpha)
        return Scalar(y, (self, ), 'LeakyReLU')
    out = _forward()

    def _backward():
        self.grad += F.d_LeakyReLU(out.data, alpha) * out.grad
    out._backward = _backward

    return out


def ELU(self, alpha=None):

    if alpha is None:
        alpha = 0.01
        warnings.warn(
            'Parameter {alpha} not specified, using default value 0.01'
        )

    def _forward():
        a = self.data
        y = alpha * (math.exp(a) - 1) if a < 0 else a
        return Scalar(y, (self, ), 'ELU')
    out = _forward()

    def _backward():
        self.grad += F.d_ELU(out.data, alpha) * out.grad
    out._backward = _backward

    return out


def softmax(ins: list):

    exps = [n.exp() for n in ins]
    sums = sum([n.data for n in exps])
    outs = [n/sums for n in exps]
    return outs
