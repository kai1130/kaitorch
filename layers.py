import random
from kaitorch.utils import unwrap
from kaitorch.core import Scalar, Module


class Dense(Module):

    class Node:

        def __init__(self, nin, activation):
            self.w = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
            self.b = Scalar(random.uniform(-1, 1))
            self.a = activation

        def __call__(self, x):
            x = [x] if isinstance(x, Scalar) else x
            signal = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            if self.a:
                signal = signal.activation(self.a)
            return signal

        def parameters(self):
            return self.w + [self.b]

    def __init__(self, nouts, activation=None):
        self.nins = None
        self.nouts = nouts
        self.nodes = None
        self.activation = activation

    def __repr__(self):
        if self.activation is not None:
            return f'Dense(units={self.nouts}, activation={self.activation})'
        else:
            return f'Dense(units={self.nouts})'

    def __build__(self, nins):
        self.nins = nins
        self.nodes = [
            self.Node(self.nins, self.activation) for _ in range(self.nouts)
        ]

    def __call__(self, x):
        outs = [n(x) for n in self.nodes]
        return unwrap(outs)

    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]


class Dropout(Module):

    class Node:

        def __init__(self, dropout_rate: float):
            self.p = 1 - dropout_rate

        def __call__(self, x, train):
            if train is True:
                mask = 1 if random.random() <= self.p else 0
                return x * (1/self.p) * mask
            else:
                return x

        def parameters(self):
            return []

    def __init__(self, dropout_rate: float = 0.5):

        self.nins = None
        self.nouts = None
        self.nodes = None
        self.p = dropout_rate

        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

    def __repr__(self):
        return f'Dropout(dropout_rate={self.p})'

    def __build__(self, nins):
        self.nins = nins
        self.nouts = nins
        self.nodes = [self.Node(self.p) for _ in range(self.nins)]

    def __call__(self, x, train):
        outs = [n(xi, train) for n, xi in zip(self.nodes, x)]
        return unwrap(outs)

    def parameters(self):
        return [p for node in self.nodes for p in node.parameters()]
