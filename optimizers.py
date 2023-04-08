from kaitorch.core import Scalar, Optimizer

__all__ = ['SGD', 'Momentum', 'Nesterov']


# Stochastic Gradient Descent
class SGD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, p: Scalar):
        step = -1 * (self.lr * p.grad)
        p.data += step

    def __repr__(self):
        return f'SGD(lr={self.lr})'


# Stochastic Gradient Descent with Momentum
class Momentum(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, p: Scalar):
        if not hasattr(p, 'step'):
            p.step = 0.0
        p.step = (self.momentum * p.step) - (self.lr * p.grad)
        p.data += p.step

    def __repr__(self):
        return f'Momentum(lr={self.lr}, Momentum={self.momentum})'


# Stochastic Gradient Descent with Nesterov Momentum
class Nesterov(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, p: Scalar):
        if not hasattr(p, 'step'):
            p.step = 0.0
        p.step = (self.momentum * p.step) - (self.lr * p.grad)
        p.data += self.momentum * p.step - (self.lr * p.grad)

    def __repr__(self):
        return f'Nesterov(lr={self.lr}, Momentum={self.momentum})'
