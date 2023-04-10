import math
import kaitorch.activations as A

activations = A.__all__
derivatives = [f'd_{activation}' for activation in activations]

__all__ = [x for y in zip(activations, derivatives) for x in y]


def sigmoid(x):

    # Operation: y = 1 / (1 + (e ** -x))
    if x < 0:
        out = math.exp(x) / (1 + math.exp(x))
    else:
        out = 1 / (1 + math.exp(-x))
    return out


def d_sigmoid(x):

    # Derivative: dy/dx = sigmoid(x) * (1 - sigmoid(x))
    # Chain Rule: dL/dx = dL/dy * dy/dx
    #                   = dL/dy * (sigmoid(x) * (1 - sigmoid(x)))
    out = sigmoid(x) * (1 - sigmoid(x))
    return out


def tanh(x):

    # Operation: y = (e ** (2 * x) - 1) / (e ** (2 * x) + 1)
    out = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    return out


def d_tanh(x):

    # Derivative: dy/dx = (1 - (tanh(x) ** 2))
    # Chain Rule: dL/dx = dL/dy * dy/dx
    #                   = dL/dy * (1 - (tanh(x) ** 2))
    out = (1 - (tanh(x) ** 2))
    return out


def ReLU(x):

    # Operation: y = x if x ≥ 0
    #                0 if x < 0
    out = 0 if x < 0 else x
    return out


def d_ReLU(x):

    # Derivative: dy/dx = 1 if x ≥ 0
    #                     0 if x < 0
    # Chain Rule: dL/dx = dL/dy * dy/dx
    #                   = dL/dy * 1 if x ≥ 0
    #                     dL/dy * 0 if x < 0
    out = (x > 0) * 1
    return out


def LeakyReLU(x, alpha=0.1):

    # Operation: y = x if x ≥ 0
    #                x * alpha if x < 0
    out = x * alpha if x < 0 else x
    return out


def d_LeakyReLU(x, alpha=0.1):

    # Derivative: dy/dx = 1 if x ≥ 0
    #                     alpha if x < 0
    # Chain Rule: dL/dx = dL/dy * dy/dx
    #                   = dL/dy * 1 if x ≥ 0
    #                     dL/dy * alpha if x < 0
    out = alpha if x < 0 else 1
    return out


def ELU(x, alpha=1.0):

    # Operation: y = x if x ≥ 0
    #                alpha * ((e ** x) - 1) if x < 0
    out = alpha * (math.exp(x) - 1) if x < 0 else x
    return out


def d_ELU(x, alpha=1.0):

    # Derivative: dy/dx = 1 if x ≥ 0
    #                     alpha * (e ** x) if x < 0
    # Chain Rule: dL/dx = dL/dy * dy/dx
    #                   = dL/dy * 1 if x ≥ 0
    #                     dL/dy * alpha * (e ** x) if x < 0
    out = (alpha * math.exp(x)) if x < 0 else 1
    return out


def swish(x, beta=1.0):

    # Operation: y = x * sigmoid(β * x)
    out = x * sigmoid(x * beta)
    return out


def d_swish(x, beta=1.0):

    # Deriv: dy/dx = swish(x, β) + sigmoid(β * x) * (1 - swish(x, β))
    # Chain: dL/dx = dL/dy * dy/dx
    #              = dL/dy * swish(x, β) + sigmoid(β * x) * (1 - swish(x, β))
    out = swish(x, beta) + sigmoid(beta * x) * (1 - swish(x, beta))
    return out

