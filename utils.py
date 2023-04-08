__all__ = ['wrap', 'unwrap', 'ffill']


def wrap(x):
    if isinstance(x[0], int) or isinstance(x[0], float):
        x = [x]
    return x, len(x[0])


def unwrap(out):

    if isinstance(out, list):
        if len(out) == 1:
            out = out[0]
            out = unwrap(out)
    return out


def ffill(x: list):
    for i in range(1, len(x)-1):
        if x[i] is None:
            x[i] = x[i-1]
    return x
