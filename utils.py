__all__ = ['wrap', 'unwrap', 'ffill', 'as_onehot']


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


def as_onehot(y_pred: list):
    max_pred = max(y_pred)
    return [1 if x == max_pred else 0 for x in y_pred]
