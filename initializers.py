import math
import random
import kaitorch.activations as A
from kaitorch.core import Optimizer


__all__ = [
    'glorot_uniform',
    'glorot_normal',
    'he_uniform',
    'he_normal',
    'lecun_uniform',
    'lecun_normal',
    'random_uniform',
    'random_normal'
]


def glorot_uniform():
    return GlorotUniform()


def glorot_normal():
    return GlorotNormal()


def he_uniform():
    return HeUniform()


def he_normal():
    return HeNormal()


def lecun_uniform():
    return LecunUniform()


def lecun_normal():
    return LecunNormal()


def random_uniform():
    return RandomUniform()


def random_normal():
    return RandomNormal()


class GlorotUniform(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, nout):

        glorot_uniform_sample = random.uniform(
            a=-math.sqrt(6 / (nin + nout)),
            b=math.sqrt(6 / (nin + nout))
        )
        return glorot_uniform_sample

    def __repr__(self):
        return 'glorot_uniform'


class GlorotNormal(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, nout):

        glorot_normal_sample = random.gauss(
            mu=0,
            sigma=math.sqrt(2 / (nin + nout))
        )
        return glorot_normal_sample

    def __repr__(self):
        return 'glorot_normal'


class HeUniform(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, nout):

        he_uniform_sample = random.uniform(
            a=-math.sqrt(6 / nin),
            b=math.sqrt(6 / nin)
        )
        return he_uniform_sample

    def __repr__(self):
        return 'he_uniform'


class HeNormal(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, *args, **kwargs):

        he_normal_sample = random.gauss(
            mu=0,
            sigma=math.sqrt(2 / nin)
        )
        return he_normal_sample

    def __repr__(self):
        return 'he_normal'


class LecunUniform(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, nout):

        lecun_uniform_sample = random.uniform(
            a=-math.sqrt(3 / nin),
            b=math.sqrt(3 / nin)
        )
        return lecun_uniform_sample

    def __repr__(self):
        return 'lecun_uniform'


class LecunNormal(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, *args, **kwargs):

        lecun_normal_sample = random.gauss(
            mu=0,
            sigma=math.sqrt(1 / nin)
        )
        return lecun_normal_sample

    def __repr__(self):
        return 'lecun_normal'


class RandomUniform(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, nout):

        random_uniform_sample = random.uniform(
            a=-0.05,
            b=0.05
        )
        return random_uniform_sample

    def __repr__(self):
        return 'random_uniform'


class RandomNormal(Optimizer):

    def __init__(self):
        pass

    def __call__(self, nin, *args, **kwargs):

        random_normal_sample = random.gauss(
            mu=0,
            sigma=0.05
        )
        return random_normal_sample

    def __repr__(self):
        return 'random_normal'
