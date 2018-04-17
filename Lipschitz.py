import numpy
from typing import List, Callable, Generic, TypeVar, Tuple

T = TypeVar('T')


def gen_lipschitz(xk: Tuple[float, float],
                  uk: float,
                  l: float = -1.0) -> Callable[[Tuple[float, float]], float]:
    return lambda x: uk + l * numpy.sqrt((xk[0] - x[0]) ** 2 + (xk[1] - x[1]) ** 2)


def max_function(fs: List[Callable[[T], float]]) -> Callable[[T], float]:
    return lambda x: max(map(lambda f: f(x), fs))


def gen_u_star(xks: List[Tuple[float, float]],
               uks: List[float]) -> Callable[[Tuple[float, float]], float]:
    fs = [gen_lipschitz(xks[i], uks[i]) for i in range(len(xks))]
    return max_function(fs)


def check_lip(xks: List[Tuple[float, float]], uks: List[float], l: float = -1.0) -> bool:
    if len(xks) != len(uks):
        raise Exception
    for i in range(len(xks)):
        for j in range(i + 1, len(xks)):
            d = numpy.sqrt((xks[i][0] - xks[j][0]) ** 2 + (xks[i][1] - xks[j][1]) ** 2)
            if abs((uks[i] - uks[j]) * l) > abs(d):
                return False
    return True
