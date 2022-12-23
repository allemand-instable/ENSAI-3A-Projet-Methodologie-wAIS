from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np
from scipy.special import factorial

from utils.log import logstr
from logging import info, debug, warn, error

from typing import Any, AnyStr, Self, Optional, Dict
from numpy.typing import NDArray

class MultinomialFamily(DistributionFamily):
    def __init__(self, n, π) -> None:
        super().__init__(numpy_random_generator_method = nprd.multinomial , θ ={"n": n, "pvals": π})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        n = θ[0]
        π = θ[1]

        x_factorial = factorial(x)
        product_x_factorial = 1
        product_pi = 1
        for i in range(len(x)):
            product_x_factorial *= x_factorial[i]
            product_pi *= π**x[i]

        proba = (factorial(n)/product_x_factorial) * product_pi
        return proba