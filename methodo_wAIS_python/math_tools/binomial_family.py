from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from scipy.special import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class BinomialFamily(DistributionFamily):
    def __init__(self, n, p) -> None:
        super().__init__(numpy_random_generator_method = nprd.binomial , θ ={"n" : n, "p" : p})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        def binomial_coefficient(n: int, m: int):
            return (factorial(n) /
                (factorial(m) * factorial(n-m)))

        n = θ[0]
        p = θ[1]

        proba = binomial_coefficient(n = n, m = x) * p**x * (1-p)**(n-x)
        
        return proba