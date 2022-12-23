from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from math import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class WeibullFamily(DistributionFamily):
    def __init__(self, a) -> None:
        super().__init__(numpy_random_generator_method = nprd.weibull, θ ={"a" : a})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        a = θ[0]
        proba = a*(x**(a-1))*np.exp(-(x**a))
    
        return proba