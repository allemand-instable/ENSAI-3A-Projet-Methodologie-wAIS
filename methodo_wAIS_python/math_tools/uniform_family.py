from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from math import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class UniformFamily(DistributionFamily):
    def __init__(self, a, b) -> None:
        super().__init__(numpy_random_generator_method = nprd.uniform, θ ={"low": a, "high": b})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        a = θ[0]
        b = θ[1]
        
        if x < a and x >= b:
            proba = 0
        else:
            proba = 1/(b-a)
        return proba