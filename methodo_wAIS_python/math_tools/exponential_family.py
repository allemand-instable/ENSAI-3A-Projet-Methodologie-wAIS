from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from math import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class ExponentialFamily(DistributionFamily):
    def __init__(self, β) -> None:
        super().__init__(numpy_random_generator_method = nprd.exponential, θ ={"scale" : β})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        β = θ[0]
        proba = (1/β)*np.exp(-(1/β)*x)
    
        return proba