from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from scipy.special import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class GeometricFamily(DistributionFamily):
    def __init__(self, p) -> None:
        super().__init__(numpy_random_generator_method = nprd.geometric , θ ={"p": p})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        p = θ[0]    
        proba = ((1-p)**(x-1)) * p
        return proba