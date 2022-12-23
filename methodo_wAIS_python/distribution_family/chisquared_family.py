from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from math import factorial
from utils.log import logstr
from logging import info, debug, warn, error
from scipy.special import factorial

class ChiSquareFamily(DistributionFamily):
    def __init__(self, k) -> None:
        super().__init__(numpy_random_generator_method = nprd.chisquare, θ ={"df" : k})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        k = θ[0]
        proba = (1 / ((2**(k/2))*factorial((k/2)-1)) 
        * (x**(k/2-1))*np.exp(-x/2))
    
        return proba