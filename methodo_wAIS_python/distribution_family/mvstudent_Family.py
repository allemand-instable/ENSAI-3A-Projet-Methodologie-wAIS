from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from scipy.stats import multivariate_t
from scipy.special import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class StudentFamily(DistributionFamily):
    def __init__(self, k) -> None:
        super().__init__(numpy_random_generator_method = nprd.student, θ ={"df" : k})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        k = θ[0]
        proba = factorial(((k + 1)/2)-1) / (np.sqrt(np.pi*k) * 
        factorial((k/2)-1)) * (1 + (x**2)/k)**(-(k+1)/2)
    
        return proba