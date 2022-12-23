from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from scipy.special import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class FisherFamily(DistributionFamily):
    def __init__(self, d1, d2) -> None:
        super().__init__(numpy_random_generator_method = nprd.f , θ ={"dfnum" : d1, "dfden": d2})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        d1 = θ[0]
        d2 = θ[1]
    
        def beta_function(z1: int, z2: int) -> float:
            res = (factorial(z1-1) * factorial(z2-1))/ \
                factorial(z1 + z2 - 1)
            return res

            
        proba = (np.sqrt((((d1*x)**d1)*(d2**d1)) / ((d1*x+d2)**(d1+d2))) /
            x*beta_function(d1/2, d2/2))

        return proba