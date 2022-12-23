from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np
from scipy.special import factorial

from utils.log import logstr
from logging import info, debug, warn, error


class PoissonFamily(DistributionFamily):
    def __init__(self, λ) -> None:
        super().__init__(numpy_random_generator_method = nprd.poisson, θ ={"lam" : λ})
    
    @staticmethod
    def density_fcn(x : int, θ) -> float:
        
        λ = θ[0]
        proba = ((λ**x) / factorial(x)) * np.exp(-λ)
        return proba