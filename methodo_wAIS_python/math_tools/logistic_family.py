from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from scipy.special import factorial
from utils.log import logstr
from logging import info, debug, warn, error


class LogisticFamily(DistributionFamily):
    def __init__(self, μ, Σ) -> None:
        super().__init__(numpy_random_generator_method = nprd.logistic , θ ={"loc": μ, "scale": Σ})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        μ = θ[0]    
        Σ = θ[1]
        proba = np.exp(-(x-μ)/Σ)/(Σ*(1+np.exp(-(x-μ)/Σ))**2)
        return proba