from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from utils.log import logstr
from logging import info, debug, warn, error


class MVNormalFamily(DistributionFamily):
    def __init__(self, μ ,Σ) -> None:
        super().__init__(numpy_random_generator_method = nprd.multivariate_normal , θ ={"loc" : μ, "scale" : Σ})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        μ = θ[0]
        Σ = θ[1]
        
        expo = np.exp(-0.5*np.matmul(np.matmul((x-μ), np.linalg.inv(Σ)), (x-μ)))
        cste_norm = np.sqrt(2*np.pi)**(-len(μ) * np.linalg.inv(np.linalg.det(Σ)))
        
        return expo/cste_norm
