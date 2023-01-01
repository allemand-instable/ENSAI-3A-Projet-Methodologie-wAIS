from typing import Any
from distribution_family.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

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

    def sample(self,n) -> list[Any] | None:
        """renvoie une liste de n tirages selon la loi de distribution de l'objet"""
        print(self.parameters)
        if type(self.parameters) is dict :
            return list(self.generator_method( **self.parameters ,size = n))
        elif type(self.parameters) is list :
            return list(self.generator_method( *self.parameters ,size = n))
        elif type(self.parameters) is np.ndarray :
            if self.parameters.shape[0] >= 2 :
                return list(self.generator_method( *self.parameters ,size = n))
            else :
                return list(self.generator_method( self.parameters ,size = n))