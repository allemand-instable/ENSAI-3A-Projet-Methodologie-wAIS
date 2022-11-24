import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from typing import Callable, Union

class DistributionFamily():
    def __init__(self, numpy_random_generator_method : Callable, θ : dict | list | ArrayLike ) -> None:
        self.generator_method : Callable = numpy_random_generator_method
        self.parameters : dict | list | ArrayLike = θ
        # check
        if type(self.parameters) not in (dict, list, ArrayLike) :
            raise TypeError("mauvais type de paramètre")
    
    def sample(self,n):
        if type(self.parameters) is dict :
            return list(self.generator_method( **self.parameters ,size = n))
        elif type(self.parameters) is list :
            return list(self.generator_method( *self.parameters ,size = n))
        elif type(self.parameters) is ArrayLike :
            if self.parameters.shape[0] >= 2 :
                return list(self.generator_method( *self.parameters ,size = n))
            else :
                return list(self.generator_method( self.parameters ,size = n))
    
    def update_parameters(self, θ):
        self.parameters = θ
    
    def density(self, x):
        if type(self.parameters) is dict :
            return self.density_fcn(x, [*self.parameters.values()])
        if type(self.parameters) is list :
            return self.density_fcn(x, self.parameters)
        if type(self.parameters) is ArrayLike :
            return self.density_fcn(x, self.parameters)
    @staticmethod
    def density_fcn(x, θ) -> float:
        pass
        