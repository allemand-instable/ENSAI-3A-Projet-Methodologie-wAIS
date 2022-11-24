import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from typing import Callable, Union

from utils.log import logstr
from logging import info, debug, warn, error


class DistributionFamily():
    def __init__(self, numpy_random_generator_method : Callable, θ : dict | list | np.ndarray ) -> None:
                
        self.generator_method : Callable = numpy_random_generator_method
        self.parameters : dict | list | ArrayLike = θ
        
        # check
        if type(self.parameters) not in (list, ArrayLike, dict) :
            raise TypeError("mauvais type de paramètre")
    
    def sample(self,n):
        if type(self.parameters) is dict :
            return list(self.generator_method( **self.parameters ,size = n))
        elif type(self.parameters) is list :
            return list(self.generator_method( *self.parameters ,size = n))
        elif type(self.parameters) is np.ndarray :
            if self.parameters.shape[0] >= 2 :
                return list(self.generator_method( *self.parameters ,size = n))
            else :
                return list(self.generator_method( self.parameters ,size = n))
    
    def update_parameters(self, θ):
        debug(logstr(f"old params : {self.parameters}\ntype : {type(self.parameters)}"))
        self.parameters = θ
        debug(logstr(f"new params : {self.parameters}\ntype : {type(self.parameters)}"))
    
    def density(self, x) -> float:
        if type(self.parameters) is dict :
            return self.density_fcn(x, [*self.parameters.values()])
        if type(self.parameters) is list :
            return self.density_fcn(x, self.parameters)
        if type(self.parameters) is ArrayLike :
            return self.density_fcn(x, self.parameters)
    @staticmethod
    def density_fcn(x, θ) -> float:
        pass
    
    def parameters_list(self):
        if type(self.parameters) is dict :
            return [*self.parameters.values()]
        elif type(self.parameters) in [list, ArrayLike] :
            return self.parameters