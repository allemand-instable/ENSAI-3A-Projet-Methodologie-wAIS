import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from typing import Callable, Union, Self, Dict, Any, List
from numpy.typing import NDArray

from utils.log import logstr
from logging import info, debug, warn, error

import copy

class DistributionFamily():
    def __init__(self, numpy_random_generator_method : Callable, θ : Dict | List | NDArray ) -> None:
                
        self.generator_method : Callable = numpy_random_generator_method
        
        if len(θ) == 0 :
            raise ValueError("θ must contain values to define the density function from the parametric family")
        else :
            self.parameters : Dict[str, float] | list | NDArray = θ
        
        # check
        if type(self.parameters) not in (list, NDArray, dict) :
            raise TypeError("mauvais type de paramètre")
    
    def sample(self,n) -> list[Any] | None:
        """renvoie une liste de n tirages selon la loi de distribution de l'objet"""
        if type(self.parameters) is dict :
            return list(self.generator_method( **self.parameters ,size = n))
        elif type(self.parameters) is list :
            return list(self.generator_method( *self.parameters ,size = n))
        elif type(self.parameters) is np.ndarray :
            if self.parameters.shape[0] >= 2 :
                return list(self.generator_method( *self.parameters ,size = n))
            else :
                return list(self.generator_method( self.parameters ,size = n))
    
    def update_parameters(self, θ : NDArray | Dict[str, float]) -> None:
        """change les paramètres de la distribution de l'objet concerné"""
        debug(logstr(f"old params : {self.parameters}\ntype : {type(self.parameters)}"))
        self.parameters = θ
        debug(logstr(f"new params : {self.parameters}\ntype : {type(self.parameters)}"))
    
    def density(self, x : float) -> float:
        """evaluates x ↦ f( x | self.parameters )

        Args:
            x (float):  where the density is evaluated

        Returns:
            float: f( x | self.parameters )
        """
        if type(self.parameters) is dict :
            return self.density_fcn(x, np.array([*self.parameters.values()]))
        if type(self.parameters) is list :
            return self.density_fcn(x, np.array(self.parameters))
        if type(self.parameters) is ArrayLike :
            return self.density_fcn(x, np.array(self.parameters))
        else :
            raise TypeError("self.parameters should be a dict, a list, or an array")
    
    @staticmethod
    def density_fcn(x : float, θ : NDArray) -> float:
        """expression of (x,θ) ↦ f(x|θ)

        Args:
            x (float): where the density is evaluated
            θ (NDArray) : parameters of the distribution

        Returns:
            float:  f(x|θ)
        """
        raise Exception("density function has not been defined for this family")
        return 0.0
    
    def parameters_list(self) -> NDArray[Any]:
        """retourne les paramètres comme liste de float, peu importe le format initial des paramètres fourni"""
        if type(self.parameters) is dict :
            return np.array([*self.parameters.values()])
        elif type(self.parameters) in [list, ArrayLike, NDArray] :
            return np.array(self.parameters)
        else :
            raise TypeError("self.parameters should be a : dict, list or array")
        
    def copy(self) -> Self:
        return copy.deepcopy(self)