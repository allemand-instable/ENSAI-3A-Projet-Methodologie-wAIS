from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

from utils.log import logstr
from logging import info, debug, warn, error

from typing import Any, AnyStr, Self, Optional, Dict
from numpy.typing import NDArray


class NormalFamily(DistributionFamily):
    def __init__(self, μ ,Σ) -> None:
        super().__init__(numpy_random_generator_method = nprd.normal , θ ={"loc" : μ, "scale" : Σ})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        μ = θ[0]
        σ_2 = θ[1]
        
        expo = np.exp( 
                      -(
                          ((x-μ)**2) / (2*σ_2)
                        ) 
                    )
        cste_norm = np.sqrt( 2*np.pi*σ_2 )
        
        return expo/cste_norm

class NormalFamily_KnownVariance(DistributionFamily):
    def __init__(self, μ ,Σ) -> None:
        super().__init__(numpy_random_generator_method = nprd.normal , θ ={"loc" : μ, "scale" : Σ})
    
    def density_fcn(self, x, μ : float | NDArray) -> float:
        
        if type(μ) is float :
            μ = np.array([μ])
        
        σ_2 = self.parameters_list()[1]
        
        expo = np.exp( -(((x-μ)**2)/(2*σ_2)) )
        cste_norm = np.sqrt( 2*np.pi*σ_2 )
        
        return expo/cste_norm
    
class NormalFamily_KnownMean(DistributionFamily):
    def __init__(self, μ_known ,Σ_init) -> None:
        super().__init__(numpy_random_generator_method = nprd.normal , θ ={"loc" : μ_known, "scale" : Σ_init})
    
    
    def density_fcn(self, x : float, σ_2 : float | NDArray) -> float:
        
        if isinstance(σ_2, float) :
            σ_2 = np.array([σ_2])
        elif isinstance(σ_2, np.ndarray):
            pass
        else :
            raise TypeError("σ_2 must be float or numpy array")
        
        μ = self.parameters_list()[1]
        
        expo = np.exp( -(((x-μ)**2)/(2*σ_2[0])) )
        cste_norm = np.sqrt( 2*np.pi*σ_2[0] )
        
        return expo/cste_norm