from math_tools.distribution_family import DistributionFamily
import numpy.random as nprd
import numpy as np

class NormalFamily(DistributionFamily):
    def __init__(self, μ ,Σ) -> None:
        super().__init__(numpy_random_generator_method = nprd.normal , θ ={"loc" : μ, "scale" : Σ})
    
    @staticmethod
    def density_fcn(x, θ) -> float:
        
        μ = θ[0]
        σ_2 = θ[1]
        
        expo = np.exp( -(((x-μ)**2)/(2*σ_2)) )
        cste_norm = np.sqrt( 2*np.pi*σ_2 )
        
        return expo/cste_norm