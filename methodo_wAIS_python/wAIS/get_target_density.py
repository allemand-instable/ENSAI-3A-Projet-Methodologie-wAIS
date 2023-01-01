from distribution_family.distribution_family import DistributionFamily
from distribution_family.dummy_distribution import DummyFamily
from typing import Callable
from wAIS.squared_relative_distance import get_density_fcn
import numpy as np

class TargetDensity(DummyFamily):
        def __init__(self, function : Callable) -> None:
            self.function = function
            super().__init__() 
        def density(self, x: float) -> float:
            return self.function(x)


def get_target_density(     π : DistributionFamily,
                            φ : Callable,
                            I_t : float,
                       ) -> DistributionFamily :
    """PORTIER DELYON [B.1]
    
    derived from asymptotic variance :
    u*Vu = ∫ q⁻¹ π²(φ-I)²
    """
    fcn = lambda x : π.density(x) * abs( φ(x) - I_t )
    f_target = TargetDensity(fcn)
    return f_target
