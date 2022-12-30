from distribution_family.distribution_family import DistributionFamily
from distribution_family.dummy_distribution import DummyFamily
from typing import Callable

def get_target_density(     π : Callable,
                            φ : Callable,
                            I_t : float,
                            target_π : bool = False
                       ) -> DistributionFamily :

    if target_π is True :
        fcn = lambda x : π(x)
    else :
        fcn = lambda x : π(x) * abs( φ(x) - I_t )
    
    class TargetDensity(DummyFamily):
        def __init__(self) -> None:
            super().__init__() 
        def density(self, x: float) -> float:
            return fcn(x)
    
    f_target = TargetDensity()
    
    return f_target
