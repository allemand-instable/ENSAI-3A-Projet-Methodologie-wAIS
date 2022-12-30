from distribution_family.distribution_family import DistributionFamily
from typing import Callable


def get_density_fcn(f : DistributionFamily | Callable) -> Callable:
    if isinstance(f, DistributionFamily):
        f_fcn = f.density
    elif isinstance(f, Callable):
        f_fcn = f
    else :
        raise TypeError(f"f should be DistributionFamily or Callable but it is {type(f)}")
    return f_fcn



def squared_relative_distance(  p : DistributionFamily | Callable, 
                                q: DistributionFamily | Callable, 
                                x : float
                            ) -> float:
    
    p_fcn = get_density_fcn(p)
    q_fcn = get_density_fcn(q)
    rapport = p_fcn(x)/q_fcn(x)
    return (rapport - 1)**2
