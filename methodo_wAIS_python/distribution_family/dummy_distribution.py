from abc import ABCMeta, abstractclassmethod, abstractmethod
from distribution_family.distribution_family import DistributionFamily
from numpy.typing import NDArray

class DummyFamily(DistributionFamily, metaclass=ABCMeta):
    def __init__(self) -> None:
        """on évite :
        
        if len(θ) == 0
        
        if type(self.parameters) not in (list, NDArray, dict)
        """
        super().__init__(numpy_random_generator_method = lambda x : None, θ = [None])
    
    # les deux sont sensés retourner la même fonction
    def density_fcn(self, x: float, θ: NDArray) -> float:
        return self.density(x)
    
    @abstractmethod
    def density(self, x) -> float :
        pass