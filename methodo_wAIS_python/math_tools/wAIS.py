from math_tools.distribution_family import DistributionFamily
import numpy as np
import numpy.typing as npt
from typing import LiteralString


update_n_t_keys = ["constant"]


class WeightedAdaptiveImportanceSampling():
    """provides an estimate of ∫ f dλ = ∫ h ⋅ p dλ = 𝔼ₚ[ h ]
    """
    def __init__(self, debug : bool = True, n_t_update_policy : LiteralString = "constant") -> None:
        
        
        self.__debug = debug
        
        
        self.I_t : float
        
        # distribution distance weight
        self.alpha_t : float
        
        # sampling distribution
        self.q_t : DistributionFamily
        
        # Nₜ = Σ nₜ
        self.N_t : int
        
        self.n_t_update_policy : LiteralString = n_t_update_policy
        
        # added space complexity for debug purpouse
        if debug is True :
            self.I_history : list[float]
            self.n_t_history : list[int]
        else :
            self.n_t : int
        
        
        
        self.X_t : list[float]
        
        pass
    
    def update_n_t(self) -> int:
        """updates nₜ by appening it in the history list and returns the new value
        """
        
        match self.n_t_update_policy:
            case "constant":
                if self.__debug is True :
                    new_n_t = self.n_t_history[-1]
                    self.n_t_history.append(new_n_t)
                else :
                    new_n_t = self.n_t
            case _ :
                raise KeyError(f"n_t_update_policy should be among :\n{update_n_t_keys}")
        return new_n_t
    
    def update_N_t(self) -> int:
        """updates N_t and returns the new value

        Returns:
            N_t [int]
        """
        if self.__debug is True :
            new_N_t = np.sum(self.n_t_history)
        else : 
            new_N_t = self.N_t + self.update_n_t()
        self.N_t = new_N_t
        return new_N_t
    
    def update_alpha_t(self):
        
        lambda srd x : self.squared_relative_distance(self , self.q_t, x)
        
        new_alpha_t = 1/(np.sum(self.X_t))
        
        pass
    
    @staticmethod
    def squared_relative_distance(f : DistributionFamily, q: DistributionFamily, x) -> float:
        return ( f.density(x)/q.density(x) - 1 )**2