import numpy as np
from typing import Any, Callable, Optional, List, Dict, Tuple
from numpy.typing import NDArray
from numpy.typing import ArrayLike
from typing import Protocol, Union
from distribution_family.distribution_family import DistributionFamily

vector_or_scalar = NDArray | float


#? BenchmarkGraph = Optional[List[ List[int]   | List[float] ]]
#                                  iterations  |   erreur relative à composante k ∈ ⟦1,len(θₜ)⟧
#    index :                       0           |   1, ... , n = len(θₜ)
#    passe mieux pour le type hinting même si le vrai est plutôt en haut
BenchmarkGraph = Optional[List[ List[float] ]]

ParamsInitiaux = Tuple[ float,                  # η_t
                        NDArray,                # θ_t
                        float,                  # norm_grad_L
                        List[float],            # X
                        DistributionFamily,     # q_0
                        BenchmarkGraph,         # benchmark_graph
                        List[bool]              # state
                      ]



class MultivariateFunction_to_R(Protocol):
    """
    function defined by the relation y = f( 𝑥ᵢ )₁,ₙ
    
    i.e
    
        Π ℝᴸᵉⁿ⁽ˣ-ⁱ⁾ ⟶   ℝ
    f : ( 𝑥ᵢ )₁,ₙ   ⟼   y
    """
    def __call__(self, *float_args : NDArray[np.float64] ) -> float: ...
