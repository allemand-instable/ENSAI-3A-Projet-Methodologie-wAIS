import numpy as np
from typing import Any, Callable, Literal, Optional, List, Dict, Tuple, TypedDict
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

class IterativeParameterMethod(Protocol):
    def __call__(self, f_target : DistributionFamily, q_init : DistributionFamily, **kwargs) -> NDArray: ...
    
class SGA_Params(TypedDict):
    """Parameters for SGA
    ➤  nb_drawn_samples : int 
    ➤  nb_stochastic_choice : int 
    ➤  step : float
    ➤  θ_0 : Optional[NDArray]  
    ➤  ɛ : float
    ➤  iter_limit : int 
    ➤  max_L_gradient_norm : float 
    ➤  param_composante : int
    """
    nb_drawn_samples : int 
    nb_stochastic_choice : int 
    step : float
    θ_0 : Optional[NDArray[np.float64]]  
    ɛ : float
    iter_limit : int 
    max_L_gradient_norm : float 
    param_composante : int
    
    
class ImportanceSamplingGradientEstimation(Protocol):
    """An integral function (estimated using importance sampling) used for stochastic gradient descent
    
    computes 𝔼[ ω(X) × h(X, θ) ] ≈ 1/N ∑ ωᵢ⋅hᵢ(θ)
    
    with ωᵢ = ω(Xᵢ)
    and  hᵢ(θ) = h(Xᵢ, θ)
    
    arguments :
    
    ➤  f_target                [DistributionFamily]
    ➤  q_t                     [DistributionFamily]
    ➤  q_importance_sampling   [DistributionFamily]
    ➤  θ_t                     [NDArray]
    ➤  nb_stochastic_choice    [int]
    ➤  max_L_gradient_norm     [float]
    ➤  X_sampled_from_uniform  [List[float]]
    ➤  param_composante        [Optional[int]]
    
    """
    def __call__(self,
                f_target : DistributionFamily, 
                q_t : DistributionFamily, 
                q_importance_sampling : DistributionFamily,
                θ_t : NDArray, 
                max_L_gradient_norm : int | float, 
                X_sampled_from_uniform : List[float],
                param_composante : Optional[int],
                )    ->  NDArray: ...
    
class UpdateParameters(TypedDict):
    frequency : int
    
    """SGD"""
    # function to be computed
    gradient_descent__compute_grad_L_importance_sampling : ImportanceSamplingGradientEstimation
    # stochastic part
    # gradient_descent__nb_drawn_samples : int 
    # gradient_descent__nb_stochastic_choice : int 
    # gradient ascent parameters
    gradient_descent__step : float 
    gradient_descent__iter_limit : int
    # other parameters
    gradient_descent__method : Literal["descent"] | Literal["ascent"]
    gradient_descent__update_η : Callable

    gradient_descent__max_L_gradient_norm : int | float
    gradient_descent__adaptive : bool
    # specific sub component of parameter of interest
    gradient_descent__param_composante : Optional[int]
    gradient_descent__given_X : Optional[List[float]]
