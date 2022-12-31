import numpy as np
# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray
from custom_typing.custom_types import ImportanceSamplingGradientEstimation
# My Modules
from gradient.gradient import gradient_selon
from distribution_family.distribution_family import DistributionFamily
# Debug
from utils.log import logstr
from logging import info, debug, warn, error




def compute_grad_L_estimator_importance_sampling(
                                f_target : DistributionFamily, 
                                q_t : DistributionFamily, 
                                q_importance_sampling : DistributionFamily,
                                θ_t : NDArray, 
                                nb_stochastic_choice : int,
                                max_L_gradient_norm : int | float, 
                                X_sampled_from_uniform : List[float],
                                #
                                α : float,
                                #
                                param_composante : Optional[int] = None
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling
    
    rapport(α) : x ⟼ (f/q_t)^(1-α)
    ωᵢ = rapport(α)(xᵢ) / ∑ₖ rapport(α)(xₖ)
    
    hᵢ(Θ) =  [𝛁̂_θ] qₜ(𝑥ᵢ) / qₜ(𝑥ᵢ) 
    
    on a donc 𝛁̂L = 1/n⋅∑  ω[X_i] × h(θ)[X_i]
   """
    def rapport(α : float, f : DistributionFamily, q_θ : DistributionFamily, x) -> float:
       return ( f.density(x) / q_θ.density(x) )**(1-α)
   
    def ω(x, θ) -> float:   
        r = rapport(α, f_target, q_t, x)
        s = np.array([rapport(α, f_target, q_t, xᵢ) for xᵢ in X_sampled_from_uniform]).sum()
   
        return r/s
   
    def h(x, θ) -> NDArray:
        if param_composante is None:
            res = gradient_selon(2, q_t.density_fcn, *[x, θ] ) / q_t.density_fcn(x, θ)
        else :
            res = gradient_selon(2, q_t.density_fcn, *[x, θ], composante=param_composante) / q_t.density_fcn(x, θ)
        return res

    def grad_L(x_i, θ) -> NDArray:
        res = h(x_i, θ) * ω(x_i, θ) #@ #res = h(x_i, θ) * ω(x_i, θ_0 )            
        norm_res = np.linalg.norm(res)
        norm_theta = np.linalg.norm(np.array(θ))
        if norm_res > max_L_gradient_norm * norm_theta :
            # norm_max * 𝛁L/‖𝛁L‖
            return max_L_gradient_norm * (res/norm_res)
        return res
    # ⟶ vecteur

    grad_L_list : list[NDArray] = [ grad_L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(nb_stochastic_choice) ]

    grad_L_estimator : NDArray = np.add.reduce( grad_L_list )/nb_stochastic_choice

    return grad_L_estimator


def give_estimator(α : float) -> ImportanceSamplingGradientEstimation:
    
    def fcn(                    
            f_target : DistributionFamily, 
            q_t : DistributionFamily, 
            q_importance_sampling : DistributionFamily,
            θ_t : NDArray, 
            nb_stochastic_choice : int,
            max_L_gradient_norm : int | float, 
            X_sampled_from_uniform : List[float],
            #
            param_composante : Optional[int] = None
        ) -> NDArray:
        return compute_grad_L_estimator_importance_sampling(
                                                            f_target, 
                                                            q_t, 
                                                            q_importance_sampling,
                                                            θ_t, 
                                                            nb_stochastic_choice,
                                                            max_L_gradient_norm, 
                                                            X_sampled_from_uniform,
                                                            α,
                                                            param_composante
                                                        )
    return fcn