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
                                # nb_stochastic_choice : int,
                                max_L_gradient_norm : int | float, 
                                X_sampled_from_uniform : List[float],
                                param_composante : Optional[int] = None,
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling
    
    ω = f / q_importance_sampling
    
    on a donc 𝛁̂L = 1/n⋅∑ [𝛁_θ]( ω × log(q_θ) )[X_i]
                 = 1/n⋅∑  ω[X_i] × [𝛁_θ]log(q_θ)[X_i]"""
    nb_stochastic_choice = len(X_sampled_from_uniform)             
    
    def ω(x,θ) -> float:
        f_val = f_target.density(x)
        q_val = q_importance_sampling.density_fcn(x, θ)
        res = f_val/q_val
        # debug(logstr(f"ω(x,θ) = {res}"))
        return res
    # ⟶ scalaire

    def h(x,θ) -> NDArray:
        # x ⟼ log qₜ(x)
        def log_q(u, theta) -> float :
            return np.log(q_t.density_fcn(u, theta)) 
        # [𝛁_θ]log qₜ(x)
        if param_composante is None:
            res = gradient_selon(2, log_q, *[x, θ] )
        else :
            res = gradient_selon(2, log_q, *[x, θ], composante=param_composante)
        # debug(logstr(f"h(x,θ) = {get_vector_str(res)}"))
        return res
    # ⟶ vecteur
    
    def grad_L(x_i, θ) -> NDArray:
        res = h(x_i, θ) * ω(x_i, θ) #@ #res = h(x_i, θ) * ω(x_i, θ_0 )            
        norm_res = np.linalg.norm(res)
        norm_theta = np.linalg.norm(np.array(θ))
        # avec les ω, si on a un ω ~ 10 000 lorsque q << f 
        # on va avoir la norme de la direction qui explose
        # on essaye d'éviter cela
        if norm_res > max_L_gradient_norm * norm_theta :
            debug(logstr(f"{norm_res} = || res || > {max_L_gradient_norm} x || θ || = {max_L_gradient_norm*norm_theta}\n\nreturning zeros..."))
            # norm_max * 𝛁L/‖𝛁L‖
            return max_L_gradient_norm * (res/norm_res)
        return res
    # ⟶ vecteur

    grad_L_list : list[NDArray] = [ grad_L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(nb_stochastic_choice) ]
    
    grad_L_estimator : NDArray = np.add.reduce( grad_L_list )/nb_stochastic_choice
    
    return grad_L_estimator
