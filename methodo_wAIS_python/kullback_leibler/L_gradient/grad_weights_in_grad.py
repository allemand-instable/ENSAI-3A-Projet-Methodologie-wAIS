import numpy as np
# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray
# My Modules
from gradient.gradient import gradient_selon
from distribution_family.distribution_family import DistributionFamily
# Debug
from utils.log import logstr
from logging import info, debug, warn, error


def compute_grad_L_estimator_adaptive(  f_target : DistributionFamily, 
                                        q_t : DistributionFamily, 
                                        θ_t : NDArray, 
                                        max_L_gradient_norm : int | float, 
                                        X_sampled_from_uniform : List[float],
                                        param_composante : Optional[int] = None
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling avec un q adaptatif
    
    ω_θ = f / q_θ
    on a donc ̂𝛁L = 1/n⋅∑ [𝛁_θ]( ω_θ × log(q_θ) )[X_i]
    """
    
    nb_stochastic_choice = len(X_sampled_from_uniform)
    
    def ω(x,θ) -> float:
        f_val = f_target.density(x)
        q_val = q_t.density_fcn(x, θ)
        res = f_val/q_val
        # debug(logstr(f"ω(x,θ) = {res}"))
        return res
    # ⟶ scalaire

    def grad_L(x_i, θ) -> NDArray:
        
        def log_q(u, theta) -> float :
            return np.log(q_t.density_fcn(u, theta)) 
        
        fcn = lambda x, theta : ω(x, theta) * log_q(x, theta)
        if param_composante is None :
            res = gradient_selon(2, fcn, *[x_i, θ])
        else :
            res = gradient_selon(2, fcn, *[x_i, θ], composante=param_composante)
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
