from typing import Callable, Dict, List, Literal, Optional
from distribution_family.distribution_family import DistributionFamily
import kullback_leibler.L_gradient.grad_importance_sampling as kl_grad_is

import numpy as np
from numpy.typing import NDArray
from general.importance_sampling import importance_sampling_given_sample

from wAIS.update_sampling_policy import update_sampling_policy
from wAIS.squared_relative_distance import squared_relative_distance

def get_alpha_t(α : NDArray[np.float64], t) -> float:
    return α[t] / α.sum()
    
def weighted_adaptive_importance_sampling(  q_init : DistributionFamily,
                                            φ : Callable,
                                            π : Callable,
                                            nb_iter : int,
                                            nb_samples_per_iteration : int,
                                            nb_gradient_steps : int, 
                                            gradient_descent_frequency : int,
                                            # update sampling policy variables
                                            update_sampling_policy__step : float,
                                            update_sampling_policy__nb_drawn_samples : int,
                                            update_sampling_policy__nb_stochastic_choice : int,
                                            update_sampling_policy__param_composante : int,
                                            #       gradient ascent kullback-leibler
                                            update_sampling_policy__compute_grad_L_importance_sampling = kl_grad_is.compute_grad_L_estimator_importance_sampling,
                                            update_sampling_policy__method: Literal["descent"] | Literal["ascent"] = "ascent",
                                            update_sampling_policy__update_η : Callable = lambda x : x,
                                            update_sampling_policy__max_L_gradient_norm : int | float = np.Infinity,
                                            ) -> float:
    """
    returns an estimation of ∫f dλ = ∫ φ⋅π dλ   = ∫ (π/q)⋅φ⋅q dλ
                                                = ∫   ω  ⋅φ⋅q dλ
                                                = 𝔼_q[ ω⋅φ ]
    """
    
    q_t = q_init.copy()
    if π is None :
        # todo : ce ne peut pas être q_t puisque sinon alpha = 0 = (q_t/q_t - 1)²
        #! mais alors quoi ???
        # π = q_t
        raise ValueError("π can't be None")
    
    # X = [X₀, ..., Xₜ]
    X : List[float]
    X = []
    # inutile : utilisé pour N mais I_f / I_π
    # n[0] = 𝑛₀            N[t] = 𝑛ₜ
    # n : NDArray[np.int64] = np.ndarray([], dtype=np.int64)
    # need ndarray for sum method
    α : NDArray[np.float64]
    α = np.array([], dtype=np.float64)
    # Xₜ : List[float]
    I_f_array = np.array([], dtype=np.float64)
    importance_sampling_π_array = np.array([], dtype=np.float64)
    
    I_f = 0
    I_π = 0
    
    I_t : Optional[float] = None
        
    for t in range(nb_iter):
        # sampling
        X_t = q_t.sample(n = nb_samples_per_iteration)
        if X_t is None :
            raise ValueError("a problem happened while generating X_t, result is None")
        X = X + X_t

        """détermination des poids"""
        relative_distance_sum_t : float
        relative_distance_sum_t = np.array([squared_relative_distance(π, q_t, x_i) for x_i in X_t]).sum()
        α = np.append(arr= α , values=1/relative_distance_sum_t)    
        # poids : 
        normalized_alpha : NDArray[np.float64] 
        normalized_alpha = α / α.sum()
        # pas utile car I = I_f / I_π
        # n = np.append(n , nb_samples_per_iteration)
        # weighted_N : float
        # weighted_N = ((n * α).sum()) / α.sum()
    
        """————— I_f —————"""
        def ω(x : float) -> float:
            return π(x) / q_t.density(x)
        
        I_t_fcn = lambda X_t : importance_sampling_given_sample(
                                                ω = ω,
                                                h = φ,
                                                X_from_sampler= X_t
                                                )
        I_f_array = np.append(arr = I_f_array, values = I_t_fcn(X_t) )
        I_f += (normalized_alpha[-1] * I_f_array[-1])
        
        """—————  I_π  —————"""
        importance_sampling_π = lambda X_t : np.array([ π.density(x_i)/q_t.density(x_i) for x_i in X_t ], dtype=np.float64).sum()
        importance_sampling_π_array = np.append(importance_sampling_π_array, importance_sampling_π(X_t))
        I_π += (normalized_alpha[-1] * importance_sampling_π_array[-1])
        
        I_t = I_f / I_π
        if I_t is None :
            raise ValueError("Iₜ should not be None")
        # si π est une densité de probabilité
        """∫ (ω×φ)⋅q dλ = 𝔼_q[ (ω×φ) ]"""
        update_sampling_policy( q_t                         = q_t,
                                t                           = t,
                                nb_gradient_steps           = nb_gradient_steps,
                                gradient_descent_frequency  = gradient_descent_frequency,
                                I_t                         = I_t,
                                φ                           = φ,
                                π                           = π,
                                X                           = X,
                                step                        = update_sampling_policy__step,
                                nb_drawn_samples            = update_sampling_policy__nb_drawn_samples, 
                                nb_stochastic_choice        = update_sampling_policy__nb_stochastic_choice, 
                                param_composante            = update_sampling_policy__param_composante, 
                                update_η                    = update_sampling_policy__update_η, 
                                max_L_gradient_norm         = update_sampling_policy__max_L_gradient_norm, 
                                method                      = update_sampling_policy__method,
                                compute_grad_L_importance_sampling = update_sampling_policy__compute_grad_L_importance_sampling
                              )
        
        
    if isinstance(I_t, float):
        return I_t
    else :
        raise TypeError("Unknown type for Iₜ")