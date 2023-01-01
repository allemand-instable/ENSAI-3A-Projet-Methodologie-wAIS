from typing import Callable, List, Tuple, Dict
from distribution_family.distribution_family import DistributionFamily
import numpy as np
from numpy.typing import NDArray
from wAIS.squared_relative_distance import squared_relative_distance
from wAIS.update_q_t import update_qₜ
from custom_typing.custom_types import UpdateParameters
from renyi_alpha_divergence.renyi_importance_sampling_gradient_estimator import renyi_gradL as R_gradL
from kullback_leibler.L_gradient.grad_importance_sampling import compute_grad_L_estimator_importance_sampling as KL_gradL

def Sₜ(Ψ : Callable, qₜ : DistributionFamily, Xₜ : List[float])-> float :
    return np.array([ Ψ(xᵢ) / qₜ.density(xᵢ) for xᵢ in Xₜ ]).sum()

def αₜ(π : DistributionFamily, qₜ : DistributionFamily, Xₜ : List[float]) -> float :
    dist_list = [ squared_relative_distance(π, qₜ, xᵢ) for xᵢ in Xₜ ]
    # print(f"\ndist_list = {dist_list}")
    # print(f"{Xₜ}")
    return 1/ sum(dist_list)

def N( list_nₜ : List[float] , list_αₜ : List[float]) -> float:
    if len(list_nₜ) == len(list_αₜ) :
        n = len(list_nₜ)
    else :
        raise IndexError("len(list_nₜ) ≠ len(list_αₜ)")
    terms = [list_nₜ[i] * list_αₜ[i] for i in range(n)]
    return sum(terms)

def Iₜ( Ψ   : Callable,
        # params
        π       : DistributionFamily,
        qₜ      : DistributionFamily,
        nₜ      : int,
        Iₜ_ᵤₙ   : float = 0. 
        )  -> Tuple[float, float, float, List]:
    Xₜ = qₜ.sample(nₜ)
    if Xₜ is None :
        raise TypeError("unable to generate X_t")
    alpha_t = αₜ(π, qₜ, Xₜ)
    S_t = Sₜ(Ψ, qₜ, Xₜ)
    I_t = Iₜ_ᵤₙ + (alpha_t * S_t)
    # print(I_t)
    return  alpha_t, S_t, I_t, Xₜ

def I(  φ           : Callable, 
        π           : DistributionFamily, 
        q_init      : DistributionFamily, 
        update_params : UpdateParameters,
        nₜ_policy   : Callable = lambda t : 50, 
        T           : int = 600,
    ) -> float:

    Ψ = lambda x :φ(x)*π.density(x)

    qₜ  = q_init.copy()
    I_t = 0
    nₜ = nₜ_policy(0)
    N_T = 1
    list_n      = []
    list_alpha  = []
    
    for t in range(T):
        alpha_t, S_t, I_t, Xₜ = Iₜ( Ψ, π, qₜ, nₜ, Iₜ_ᵤₙ = I_t)
        list_n.append(nₜ)
        list_alpha.append(alpha_t)
        N_T = N( list_n, list_alpha )
        update_qₜ(
            t=t,
            qₜ=qₜ, 
            Xₜ=Xₜ, 
            φ=φ, 
            π=π, 
            Iₜ=I_t/N_T, 
            update_params=update_params
            )
        
    
    return I_t/N_T
    
def weighted_adaptive_importance_sampling(
                                            φ           : Callable,
                                            π           : DistributionFamily,
                                            q_init      : DistributionFamily,
                                            update_params : UpdateParameters,
                                            nₜ_policy   : Callable = lambda t : 50,
                                            T           : int = 100
                                        ) -> float:
    numerator =I( φ= φ , π = π, q_init = q_init, nₜ_policy = nₜ_policy, T = T, update_params =update_params )
    denominator = I( φ = lambda x : 1 , π = π, q_init = q_init, nₜ_policy = nₜ_policy, T = T, update_params = update_params)
    
    # print("\n\nnum et denum")
    # print(numerator)
    # print(denominator)
    # print("\n\n")
    
    return  numerator/ denominator



default_params_KL = dict(
    frequency = 6,
    # function to be computed
    gradient_descent__compute_grad_L_importance_sampling = KL_gradL,
    # gradient ascent parameters
    gradient_descent__step = 0.3,
    gradient_descent__iter_limit = 1,
    # other parameters
    gradient_descent__method = "ascent",
    gradient_descent__update_η = lambda x : x,
    gradient_descent__max_L_gradient_norm = 50,
    gradient_descent__adaptive = True,
    # specific sub component of parameter of interest
    gradient_descent__param_composante = None,
    # todo
    gradient_descent__given_X = None
)

default_params_R = dict(
    frequency = 6,
    # function to be computed
    gradient_descent__compute_grad_L_importance_sampling = R_gradL(2),
    # gradient ascent parameters
    gradient_descent__step = 0.3,
    gradient_descent__iter_limit = 1,
    # other parameters
    gradient_descent__method = "descent",
    gradient_descent__update_η = lambda x : x,
    gradient_descent__max_L_gradient_norm = 50,
    gradient_descent__adaptive = True,
    # specific sub component of parameter of interest
    gradient_descent__param_composante = None,
    # todo
    gradient_descent__given_X = None
)