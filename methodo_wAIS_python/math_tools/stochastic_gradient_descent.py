from math_tools.gradient import gradient_selon
import numpy as np
from typing import Callable, Any, Optional
from random import randint
import numpy.random as nprd
from math_tools.distribution_family import DistributionFamily
from numpy.typing import ArrayLike

from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str


def SGD_L( f : DistributionFamily ,q : DistributionFamily , N : int, 𝛾 : int, η_0 : float, θ_0 : Optional[ArrayLike] = None, ɛ : float = 1e-10) -> ArrayLike:
    """_summary_
    
    (X       — observations X = [... X_i ...] samplées depuis q)
    
    q       — sampling policy : q(𝑥, θ)
    
                                parametric family of sampling policies / distributions
                                given as a (lambda) function of 𝑥, θ ∈ 𝘟 × Θ
                                
                                q = lambda x,θ : np.exp( - (x-θ[0])**2 / (2*θ[1]) )/(np.sqrt(2*np.pi*θ[1]))
                                gives a normal law density
    
    
    N       — Nombre de samples tirés par la distribution q à chaque itération
    
    𝛾       — nombre d'observations à tirer aléatoirement
    
    η_0     — initialisation du pas
    
    θ_0     — initialisation des paramètres
    
    ɛ       — threshold pour la norme du gradient
    
    """
    
    info(logstr(f"\n=========    BEGIN : SGD_L   ========="))
    
    # initialisation
    η_t = η_0
    if θ_0 is None :
        θ_t = q.parameters_list()
    else :
        θ_t = θ_0
        q.update_parameters(θ_0)
    # on s'assure de commencer la première itération
    norm_grad_L = (ɛ + 1)
    
    X = []
    
    debug(logstr(
        f"\nη_t = {η_t}\nθ_t = {θ_t}\n𝛾 = {𝛾}\nN = {N}\n"
                ))
    
    
    counter = 1
    while norm_grad_L > ɛ :
        
        debug(logstr(f"——————————————————————————————————————————\n             ITERATION N° {counter}              \n——————————————————————————————————————————"))
        
        # on rajoute N observations samplées depuis la sampling policy q_t
        new_sample = q.sample(N)
        
        debug(logstr(f"Nouveau Sample selon la distribution q:\n    —> params : {q.parameters}\n\n{new_sample}"))
        
        X = X +  new_sample
        
        debug(logstr(f"\nX = {X}"))
        debug(logstr(f"\nlen(X) = {len(X)}"))
        
        # on détermine les observations aléatoires tirées :
        
        X_sampled_from_uniform = [ X[randint(a=0, b= (len(X)-1) )] for k in range(𝛾)  ]
        #                                             b inclu
        debug(logstr(f"\nX_sampled_from_uniform = {X_sampled_from_uniform}"))
        
        # on update la valeur de L_i(θ)
        
        
        #ω : Callable[[Any, Any], float]     = lambda x, θ : f(x)/q.density_fcn(x, θ)
        def ω(x,θ) -> float:
            res = f.density(x)/q.density_fcn(x, θ)
            debug(logstr(f"ω(x,θ) = {res}"))
            return res
        # ⟶ scalaire
        
        
        #h : Callable[[Any, Any], Any]       = lambda x, θ : gradient_selon(2, lambda u, v : np.log(q.density_fcn(u, v)), *[x, θ] )
        def h(x,θ):
            res = gradient_selon(2, lambda u, v : np.log(q.density_fcn(u, v)), *[x, θ] )
            debug(logstr(f"h(x,θ) = {get_vector_str(res)}"))
            return res
        # ⟶ vecteur
        
        
        #L : Callable[[Any, Any], float]     = lambda x_i, θ : h(x_i, θ) * ω(x_i, θ)
        def L(x_i, θ):
            res = h(x_i, θ) * ω(x_i, θ)
            debug(logstr(f"L_i(θ) = \n{get_vector_str(res)}"))
            return res
        # ⟶ vecteur
        
        
        # on update la valeur du gradient de L selon la méthode de la SGD
        debug(logstr("calcul de L_list_divided_by_𝛾"))
        L_list_divided_by_𝛾 = [ L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(𝛾) ]
        debug(logstr(f"L_list_divided_by_𝛾 = \n"))
        for k in range(len(L_list_divided_by_𝛾)):
            debug(logstr(f"L_{k+1}(θ) = {get_vector_str(L_list_divided_by_𝛾[k])}"))
        # ⟶ list[vecteur]
        
        
        un_sur_𝛾_Σ_gradL_i_θt = np.add.reduce( L_list_divided_by_𝛾 )
        debug(logstr(f"un_sur_𝛾_Σ_gradL_i_θt = {un_sur_𝛾_Σ_gradL_i_θt}"))
        # ⟶ vecteur de la dim de θ
        
        norm_grad_L = np.linalg.norm(un_sur_𝛾_Σ_gradL_i_θt)
        
        
        # update des (hyper) paramsw
        
        # paramètre
        θ_t = θ_t - η_t * un_sur_𝛾_Σ_gradL_i_θt
        debug(logstr(f"θ_t+1 = {θ_t}"))
        # ⟶ vecteur de la dim de θ
        
        # sampling policy
        q.update_parameters(θ_t)
        
        # pas
        η_t = update_η(η_t)
        debug(logstr(f"η_t+1 = {η_t}"))
        counter += 1
    
    info(logstr("\n=========     FIN : SGD_L     ========="))
    
    return θ_t
       
        

def update_η(η_t):
    η_t_plus_1 = η_t
    return η_t_plus_1

