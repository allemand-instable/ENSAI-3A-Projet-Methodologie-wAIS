from math_tools.gradient import gradient_selon
import numpy as np
from typing import Callable, Any
from random import randint
import numpy.random as nprd
from math_tools.distribution_family import DistributionFamily
from numpy.typing import ArrayLike

def SGD_L( q : DistributionFamily , N : int, 𝛾 : int, η_0 : float, θ_0 : float, ɛ : float) -> ArrayLike:
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
    f = lambda x : x
    # initialisation
    η_t = η_0
    θ_t = θ_0
    # on s'assure de commencer la première itération
    norm_grad_L = (ɛ + 1)
    
    X = []
    
    while norm_grad_L > ɛ :
        
        # on rajoute N observations samplées depuis la sampling policy q_t
        X.append( q.sample(N) )
        
        # on détermine les observations aléatoires tirées :
        
        X_sampled_from_uniform = [ X[randint(a=0, b=len(X))] for k in range(𝛾)  ]
        print(X_sampled_from_uniform)
        
        # on update la valeur de L_i(θ)
        # q : Callable[[Any, Any], float]     = lambda x,θ : 
        h : Callable[[Any, Any], float]     = lambda x, θ : f(x)/q.density_fcn(x, θ)
        # ⟶ scalaire
        ω : Callable[[Any, Any], Any]       = lambda x, θ : gradient_selon(2, lambda u, v : np.log(q.density_fcn(u, v)), *[x, θ] )
        # ⟶ vecteur
        L : Callable[[Any, Any], float]     = lambda x_i, θ : h(x_i, θ) * ω(x_i, θ)
        # ⟶ vecteur
        # on update la valeur du gradient de L selon la méthode de la SGD
        L_list_divided_by_𝛾 = [ L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(𝛾) ]
        # ⟶ list[vecteur]
        un_sur_𝛾_Σ_gradL_i_θt = np.add.reduce( L_list_divided_by_𝛾 )
        # ⟶ vecteur de la dim de θ
        
        norm_grad_L = np.linalg.norm(un_sur_𝛾_Σ_gradL_i_θt)
        
        
        # update des (hyper) paramsw
        
        # paramètre
        θ_t = θ_t - η_t * un_sur_𝛾_Σ_gradL_i_θt
        # ⟶ vecteur de la dim de θ
        
        # sampling policy
        q.update_parameters(θ_t)
        
        # pas
        η_t = update_η(η_t)
    
    return θ_t
       
        

def update_η(η_t):
    η_t_plus_1 = η_t
    return η_t_plus_1

