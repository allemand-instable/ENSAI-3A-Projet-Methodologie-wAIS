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

import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as plgo

from copy import deepcopy

def xor(a : bool,b : bool) -> bool:
    return (a and not b) or (not a and b)


def SGA_L( f : DistributionFamily ,q : DistributionFamily , N : int, 𝛾 : int, η_0 : float, θ_0 : Optional[ArrayLike] = None, ɛ : float = 1e-6, iter_limit = 100, benchmark : bool = False) -> ArrayLike:
    """effectue une stochastic gradient ascent pour le problème d'optimisation de θ suivant le critère de la vraissemblance de Kullback-Leibler
        
    f           — target density
                    ➤ va être utilisée pour la comparaison avec q dans la maximisation de la vraissemblance de Kullback-Leibler
                    
                    L(θ) = - KL( f || q )
    
    q           — original sampling policy : q(𝑥, θ)
    
                                    parametric family of sampling policies / distributions
                                    given as a (lambda) function of 𝑥, θ ∈ 𝘟 × Θ

                                    q = lambda x,θ : np.exp( - (x-θ[0])**2 / (2*θ[1]) )/(np.sqrt(2*np.pi*θ[1]))
                                    gives a normal law density

                ➤ va être modifiée à chaque itération
    
    
    N           — Nombre de samples tirés par la distribution q à chaque itération
    
    𝛾           — nombre d'observations à tirer aléatoirement
    
    η_0         — initialisation du pas
    
    θ_0         — initialisation des paramètres
    
    ɛ           — threshold pour la norme du gradient
    
    iter_limit  - nombre d'itérations max du gradient descent avant l'arrêt
    
    """
    
    
    info(logstr(f"\n=========    BEGIN : SGD_L   ========="))
    
    # initialisation
    η_t = η_0
    if θ_0 is None :
        θ_0 = q.parameters_list()
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
    
    #! importance sampling selon q(θ_0)
    q_0 = q.copy()
    
    counter = 1
    
    if benchmark is True :
        benchmark_graph : list[list[float]] = [ list([]) for k in range( len(θ_t) + 1)]

    
    while norm_grad_L > ɛ:
        
        if counter > iter_limit :
            break
        
        debug(logstr(f"——————————————————————————————————————————\n             ITERATION N° {counter}              \n——————————————————————————————————————————"))
        
        # on rajoute N observations samplées depuis la sampling policy q_t
        #? un seul grand échantillon
        #! importance sampling selon q(θ_0)
        #!               |
        #!               v
        new_sample = q.sample(N)
        #?          |
        #? debug(logstr(f"Nouveau Sample selon la distribution q:\n    —> params : {q_0.parameters}\n\n{new_sample}"))
        #?          |
        #? un seul grand échantillon
        X = X +  new_sample
        
        debug(logstr(f"\nX = {X}"))
        debug(logstr(f"\nlen(X) = {len(X)}"))
        
        # on détermine les observations aléatoires tirées :
        
        obs_tirées = nprd.choice(range(len(X)), 𝛾, replace=False)
        X_sampled_from_uniform = [  X[i] for i in obs_tirées  ]
        #                                             b inclu
        debug(logstr(f"\nX_sampled_from_uniform = {X_sampled_from_uniform}"))
        
        # on update la valeur de L_i(θ)
        
        
        #ω : Callable[[Any, Any], float]     = lambda x, θ : f(x)/q.density_fcn(x, θ)
        def ω(x,θ) -> float:
            
            debug(logstr(f"θ = {θ}"))
            
            f_val = f.density(x)
            q_val = q.density_fcn(x, θ)
            
            debug(logstr(f"f(x) = {f_val}"))
            debug(logstr(f"q(x, theta) = {q_val}"))
            
            res = f_val/q_val
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
        def grad_L(x_i, θ):
            debug("calcul de L :")
            #! importance sampling selon q(θ_0)
            #!                        |
            #!                        v
            res = h(x_i, θ) * ω(x_i, θ) #@ #res = h(x_i, θ) * ω(x_i, θ_0 )
            debug(logstr(f"∇L_i(θ) = \n{get_vector_str(res)}"))
            
            norm_res = np.linalg.norm(res)
            norm_theta = np.linalg.norm(np.array(θ))
            alpha = 10
            
            # avec les ω, si on a un ω ~ 10 000 lorsque q << f 
            # on va avoir la norme de la direction qui explose
            # on essaye d'éviter cela
            
            if norm_res > alpha * norm_theta :
                debug(logstr(f"{norm_res} = || res || > {alpha} x || θ || = {alpha*norm_theta}\n\nreturning zeros..."))
                return np.zeros(θ.shape)
            return res
        
        
        # on update la valeur du gradient de L selon la méthode de la SGD
        debug(logstr("calcul de L_list_divided_by_𝛾"))
        grad_L_list = [ grad_L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(𝛾) ]
        debug(logstr(f"∇L_list_divided_by_𝛾 = \n"))
        
        
        
        #?  ——————————————————————————————————————————————————————————————————————————  ?#
        #?                                  DEBUG                                       ?#
        #?                   afficher chaque composante ∇L_i/𝛾
        #for k in range(len(grad_L_list)):
        #    debug(logstr(f"∇L_{k+1}(θ) = {get_vector_str(grad_L_list[k])}"))
        #?  ——————————————————————————————————————————————————————————————————————————  ?#
        
        
        
        un_sur_𝛾_Σ_gradL_i_θt = np.add.reduce( grad_L_list )/𝛾
        debug(logstr(f"un_sur_𝛾_Σ_gradL_i_θt = {un_sur_𝛾_Σ_gradL_i_θt}"))

        
        norm_grad_L = np.linalg.norm(un_sur_𝛾_Σ_gradL_i_θt)
        
        
        # update des (hyper) paramsw
        
        # paramètre
        θ_t = θ_t + η_t * un_sur_𝛾_Σ_gradL_i_θt #θ_t = θ_t - η_t * un_sur_𝛾_Σ_gradL_i_θt #@
        str_theta = f"θ_{counter} = {θ_t}"
        print(str_theta)
        debug(logstr(str_theta))
        # ⟶ vecteur de la dim de θ
        
        # sampling policy
        q.update_parameters(θ_t)
        
        # pas
        η_t = update_η(η_t)
        debug(logstr(f"η_t+1 = {η_t}"))
        counter += 1
        
        if benchmark is True :
            target = f.parameters_list()
            benchmark_graph[0].append(counter)
            for k in range(len(θ_t)) :
                d_k = np.abs((θ_t[k] - target[k])/(target[k] + 1e-4))
                benchmark_graph[1+k].append(d_k)
            
            
            
    
    info(logstr("\n=========     FIN : SGD_L     ========="))
    
    if benchmark is True :
        
        print(f"c/2 = {counter//2}\n c % 2 ={counter%2}")
        print(benchmark_graph)
        
        fig = make_subplots(
                            rows= len(θ_t)//2 + len(θ_t)%2 , cols=2,
                            subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(len(θ_t))]
                    )

        for k in range(len(θ_t)):
            print(f"({1 + k//2}, {1 + k%2})")
            fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+k]), row = 1 + k//2 , col = 1 + k%2)
    
        fig.show()
    return θ_t
       
        

def update_η(η_t):
    η_t_plus_1 = η_t
    return η_t_plus_1



