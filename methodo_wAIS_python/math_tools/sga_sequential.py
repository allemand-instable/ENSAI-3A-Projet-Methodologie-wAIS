import numpy as np
from copy import deepcopy

# typing
from typing import Callable, Any, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray

# random
from random import randint
import numpy.random as nprd

# My Modules
from math_tools.gradient import gradient_selon
from math_tools.distribution_family import DistributionFamily

# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str

# Plots
import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as plgo


def initialisation(q, ɛ, θ_0, N, 𝛾, η_0, benchmark) -> Tuple[float, NDArray, float, list[float],DistributionFamily, list[list[float]] | None, list[bool]]:
    """initialisationdes paramètres pour la SGA de la fonction L
    
    
    Inputs : 
    
        q                               — original sampling policy : q(𝑥, θ)
        
        ɛ                               — threshold pour la norme du gradient
        
        θ_0                             — initialisation des paramètres
        
        nb_drawn_samples (N)            — Nombre de samples tirés par la distribution q à chaque itération
        
        nb_stochastic_choice (𝛾)        — nombre d'observations à tirer aléatoirement
        
        step (η_0)                      — initialisation du pas
        
        benchmark                       - if True, produces error graphs




    Return :
    
        η_t, θ_t, norm_grad_L, X, q_0, counter, benchmark_graph, state
        
        η_t             : float
        θ_t             : NDArray
        norm_grad_L     : float
        X               : list[float]
        q_0             : DistributionFamily
        benchmark_graph : list[list[float]] | None
        state           : list[bool]
    
    """
    
    """TYPES DEFINITION"""
    η_t             : float
    θ_t             : NDArray
    norm_grad_L     : float
    X               : list[float]
    q_0             : DistributionFamily
    counter         : int
    benchmark_graph : list[list[float]] | None
    state           : list[bool]
    
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
        
    
    if benchmark is True :
        benchmark_graph = [ list([]) for k in range( len(θ_t) + 1)]
    else :
        benchmark_graph = None

    state = [False for k in range(3)]
    
    return η_t, θ_t, norm_grad_L, X, q_0, benchmark_graph, state



def update_η(η_t):
    η_t_plus_1 = η_t
    return η_t_plus_1


def cond_n_de_suite__update_state(cond, state : list[bool]) -> list[bool]:
    # true and false in state
    if all(state):
        new_state = state
    else:
        if cond is True:
            first_false = state.index(False)
            new_state = deepcopy(state)
            new_state[first_false] = True
        else:
            new_state = [False for k in range(len(state))]
    # all true
    return new_state



def compute_grad_L_estimator(f_target, q, θ_t, nb_stochastic_choice,max_L_gradient_norm, X_sampled_from_uniform) -> NDArray:
    def ω(x,θ) -> float:
        f_val = f_target.density(x)
        q_val = q.density_fcn(x, θ)
        res = f_val/q_val
        debug(logstr(f"ω(x,θ) = {res}"))
        return res
    # ⟶ scalaire

    def h(x,θ) -> NDArray:
        res = gradient_selon(2, lambda u, v : np.log(q.density_fcn(u, v)), *[x, θ] )
        debug(logstr(f"h(x,θ) = {get_vector_str(res)}"))
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
            # return np.zeros(θ.shape)
            return max_L_gradient_norm * (res/norm_res)
        return res
    # ⟶ vecteur

    grad_L_list : list[NDArray] = [ grad_L(x_i = X_sampled_from_uniform[i], θ = θ_t) for i in range(nb_stochastic_choice) ]
    
    grad_L_estimator : NDArray = np.add.reduce( grad_L_list )/nb_stochastic_choice
    
    return grad_L_estimator



def sga_kullback_leibler_likelihood(
     f_target : DistributionFamily ,
     q_init : DistributionFamily , 
     nb_drawn_samples : int, 
     nb_stochastic_choice : int, 
     step : float, 
     θ_0 : Optional[ArrayLike] = None, 
     ɛ : float = 1e-6, 
     iter_limit = 100, 
     benchmark : bool = False, 
     max_L_gradient_norm : int = 10
) -> NDArray:
    """effectue une stochastic gradient ascent pour le problème d'optimisation de θ suivant le critère de la vraissemblance de Kullback-Leibler
        
    f                               — target density
                                        ➤ va être utilisée pour la comparaison avec q dans la maximisation de la vraissemblance de Kullback-Leibler
                                        
                                        L(θ) = - KL( f || q )
    
    q                               — original sampling policy : q(𝑥, θ)
    
                                                        parametric family of sampling policies / distributions
                                                        given as a (lambda) function of 𝑥, θ ∈ 𝘟 × Θ

                                                        q = lambda x,θ : np.exp( - (x-θ[0])**2 / (2*θ[1]) )/(np.sqrt(2*np.pi*θ[1]))
                                                        gives a normal law density

                                    ➤ va être modifiée à chaque itération
    
    
    nb_drawn_samples (N)            — Nombre de samples tirés par la distribution q à chaque itération
    
    nb_stochastic_choice (𝛾)        — nombre d'observations à tirer aléatoirement
    
    step (η_0)                      — initialisation du pas
    
    θ_0                             — initialisation des paramètres
    
    ɛ                               — threshold pour la norme du gradient
    
    iter_limit                      - nombre d'itérations max du gradient descent avant l'arrêt

    benchmark                       - if True, produces error graphs
    
    max_L_gradient_norm             - safety coefficient : if ‖ 𝛁L ‖ > 𝜶 ‖ θ_t ‖
                                        ↪ we use 𝜶 × (𝛁L / ‖ 𝛁L ‖)
    """
    
    η_t, θ_t, norm_grad_L, X, q, benchmark_graph, state = initialisation(q_init, ɛ, θ_0, nb_drawn_samples, nb_stochastic_choice, step, benchmark)
    # useful for computing error
    if benchmark_graph is not None :
        target : NDArray = f_target.parameters_list()
    
    # new_samples = []
    
    for counter in range(iter_limit):
        if all(cond_n_de_suite__update_state(norm_grad_L <= ɛ, state)):
            debug(logstr(f"norm_grad_L = {norm_grad_L}"))
            break
        
        new_sample : list | None = q.sample(nb_drawn_samples)
        # new_samples.append(new_sample)
        if new_sample is None :
            raise ValueError(f"could not sample from q \n(params = {q.parameters})\nnew_sample = None")
        
        # todo
        # comprendre pourquoi si je mets juste X = new_sample
        # on finit par avoir des variances négatives ?
        X = new_sample #+ X
        # X = new_samples.pop(0)
        
        if nb_stochastic_choice == nb_drawn_samples :
            X_sampled_from_uniform = X
        else :
            obs_tirées = nprd.choice(range(len(X)), nb_stochastic_choice, replace=False)
            X_sampled_from_uniform = [  X[i] for i in obs_tirées  ]
        
        
        # 𝛁L
        grad_L_estimator = compute_grad_L_estimator(f_target, q, θ_t, nb_stochastic_choice,max_L_gradient_norm, X_sampled_from_uniform)
        # ‖𝛁L‖
        norm_grad_L = np.linalg.norm(grad_L_estimator)
        
        # gradient ascent
        θ_t = θ_t + η_t * grad_L_estimator
        
        str_theta = f"θ_{counter} = {θ_t}"
        print(str_theta)
        debug(logstr(str_theta))
        
        # aprameters update
        q.update_parameters(θ_t)
        
        η_t = update_η(η_t)
        debug(logstr(f"η_t+1 = {η_t}"))
        
        # if we desire to benchmark : we build the error graph
        if benchmark_graph is not None :
            
            #           X AXIS
            benchmark_graph[0].append(counter)
            
            #           Y AXIS
            for k in range(len(θ_t)) :
                # we add the relative error between θ_t and θ_target
                d_k = np.abs((θ_t[k] - target[k])/(target[k] + 1e-4))
                #####################################################
                benchmark_graph[1+k].append(d_k)
        
    # à la fin on plot le graphe des erreurs
    if benchmark_graph is not None :
        fig = make_subplots(
                            rows= len(θ_t)//2 + len(θ_t)%2 , cols=2,
                            subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(len(θ_t))]
                    )
        for k in range(len(θ_t)):
            print(f"({1 + k//2}, {1 + k%2})")
            fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+k]), row = 1 + k//2 , col = 1 + k%2)
        fig.show()
        
    return θ_t




