import numpy as np
from copy import deepcopy
# typing
from typing import Optional, Tuple
from numpy.typing import NDArray
# random
from random import randint
import numpy.random as nprd
from benchmark.combine_error_graphs import BenchmarkGraph
from benchmark.show_error_graph import show_error_graph
# My Modules
from distribution_family.distribution_family import DistributionFamily
# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str
# Kullback Leibler related functions
from kullback_leibler.parameters_initialisation import initialisation
from kullback_leibler.L_gradient.grad_importance_sampling import compute_grad_L_estimator_importance_sampling
from kullback_leibler.L_gradient.grad_weights_in_grad import compute_grad_L_estimator_adaptive




def update_η(η_t : float) -> float:
    """fonction d'update du pas : ici constante, mais devrait être mise en pas variable pour une meilleure convergence"""
    η_t_plus_1 = η_t
    return η_t_plus_1


def cond_n_de_suite__update_state(cond, state : list[bool]) -> list[bool]:
    """lors du Gradient Ascent, on veut arrêter les itérations si on remarque que la norme du gradient est en dessous d'un seuil | pour s'assurer de ne pas être dans un minimum local non global, on regarde si cette condition est vérifiée plusieurs fois d'affilée"""
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


def sga_kullback_leibler_likelihood(    # distributions
                                        f_target : DistributionFamily ,
                                        q_init : DistributionFamily , 
                                        # stochastic part
                                        nb_drawn_samples : int, 
                                        nb_stochastic_choice : int, 
                                        # gradient ascent parameters
                                        step : float, 
                                        θ_0 : Optional[NDArray] = None, 
                                        ɛ : float = 1e-6, 
                                        iter_limit = 100, 
                                        # other parameters
                                        benchmark : bool = False, 
                                        max_L_gradient_norm : int | float = np.Infinity,
                                        adaptive : bool = False,
                                        weight_in_gradient : bool = False,
                                        show_benchmark_graph : bool = False,
                                        # specific sub component of parameter of interest
                                        param_composante : Optional[int] = None
                                    ) -> Tuple[NDArray, Optional[BenchmarkGraph]]:
    """effectue une stochastic gradient ascent pour le problème d'optimisation de θ suivant le critère de la vraissemblance de Kullback-Leibler
        
    f_target                        — target density
                                        ➤ va être utilisée pour la comparaison avec q dans la maximisation de la vraissemblance de Kullback-Leibler
                                        
                                        L(θ) = - KL( f || q )
    
    q_init                          — original sampling policy : q(𝑥, θ)
    
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
    
    iter_limit                      — nombre d'itérations max du gradient descent avant l'arrêt

    benchmark                       — if True, produces error graphs
    
    max_L_gradient_norm             — safety coefficient : if ‖ 𝛁L ‖ > 𝜶 ‖ θ_t ‖
                                        ↪ we use 𝜶 × (𝛁L / ‖ 𝛁L ‖)
                                        ↪ Default : unbound         [ np.Infinity ]
                                        
    adaptive                        — sample X = (𝑥ᵢ)₁,ₙ
                                            ↪ à partir de q_init    [ False ]
                                            
                                            ↪ à partir de qₜ        [ True  ]
                                            
    """
    
    """Initialisation"""
    η_t, θ_t, norm_grad_L, X, q_t, benchmark_graph, state = initialisation(q_init, ɛ, θ_0, nb_drawn_samples, nb_stochastic_choice, step, benchmark)
    
    
    # useful for computing error
    if benchmark_graph is not None :
        target : NDArray = f_target.parameters_list()
        theta_init = deepcopy(θ_t)
    else :
        target = np.array([])
        theta_init = np.array([])
    
    
    # How samples are being drawn depending on whether importance sampling is from a fixed distribution or adaptive
    if not adaptive :
        X = q_init.sample(nb_drawn_samples)
        if X is None :
            X = []
            raise ValueError("generated sample is None !")
    else :
        X = []
    
    # adding the initial θ₀ to the benchmark graph
    if benchmark_graph is not None :
        benchmark_graph[0].append(0)
        for k in range(len(θ_t)) :
                # we add the relative error between θ_t and θ_target
                d_k = np.abs((θ_t[k] - target[k])/(target[k] + 1e-4))
                #####################################################
                benchmark_graph[1+k].append(d_k)
    
    
    """MAIN LOOP"""
    for counter in range(iter_limit):
        
        # si on a atteint l'objectif (on est dans un minimum local et on espère global) on peut arrêter els itérations
        if all(cond_n_de_suite__update_state(norm_grad_L <= ɛ, state)):
            debug(logstr(f"norm_grad_L = {norm_grad_L}"))
            break
        
        # Adaptive ⇒ we sample from the new distribution at each iteration
        if adaptive :
            new_sample : list | None = q_t.sample(nb_drawn_samples)
            # la méthode sample peut retourner un None en cas de problème, on doit gérer le cas
            if new_sample is None :
                raise ValueError(f"could not sample from q_t \n(params = {q_t.parameters})\nnew_sample = None")
            else :
                X = new_sample + X
        
        # non stochastic gradient ascent ⇒ use all X
        if nb_stochastic_choice == nb_drawn_samples :
            X_sampled_from_uniform = X
        # stochastic gradient descent ⇒ use a subset of X
        else :
            obs_tirées = nprd.choice(range(len(X)), nb_stochastic_choice, replace=False)
            X_sampled_from_uniform = [  X[i] for i in obs_tirées  ]
        
        
        # computation of an estimator of 𝛁L
        if adaptive :
            if weight_in_gradient :
                grad_L_estimator = compute_grad_L_estimator_adaptive(f_target, q_t, 
                                                            θ_t, 
                                                            # nb_stochastic_choice,
                                                            max_L_gradient_norm, 
                                                            X_sampled_from_uniform,
                                                            param_composante)
            else :
                grad_L_estimator = compute_grad_L_estimator_importance_sampling(f_target, q_t, q_t,
                                                            θ_t, 
                                                            # nb_stochastic_choice,
                                                            max_L_gradient_norm, 
                                                            X_sampled_from_uniform,
                                                            param_composante)
        else :
            grad_L_estimator = compute_grad_L_estimator_importance_sampling(f_target, q_t, q_init,
                                                        θ_t, 
                                                        # nb_stochastic_choice,
                                                        max_L_gradient_norm, 
                                                        X_sampled_from_uniform,
                                                        param_composante)
        # computation of ‖𝛁L‖
        norm_grad_L = np.linalg.norm(grad_L_estimator)
        
        # gradient ascent
        θ_t = θ_t + η_t * grad_L_estimator
                
        """——— Parameters update ———"""
        q_t.update_parameters(θ_t)
        η_t = update_η(η_t)
        
        #printing every 20 iterations in the console
        if counter % 20 == 0 :
            str_theta = f"θ_{counter} = {θ_t}"
            print(str_theta)
            debug(logstr(str_theta))
            debug(logstr(f"η_t+1 = {η_t}"))
            # todo : one may also implement here a save of the parameter found every 20 iterations, or even better choose a number of iterations before a checkmark (...)
        
        """——— Benchmarking ———"""
        # if we desire to benchmark : we build the error graph
        if benchmark_graph is not None :
            #           X AXIS
            benchmark_graph[0].append(counter+1)
            #? adding the likelihood graph 
            # benchmark_graph[len(θ_t)+1].append(compute_likelihood(f_target=f_target, q_t = q, theta_t=θ_t, q_importance_sampling= q_init, X_sampled_from_uniform = X_sampled_from_uniform))
            
            #           Y AXIS
            for k in range(len(θ_t)) :
                # we add the relative error between θ_t and θ_target
                d_k = np.abs((θ_t[k] - target[k])/(target[k] + 1e-4))
                #####################################################
                benchmark_graph[1+k].append(d_k)
    # à la fin on plot le graphe des erreurs
    if (benchmark_graph is not None) and (show_benchmark_graph is True) :
        show_error_graph(last_θ_t = θ_t, 
                         θ_target = target, 
                         θ_init = theta_init,
                         benchmark_graph = benchmark_graph,
                         # subtitle
                         nb_drawn_samples = nb_drawn_samples, 
                         nb_stochastic_choice = nb_stochastic_choice, 
                         step = step, 
                         max_L_gradient_norm= max_L_gradient_norm
                         )        
    if benchmark :
        return θ_t, benchmark_graph
    else :
        return θ_t, None


