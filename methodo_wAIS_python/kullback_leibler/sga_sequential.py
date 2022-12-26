import numpy as np
from copy import deepcopy

# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray

# random
from random import randint
import numpy.random as nprd

# My Modules
from gradient.gradient import gradient_selon
from distribution_family.distribution_family import DistributionFamily

# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str

# Plots
import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as plgo

#? BenchmarkGraph = Optional[List[ List[int]   | List[float] ]]
#                                  iterations  |   erreur relative à composante k ∈ ⟦1,len(θₜ)⟧
#    index :                       0           |   1, ... , n = len(θₜ)
#    passe mieux pour le type hinting même si le vrai est plutôt en haut
BenchmarkGraph = Optional[List[ List[float] ]]

ParamsInitiaux = Tuple[ float,                  # η_t
                        NDArray,                # θ_t
                        float,                  # norm_grad_L
                        List[float],            # X
                        DistributionFamily,     # q_0
                        BenchmarkGraph,         # benchmark_graph
                        List[bool]              # state
                      ]

def initialisation(q : DistributionFamily, ɛ : float, θ_0 : Optional[NDArray], N : int, 𝛾 : float, η_0 : float, benchmark : bool) -> ParamsInitiaux:
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


""""""
def compute_grad_L_estimator(f_target : DistributionFamily, 
                             q : DistributionFamily, 
                             θ_t : NDArray, 
                             nb_stochastic_choice : int,
                             max_L_gradient_norm : int | float, 
                             X_sampled_from_uniform : List[float]
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling
    
    𝛁_θ ∫ f(u)×log[q_θ(u)]du = ∫    f(u)       × 𝛁_θ[log q_θ(u)] du 
                             = ∫ [f(u)/q_θ(u)] × 𝛁_θ[log q_θ(u)] × q_θ(u) du
                             = 𝔼_θ[ (f(u)/q_θ(u)) × 𝛁_θ(log q_θ(u)) ]   
    """
    def ω(x,θ) -> float:
        f_val = f_target.density(x)
        q_val = q.density_fcn(x, θ)
        res = f_val/q_val
        # debug(logstr(f"ω(x,θ) = {res}"))
        return res
    # ⟶ scalaire

    def h(x,θ) -> NDArray:
        # x ⟼ log qₜ(x)
        def log_q(u, theta) -> float :
            return np.log(q.density_fcn(u, theta)) 
        # [𝛁_θ]log qₜ(x)
        res = gradient_selon(2, log_q, *[x, θ] )
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


def compute_grad_L_estimator_adaptive(  f_target : DistributionFamily, 
                                        q_t : DistributionFamily, 
                                        θ_t : NDArray, 
                                        nb_stochastic_choice : int,
                                        max_L_gradient_norm : int | float, 
                                        X_sampled_from_uniform : List[float]
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling avec un q adaptatif
    
    ω_θ = f / q_θ
    on a donc ̂𝛁L = 1/n⋅∑ [𝛁_θ]( ω_θ × log(q_θ) )[X_i]
    """
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
        res = gradient_selon(2, fcn, *[x_i, θ])
        
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



def compute_grad_L_estimator_importance_sampling(f_target : DistributionFamily, 
                             q_t : DistributionFamily, 
                             q_importance_sampling : DistributionFamily,
                             θ_t : NDArray, 
                             nb_stochastic_choice : int,
                             max_L_gradient_norm : int | float, 
                             X_sampled_from_uniform : List[float]
                             ) -> NDArray:
    """calcul de l'estimateur de 𝛁L(θ) obtenu par la loi des grands nombres et la méthode d'Importance Sampling
    
    ω = f / q_importance_sampling
    
    on a donc 𝛁̂L = 1/n⋅∑ [𝛁_θ]( ω × log(q_θ) )[X_i]
                 = 1/n⋅∑  ω[X_i] × [𝛁_θ]log(q_θ)[X_i]"""
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
        res = gradient_selon(2, log_q, *[x, θ] )
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





def show_error_graph(last_θ_t : NDArray, θ_target : NDArray, θ_init : NDArray, benchmark_graph : BenchmarkGraph,
                     nb_drawn_samples, nb_stochastic_choice, step, max_L_gradient_norm  # subtitle parameters
                    ) -> None:
    """à partir du résultat de la SGA et des paramètres initiaux, produit le graphe des erreurs **relatives** du paramètre obtenu à partir du paramètre qui était visé, et ce en produisant un graphe composante par composante du paramètre θ estimé"""
    if benchmark_graph is None :
        raise TypeError("the benchmark_graph should not be None")
    
    n = len(last_θ_t)
    
    fig = make_subplots(
                        rows= n//2 + n%2 , cols=2,
                        subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(n)]
                )
    
    axis_range_dict = {}
    
    for k in range(n):
        # print(f"({1 + k//2}, {1 + k%2})")
        fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+k]), row = 1 + k//2 , col = 1 + k%2)
        y_max = max(benchmark_graph[1+k])
        axis_range_dict[f"yaxis{k+1}"] = dict(range=[0, 1.1 * y_max])
    
    fig.update_xaxes(title_text='iteration')
    fig.update_yaxes(title_text='Relative error to target parameter')
    
    
    
    fig.update_layout(title=f"θ_target = {[round(composante, 2) for composante in θ_target]}      θ_init = {[round(composante, 2) for composante in θ_init]} <br><br><sup>N = {nb_drawn_samples}  |  𝛾 = {nb_stochastic_choice}  | η₀ = {step}  | safety_coeff = {max_L_gradient_norm}</sup>",
    **axis_range_dict)
    fig.show()


def combine_error_graph(list_last_θ_t : Dict[str, NDArray], θ_target : NDArray, θ_init : NDArray, list_benchmark_graph : Dict[str, BenchmarkGraph],
                     color_dict : Dict[str, str], nb_drawn_samples, nb_stochastic_choice, step, max_L_gradient_norm  # subtitle parameters
                    ) -> None:
    """à partir du résultat de la SGA et des paramètres initiaux, produit le graphe des erreurs **relatives** du paramètre obtenu à partir du paramètre qui était visé, et ce en produisant un graphe composante par composante du paramètre θ estimé"""
    if not(len(list_last_θ_t) == len(list_benchmark_graph)) :
        raise ValueError("lists are not of the same lenghth")        
    
    n = [len(list_last_θ_t[key]) for key in list_last_θ_t][0]
    
    fig = make_subplots(
                        rows= n//2 + n%2 , cols=2,
                        subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(n)]
                )
    
    axis_range_dict = {}
    
    y_max_list = []
    
    for k in range(n):
        y_max_list.append( max([max(benchmark_graph[1+k]) for key, benchmark_graph in list_benchmark_graph.items() if benchmark_graph is not None]) )
    
    for key, benchmark_graph in list_benchmark_graph.items() :
        if benchmark_graph is None :
            raise TypeError("the benchmark_graph should not be None")
        for k in range(n):
            # print(f"({1 + k//2}, {1 + k%2})")
            fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+k], name=f"θ_{k} : {key}", marker_color=color_dict[key]), row = 1 + k//2 , col = 1 + k%2)
            y_max = y_max_list[k]
            print(y_max)
            axis_range_dict[f"yaxis{k+1}"] = dict(range=[0, 1.1 * y_max])
        
    fig.update_xaxes(title_text='iteration')
    fig.update_yaxes(title_text='Relative error to target parameter')
    
    
    
    fig.update_layout(title=f"θ_target = {[round(composante, 2) for composante in θ_target]}      θ_init = {[round(composante, 2) for composante in θ_init]} <br><br><sup>N = {nb_drawn_samples}  |  𝛾 = {nb_stochastic_choice}  | η₀ = {step}  | safety_coeff = {max_L_gradient_norm}</sup>",
    **axis_range_dict)
    fig.show()



def sga_kullback_leibler_likelihood(
                                        f_target : DistributionFamily ,
                                        q_init : DistributionFamily , 
                                        nb_drawn_samples : int, 
                                        nb_stochastic_choice : int, 
                                        step : float, 
                                        θ_0 : Optional[NDArray] = None, 
                                        ɛ : float = 1e-6, 
                                        iter_limit = 100, 
                                        benchmark : bool = False, 
                                        max_L_gradient_norm : int | float = np.Infinity,
                                        adaptive : bool = False,
                                        weight_in_gradient : bool = False,
                                        show_benchmark_graph : bool = False
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
    
    η_t, θ_t, norm_grad_L, X, q, benchmark_graph, state = initialisation(q_init, ɛ, θ_0, nb_drawn_samples, nb_stochastic_choice, step, benchmark)
    # useful for computing error
    if benchmark_graph is not None :
        target : NDArray = f_target.parameters_list()
        theta_init = deepcopy(θ_t)
    else :
        target = np.array([])
        theta_init = np.array([])
    
    # new_samples = []
    if not adaptive :
        X = q_init.sample(500)
        if X is None :
            X = []
            raise ValueError("generated sample is None !")
    else :
        X = []
    
    for counter in range(iter_limit):
        if all(cond_n_de_suite__update_state(norm_grad_L <= ɛ, state)):
            debug(logstr(f"norm_grad_L = {norm_grad_L}"))
            break
        
        if adaptive :
            new_sample : list | None = q.sample(nb_drawn_samples)
            # new_samples.append(new_sample)
            if new_sample is None :
                raise ValueError(f"could not sample from q \n(params = {q.parameters})\nnew_sample = None")
            # todo
            # comprendre pourquoi si je mets juste X = new_sample
            # on finit par avoir des variances négatives ?
            else :
                X = new_sample + X
            # X = new_samples.pop(0)
        
        if nb_stochastic_choice == nb_drawn_samples :
            X_sampled_from_uniform = X
        else :
            obs_tirées = nprd.choice(range(len(X)), nb_stochastic_choice, replace=False)
            X_sampled_from_uniform = [  X[i] for i in obs_tirées  ]
        
        
        # 𝛁L
        if adaptive :
            if weight_in_gradient :
                grad_L_estimator = compute_grad_L_estimator_adaptive(f_target, q, 
                                                            θ_t, 
                                                            nb_stochastic_choice,
                                                            max_L_gradient_norm, 
                                                            X_sampled_from_uniform)
            else :
                grad_L_estimator = compute_grad_L_estimator_importance_sampling(f_target, q, q,
                                                            θ_t, 
                                                            nb_stochastic_choice,
                                                            max_L_gradient_norm, 
                                                            X_sampled_from_uniform)
        else :
            grad_L_estimator = compute_grad_L_estimator_importance_sampling(f_target, q, q_init,
                                                        θ_t, 
                                                        nb_stochastic_choice,
                                                        max_L_gradient_norm, 
                                                        X_sampled_from_uniform)
        # ‖𝛁L‖
        norm_grad_L = np.linalg.norm(grad_L_estimator)
        
        # gradient ascent
        θ_t = θ_t + η_t * grad_L_estimator
        
        str_theta = f"θ_{counter} = {θ_t}"
        print(str_theta)
        debug(logstr(str_theta))
        
        # aprameters update
        q.update_parameters(θ_t)
        # print(q.parameters)
        
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


def compare_sga_methods(
                        f_target : DistributionFamily ,
                        q_init : DistributionFamily , 
                        nb_drawn_samples : int, 
                        nb_stochastic_choice : int, 
                        step : float, 
                        ɛ : float = 1e-6, 
                        iter_limit = 100, 
                        max_L_gradient_norm : int | float = np.Infinity) -> None:
    # adaptive : false
    # weight in grad : false
    last_param_1, graph1 = sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step, None,  ɛ, iter_limit , True, max_L_gradient_norm, False, False)
    # adaptive : true
    # weight in grad : false
    last_param_2, graph2 = sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step,  None, ɛ, iter_limit , True, max_L_gradient_norm, True, False)
    # adaptive : true
    # weight in grad : true
    last_param_3, graph3 =sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step,  None, ɛ, iter_limit , True, max_L_gradient_norm, True, True)
    
    last_theta_dict = {
        "IS q0": last_param_1,
        "IS qt": last_param_2,
        "ω in grad":last_param_3
    }
    
    graph_dict = {
        "IS q0": graph1,
        "IS qt": graph2,
        "ω in grad":graph3
    }
    
    color_dict = {
        "IS q0": "#227093",
        "IS qt": "#ff793f",
        "ω in grad": "#218c74"
    }
    
    combine_error_graph(last_theta_dict, f_target.parameters_list(), q_init.parameters_list(), graph_dict, color_dict, nb_drawn_samples, nb_stochastic_choice, step, max_L_gradient_norm)


