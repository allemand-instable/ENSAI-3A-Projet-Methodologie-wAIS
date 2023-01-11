import numpy as np
# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray
from custom_typing.custom_types import SGA_Params
# My Modules
from distribution_family.distribution_family import DistributionFamily
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood
# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str
from benchmark.combine_error_graphs import combine_error_graph
import benchmark.compare_methods

def compare_sga_methods(
                        f_target : DistributionFamily ,
                        q_init : DistributionFamily , 
                        nb_drawn_samples : int, 
                        nb_stochastic_choice : int, 
                        step : float, 
                        ɛ : float = 1e-6, 
                        iter_limit = 100, 
                        max_L_gradient_norm : int | float = np.Infinity,
                        param_composante : Optional[int] = None ) -> None:
    # adaptive : false
    # weight in grad : false
    last_param_1, graph1 = sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step, None,  ɛ, iter_limit , True, max_L_gradient_norm, False, False, param_composante = param_composante)
    # adaptive : true
    # weight in grad : false
    last_param_2, graph2 = sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step,  None, ɛ, iter_limit , True, max_L_gradient_norm, True, False, param_composante = param_composante)
    # adaptive : true
    # weight in grad : true
    # last_param_3, graph3 =sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step,  None, ɛ, iter_limit , True, max_L_gradient_norm, True, True, param_composante= param_composante)
    
    last_theta_dict = {
        "IS q0": last_param_1,
        "IS qt": last_param_2,
    }
    
    graph_dict = {
        "IS q0": graph1,
        "IS qt": graph2,
    }
    
    color_dict = {
        "IS q0": "#227093",
        "IS qt": "#ff793f",
    }
    
    combine_error_graph(last_theta_dict, f_target.parameters_list(), q_init.parameters_list(), graph_dict, color_dict, nb_drawn_samples, nb_stochastic_choice, step, max_L_gradient_norm)


def compare_methods_2(
                        f_target : DistributionFamily ,
                        q_init : DistributionFamily , 
                        sga_params : SGA_Params ) -> None:
    # method_params = dict(
    #     nb_drawn_samples=nb_drawn_samples, 
    #     nb_stochastic_choice=nb_stochastic_choice, 
    #     step=step,
    #     θ_0=None,  
    #     ɛ=ɛ, 
    #     iter_limit=iter_limit , 
    #     max_L_gradient_norm=max_L_gradient_norm, 
    #     param_composante = param_composante
    # )
    subtitles = {
        "nb_drawn_samples" : sga_params["nb_drawn_samples"], 
        "nb_stochastic_choice" : sga_params["nb_stochastic_choice"], 
        "step" : sga_params["step"], 
        "max_L_gradient_norm" : sga_params["max_L_gradient_norm"]
    }
    
    non_adaptive = lambda f_target, q_init, method_params : sga_kullback_leibler_likelihood(
                                                    f_target=f_target, 
                                                    q_init=q_init, 
                                                    benchmark=True, 
                                                    show_benchmark_graph = True,
                                                    adaptive=False, 
                                                    weight_in_gradient=False, 
                                                    **method_params)
    
    adaptive = lambda f_target, q_init, method_params : sga_kullback_leibler_likelihood(
        f_target, 
        q_init, 
        benchmark=True, 
        show_benchmark_graph = True,
        adaptive=True, 
        weight_in_gradient=False, 
        **method_params)
    
    adaptive_weights_in_grad = lambda f_target, q_init, method_params : sga_kullback_leibler_likelihood(
        f_target, 
        q_init, 
        benchmark=True, 
        show_benchmark_graph = True,
        adaptive=True, 
        weight_in_gradient=True, 
        **method_params)
    
    methods = [non_adaptive, adaptive, adaptive_weights_in_grad]

    
    benchmark.compare_methods.compare_sga_methods(methods, f_target, q_init, subtitles, None, **sga_params)