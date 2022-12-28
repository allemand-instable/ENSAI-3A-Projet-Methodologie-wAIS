import numpy as np
# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray
# My Modules
from distribution_family.distribution_family import DistributionFamily
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood
# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str
from benchmark.combine_error_graphs import combine_error_graph

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
    last_param_3, graph3 =sga_kullback_leibler_likelihood(f_target, q_init, nb_drawn_samples, nb_stochastic_choice, step,  None, ɛ, iter_limit , True, max_L_gradient_norm, True, True, param_composante= param_composante)
    
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
