import numpy as np
# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import ArrayLike, NDArray
# My Modules
from distribution_family.distribution_family import DistributionFamily
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood
from custom_typing.custom_types import IterativeParameterMethod

from benchmark.combine_error_graphs import combine_error_graph

colors = [
        "#227093",
        "#ff793f",
        "#218c74",
        "#82589F",
        "#EAB543",
        "#B33771",
        "#1B9CFC",
        "#2C3A47",
        "#5D4037",
        "#C2185B",
        "#D32F2F",
        "#388E3C",
        "#F57C00",
        "#AFB42B"
    ]


def compare_sga_methods(methods : Dict[str, IterativeParameterMethod],
                        f_target : DistributionFamily ,
                        q_init : DistributionFamily , 
                        error_graph_subtitles : Dict[str, Any],
                        color_dict : Optional[Dict[str, str]] = None,
                        **params ) -> None:
    """
    
    ➤ error_graph_subtitles (SGA)
        nb_drawn_samples, nb_stochastic_choice, step, max_L_gradient_norm
        [    int        ,            int      ,float,         float     ]
    """
    
    
    
    last_theta_dict = {}
    graph_dict = {}
    
    if color_dict is None :
        generate_colors = True
        colors_dict = {}
    else :
        generate_colors = False
        colors_dict = color_dict
    
    
    counter = 0
    for name, method in methods.items() :
        last_param, graph =method(f_target, q_init, **params)
        last_theta_dict.update({name : last_param})
        graph_dict.update({name : graph})
        if generate_colors :
            colors_dict.update({name : colors[counter]})    
        
        

    
    combine_error_graph(last_theta_dict, f_target.parameters_list(), q_init.parameters_list(), graph_dict, colors_dict, **error_graph_subtitles)
