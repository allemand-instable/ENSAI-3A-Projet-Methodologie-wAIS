# typing
from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from numpy.typing import NDArray
# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str
# Plots
from plotly.subplots import make_subplots
import plotly.graph_objects as plgo

#? BenchmarkGraph = Optional[List[ List[int]   | List[float] ]]
#                                  iterations  |   erreur relative à composante k ∈ ⟦1,len(θₜ)⟧
#    index :                       0           |   1, ... , n = len(θₜ)
#    passe mieux pour le type hinting même si le vrai est plutôt en haut
BenchmarkGraph = Optional[List[ List[float] ]]


def combine_error_graph(list_last_θ_t : Dict[str, NDArray], 
                        θ_target : NDArray, 
                        θ_init : NDArray, 
                        list_benchmark_graph : Dict[str, BenchmarkGraph],
                    color_dict : Dict[str, str], 
                    nb_drawn_samples : int, 
                    nb_stochastic_choice : int, 
                    step : float, 
                    max_L_gradient_norm : float,  # subtitle parameters
                    ) -> None:
    """à partir du résultat de la SGA et des paramètres initiaux, produit le graphe des erreurs **relatives** du paramètre obtenu à partir du paramètre qui était visé, et ce en produisant un graphe composante par composante du paramètre θ estimé"""
    if not(len(list_last_θ_t) == len(list_benchmark_graph)) :
        raise ValueError("lists are not of the same lenghth")        
    
    n = [len(list_last_θ_t[key]) for key in list_last_θ_t][0]
    
    if n == 1 :
        fig = plgo.Figure()
        axis_range_dict = {}
        y_max_list = []
        
        y_max_list.append( max([max(benchmark_graph[1]) for key, benchmark_graph in list_benchmark_graph.items() if benchmark_graph is not None]) )
        
        for key, benchmark_graph in list_benchmark_graph.items() :
            if benchmark_graph is None :
                raise TypeError("the benchmark_graph should not be None")
            fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1], name=f"θ : {key}", marker_color=color_dict[key]))
            y_max = y_max_list[0]
            # print(y_max)
            axis_range_dict[f"yaxis"] = dict(range=[0, 1.1 * y_max])
    
    else :
        #? adding likelihood graph
        # len_benchmark_graph = [len(list_benchmark_graph[key]) for key in list_benchmark_graph][0]
        # if len_benchmark_graph == len(θ_target) +1  :
        fig = make_subplots(
                            rows= n//2 + n%2 , cols=2,
                            subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(n)]
                    )
        #? adding likelihood graph
        # elif len_benchmark_graph == len(θ_target) + 2  :
        #     fig = make_subplots(
        #                         rows= n//2 + n%2 +1, cols=2,
        #                         subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(n)]
        #                 )
        # else :
        #     raise Exception("wrong list lenghth for benchmark_graph")
        axis_range_dict = {}
        y_max_list = []
        
        for k in range(n):
            y_max_list.append( max([max(benchmark_graph[1+k]) for key, benchmark_graph in list_benchmark_graph.items() if benchmark_graph is not None]) )
        
        for key, benchmark_graph in list_benchmark_graph.items() :
            if benchmark_graph is None :
                raise TypeError("the benchmark_graph should not be None")
            for k in range(n):
                # debug(f"({1 + k//2}, {1 + k%2})")
                fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+k], name=f"θ_{k} : {key}", marker_color=color_dict[key]), row = 1 + k//2 , col = 1 + k%2)
                y_max = y_max_list[k]
                # debug(y_max)
                axis_range_dict[f"yaxis{k+1}"] = dict(range=[0, 1.1 * y_max])
            #? addding likelihood graph 
            # if len_benchmark_graph == len(θ_target) + 2  :
            #     fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1+n], name=f"L", marker_color=color_dict[key]), row = n//2 + n%2 +1 , col = 1)
        
    fig.update_xaxes(title_text='iteration')
    fig.update_yaxes(title_text='Relative error to target parameter')
        
    fig.update_layout(title=f"θ_target = {[round(composante, 2) for composante in θ_target]}      θ_init = {[round(composante, 2) for composante in θ_init]} <br><br><sup>N = {nb_drawn_samples}  |  𝛾 = {nb_stochastic_choice}  | η₀ = {step}  | safety_coeff = {max_L_gradient_norm}</sup>",
    **axis_range_dict)
    fig.show()
    return
