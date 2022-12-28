import numpy as np
from numpy.typing import NDArray

from benchmark.combine_error_graphs import BenchmarkGraph

import plotly.graph_objects as plgo
from plotly.subplots import make_subplots

def show_error_graph(last_θ_t : NDArray, 
                     θ_target : NDArray, 
                     θ_init : NDArray, 
                     benchmark_graph : BenchmarkGraph,
                     nb_drawn_samples, 
                     nb_stochastic_choice, 
                     step, 
                     max_L_gradient_norm  # subtitle parameters
                    ) -> None:
    """à partir du résultat de la SGA et des paramètres initiaux, produit le graphe des erreurs **relatives** du paramètre obtenu à partir du paramètre qui était visé, et ce en produisant un graphe composante par composante du paramètre θ estimé"""
    if benchmark_graph is None :
        raise TypeError("the benchmark_graph should not be None")
    
    n = len(last_θ_t)
    
    if n == 1 :
        fig = plgo.Figure()
        y_max = max(benchmark_graph[1])
        fig.add_trace(plgo.Scatter(x=benchmark_graph[0] , y=benchmark_graph[1]))
        axis_range_dict = {"yaxis" : dict(range=[0, 1.1 * y_max])}
    else:
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
