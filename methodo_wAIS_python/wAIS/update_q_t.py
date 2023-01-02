import time
from custom_typing.custom_types import ImportanceSamplingGradientEstimation, UpdateParameters
from general.stochastic_gradient_descent import gradient_descent
from general.importance_sampling import importance_sampling_given_sample

from distribution_family.distribution_family import DistributionFamily
from wAIS.get_target_density import get_target_density

from typing import List, Dict, Any, Callable, Literal, Optional, Tuple, TypedDict

    



def iteration_condition(t : int, frequency : int):
    return t % frequency == 0


def update_qₜ(
            t : int, 
            # gradient descent 
            qₜ : DistributionFamily, 
            Xₜ : List, 
            # target density
            φ : Callable,
            π : DistributionFamily,
            Iₜ : float,
            # other parameters
            update_params : UpdateParameters
            ) -> None:
    """updates qₜ
    ▶ t : int
    
    used for sgd
    ▶ qₜ : DistributionFamily,
    ▶ Xₜ : List,
    
    used for defining target density (optimal)
    ▶ φ : Callable,
    ▶ π : DistributionFamily,
    ▶ Iₜ : float,
    
    update_params :
        ▶ custom_type [ UpdateParameters ]
    """
    if iteration_condition(t, update_params["frequency"]):
        target_density = get_target_density(π,φ,Iₜ) # π | φ - I |
        
        
        
        # gradient descent
        start = time.process_time()
        θₜ , _ = gradient_descent(
                        given_X  = Xₜ,
                        # distributions
                        f_target = target_density ,
                        q_init = qₜ, 
                        # function to be computed
                        compute_grad_L_importance_sampling =update_params["gradient_descent__compute_grad_L_importance_sampling"],
                        # stochastic part
                        nb_drawn_samples =None,
                        nb_stochastic_choice =update_params["gradient_descent__nb_stochastic_choice"], 
                        # gradient ascent parameters
                        step =update_params["gradient_descent__step"], 
                        θ_0 =None, 
                        ɛ = 1e-6, 
                        iter_limit = update_params["gradient_descent__iter_limit"], 
                        # other parameters
                        method = update_params["gradient_descent__method"],
                        update_η  = update_params["gradient_descent__update_η"],
                        benchmark = False, 
                        max_L_gradient_norm  = update_params["gradient_descent__max_L_gradient_norm"],
                        adaptive = update_params["gradient_descent__adaptive"],
                        show_benchmark_graph = False,
                        # specific sub component of parameter of interest
                        param_composante = update_params["gradient_descent__param_composante"],
        )
        print(f"sga_time : {time.process_time() - start}")
        qₜ.update_parameters(θₜ)