from distribution_family.distribution_family import DistributionFamily
from typing import Callable, List, Literal, Optional
from general.stochastic_gradient_descent import gradient_descent
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood

from custom_typing.custom_types import ImportanceSamplingGradientEstimation

from wAIS.get_target_density import get_target_density
from wAIS.squared_relative_distance  import get_density_fcn

def update_sampling_policy( 
                            # iteration
                            t : int,
                            # last information
                            q_t : DistributionFamily, 
                            I_t : float,
                            # initial parameters
                            φ : Callable,
                            π : Callable | DistributionFamily,
                            # data
                            X : List[float],
                            # update parameters 
                            gradient_descent_frequency : int,
                            nb_gradient_steps : int, 
                            # gradient descent parameters
                            gradient_descent__method : Literal["ascent"] | Literal["descent"],
                            gradient_descent__compute_grad_L_importance_sampling : ImportanceSamplingGradientEstimation,
                            gradient_descent__param_composante : Optional[int],
                            gradient_descent__nb_drawn_samples : int,
                            gradient_descent__nb_stochastic_choice : int,
                            gradient_descent__step : float,
                            gradient_descent__update_η : Callable,
                            gradient_descent__max_L_gradient_norm : int | float,
                            # target distribution
                            target_π : bool = True
                        ) -> None:
    if t % gradient_descent_frequency == 0 :
        
        f_target : DistributionFamily = get_target_density( π, φ, I_t, target_π = target_π)
        
        new_θ, _ =  gradient_descent(  
                                    # density
                                    f_target = f_target, 
                                    q_init= q_t,
                                    # data
                                    # given_X= X,
                                    # method
                                    compute_grad_L_importance_sampling=compute_grad_L_importance_sampling,
                                    param_composante= param_composante,
                                    adaptive=True,
                                    # weight_in_gradient=False,
                                    method=method,
                                    # parameters
                                    nb_drawn_samples=nb_drawn_samples,
                                    nb_stochastic_choice=nb_stochastic_choice,
                                    iter_limit= nb_gradient_steps,
                                    step=step,
                                    # update_η= update_η,
                                    max_L_gradient_norm=max_L_gradient_norm,
                                    #  benchmark
                                    #! doesn't work because of dummy family
                                    benchmark=False,
                                    )
        q_t.update_parameters(new_θ)
