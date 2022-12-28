import numpy as np
from kullback_leibler.compare_methods import compare_sga_methods
from typing import Optional
from distribution_family.normal_family import NormalFamily
from distribution_family.binomial_family import BinomialFamily
from distribution_family.exponential_family import ExponentialFamily
from distribution_family.weibull_family import WeibullFamily
from distribution_family.student_family import StudentFamily
from distribution_family.logistic_family import LogisticFamily


def mean_seq(
    var_init, 
    magnitude = 20, 
    max_iter = 200, 
    step = 1.5, 
    adaptive :bool = False,
    μ_target : Optional[float] = None, 
    θ_target: Optional[float] = None, 
    μ_inital: Optional[float] = None, 
    θ_initial: Optional[float] = None
    ) -> dict[str, float | None]:
    # si il y a un None
    if all([elem is None for elem in [μ_target, θ_target, μ_inital, θ_initial]]) :
        μ_target : float = magnitude * (0.5 - np.random.rand())
        
        print(f"θ_target = {θ_target}")
        
        μ_inital : float = μ_target + (0.5 - np.random.rand())*(2*magnitude)
        
        print(f"θ_initial = {θ_initial}")
    
    θ_target_array = np.array([μ_target,1])
    θ_initial_array = np.array([μ_inital, var_init])
    
    print(f"target param : {θ_target_array}")
    print(f"initial param : {θ_initial_array}")
    
    target_f = NormalFamily(*θ_target_array)
    
    intial_q = NormalFamily(*θ_initial_array)
    
    N = 80
    u = 20
    eta_0 = step
    
    # res = SGA_L(f=target_f, q=intial_q, N=N, γ = u, η_0 = eta_0, iter_limit=max_iter, benchmark=True)   
    # res = sga_kullback_leibler_likelihood(
    #     f_target=target_f,
    #     q_init=intial_q,
    #     nb_drawn_samples=N,
    #     nb_stochastic_choice=u,
    #     step=eta_0,
    #     iter_limit = max_iter,
    #     benchmark=True,
    #     max_L_gradient_norm=50,
    #     adaptive=adaptive
    # )

    # print(f"\n\nres = {res}\n\n\n")
    
    compare_sga_methods(f_target=target_f, q_init=intial_q, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50 )
    
    
    
    return {"μ_target" : μ_target, 
            "θ_target" : θ_target, 
            "μ_inital" : μ_inital, 
            "θ_initial" : θ_initial}




def known_variance(var_init, 
    magnitude = 20, 
    max_iter = 200, 
    step = 1.5, 
    adaptive :bool = False,
    μ_target : Optional[float] = None, 
    θ_target: Optional[float] = None, 
    μ_inital: Optional[float] = None, 
    θ_initial: Optional[float] = None) -> dict[str, float | None]:
    if all([elem is None for elem in [μ_target, θ_target, μ_inital, θ_initial]]) :
        μ_target : float = magnitude * (0.5 - np.random.rand())  
        print(f"θ_target = {θ_target}")
        μ_inital : float = μ_target + (0.5 - np.random.rand())*(2*magnitude)
        print(f"θ_initial = {θ_initial}")
    
    θ_target_array = np.array([μ_target,var_init])
    θ_initial_array = np.array([μ_inital, var_init])
    target_f = NormalFamily(*θ_target_array)
    
    intial_q = NormalFamily(*θ_initial_array)
    
    N = 80
    u = 20
    eta_0 = step
    compare_sga_methods(f_target=target_f, q_init=intial_q, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50, param_composante = 0 )

    
    

    
    return {"μ_target" : μ_target, 
            "θ_target" : θ_target, 
            "μ_inital" : μ_inital, 
            "θ_initial" : θ_initial}


def other_distrib(step = 0.5):
    max_iter = 1000
    N = 80
    u = 20
    eta_0 = step
    
    exp_target = ExponentialFamily(3)
    exp_initial = ExponentialFamily(6)
    compare_sga_methods(f_target=exp_target, q_init=exp_initial, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50)
    
    
    weib_t = WeibullFamily(8)
    weib_i = WeibullFamily(2)
    compare_sga_methods(f_target=weib_t, q_init=weib_i, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50)    
    
    student_t = StudentFamily(3)
    student_i = StudentFamily(6)
    compare_sga_methods(f_target=student_t, q_init=student_i, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50)    
    
    
    
    logistic_t = LogisticFamily(20, 5)
    logistic_i = LogisticFamily(-15, 10)
    compare_sga_methods(f_target=logistic_t, q_init=logistic_i, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50)    
    
    # binom_target = BinomialFamily(20,0.35)
    # binom_initial = BinomialFamily(7,0.8)
    # compare_sga_methods(f_target=binom_target, q_init=binom_initial, nb_drawn_samples=N, nb_stochastic_choice=u, step=eta_0, iter_limit=max_iter, max_L_gradient_norm = 50)