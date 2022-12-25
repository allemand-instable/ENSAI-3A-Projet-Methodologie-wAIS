import numpy as np
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood
from typing import Optional
from distribution_family.normal_family import NormalFamily

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
    ):
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
    res = sga_kullback_leibler_likelihood(
        f_target=target_f,
        q_init=intial_q,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step=eta_0,
        iter_limit = max_iter,
        benchmark=True,
        max_L_gradient_norm=50,
        adaptive=adaptive
    )

    print(f"\n\nres = {res}\n\n\n")
    
    return {"μ_target" : μ_target, 
            "θ_target" : θ_target, 
            "μ_inital" : μ_inital, 
            "θ_initial" : θ_initial}

