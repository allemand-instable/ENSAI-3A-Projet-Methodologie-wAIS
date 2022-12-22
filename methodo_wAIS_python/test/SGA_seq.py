import numpy as np
from math_tools.sga_sequential import sga_kullback_leibler_likelihood

from math_tools.normal_family import NormalFamily

def mean_seq(var_init, magnitude = 20, max_iter = 200):

    
    μ_target : float = magnitude * (0.5 - np.random.rand())
    θ_target = np.array([μ_target,1])
    
    print(f"θ_target = {θ_target}")
    
    μ_inital : float = μ_target + (0.5 - np.random.rand())*(2*magnitude)
    θ_initial = np.array([μ_inital, var_init])
    
    print(f"θ_initial = {θ_initial}")
    
    
    target_f = NormalFamily(*θ_target)
    
    intial_q = NormalFamily(*θ_initial)
    
    N = 500
    u = 100
    eta_0 = 1.5
    
    # res = SGA_L(f=target_f, q=intial_q, N=N, γ = u, η_0 = eta_0, iter_limit=max_iter, benchmark=True)   
    res = sga_kullback_leibler_likelihood(
        f_target=target_f,
        q_init=intial_q,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step=eta_0,
        iter_limit = max_iter,
        benchmark=True,
        max_L_gradient_norm=10
    )
    
    d = np.abs(θ_target - res)
    print(d)
    return

