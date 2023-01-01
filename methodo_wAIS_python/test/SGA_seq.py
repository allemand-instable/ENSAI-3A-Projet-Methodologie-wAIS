import numpy as np
from kullback_leibler.compare_methods import compare_sga_methods
from kullback_leibler.sga_sequential import sga_kullback_leibler_likelihood
from benchmark.show_error_graph import show_error_graph
from typing import Optional
from distribution_family.normal_family import NormalFamily
from distribution_family.binomial_family import BinomialFamily
from distribution_family.exponential_family import ExponentialFamily
from distribution_family.weibull_family import WeibullFamily
from distribution_family.student_family import StudentFamily
from distribution_family.logistic_family import LogisticFamily
from general.stochastic_gradient_descent import gradient_descent
from kullback_leibler.L_gradient.grad_importance_sampling import compute_grad_L_estimator_importance_sampling
from kullback_leibler.L_gradient.grad_importance_sampling import compute_grad_L_estimator_importance_sampling as K_grad_L
from benchmark.combine_error_graphs import combine_error_graph
from renyi_alpha_divergence.renyi_importance_sampling_gradient_estimator import compute_grad_L_estimator_importance_sampling as R_grad_L, renyi_gradL

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

# la méthode généralisée marche bien

def generalized_code():
    θ_target_array  = np.array([1,1])
    θ_initial_array = np.array([12, 9])
    
    target_f = NormalFamily(*θ_target_array)
    intial_q = NormalFamily(*θ_initial_array)
    
    N = 80
    u = 20
    
    res1, graph1 = sga_kullback_leibler_likelihood(target_f, intial_q, N, u, 0.2,  None, ɛ = 1e-5, iter_limit = 1000 ,  benchmark=True, max_L_gradient_norm=50, adaptive = True, weight_in_gradient= False, param_composante = None, show_benchmark_graph=False)
    
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    res2, graph2 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= compute_grad_L_estimator_importance_sampling,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False
    )
    
    # show_error_graph(res2, θ_target_array, θ_initial_array, graph2, N, u, 0.2, 50)
    combine_error_graph({"specialKL" : res1, "general" : res2}, θ_target_array, θ_initial_array, 
                        {"specialKL" : graph1, "general" : graph2}, 
                        {"specialKL" : "#6ab04c", "general" : "#be2edd"},
                        N, u, 0.2, 50)
    
    
    θ_target_array  = np.array([1,1])
    θ_initial_array = np.array([5,1])
    
    target_f = NormalFamily(*θ_target_array)
    intial_q = NormalFamily(*θ_initial_array)
    
    
    
    res1, graph1 = sga_kullback_leibler_likelihood(target_f, intial_q, N, u, 0.2,  None, ɛ = 1e-5, iter_limit = 1000 ,  benchmark=True, max_L_gradient_norm=np.inf, adaptive = True, weight_in_gradient= False, param_composante = 0, show_benchmark_graph=False)
    
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    res2, graph2 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= compute_grad_L_estimator_importance_sampling,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )
    
    # show_error_graph(res2, θ_target_array, θ_initial_array, graph2, N, u, 0.2, 50)
    combine_error_graph({"specialKL" : res1, "general" : res2}, θ_target_array, θ_initial_array, 
                        {"specialKL" : graph1, "general" : graph2}, 
                        {"specialKL" : "#6ab04c", "general" : "#be2edd"},
                        N, u, 0.2, 50)
    
    
def renyi_vs_kullback_knwon_var() -> None:
    θ_target_array  = np.array([1,1])
    θ_initial_array = np.array([12, 9])
    
    target_f = NormalFamily(*θ_target_array)
    intial_q = NormalFamily(*θ_initial_array)
    
    N = 80
    u = 20
        
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    θ_target_array  = np.array([1,1])
    θ_initial_array = np.array([5,1])
    
    target_f = NormalFamily(*θ_target_array)
    intial_q = NormalFamily(*θ_initial_array)
    
    
    
    res1, graph1 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= K_grad_L,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=np.inf,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )    
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    # Entropie de Hartley
    res2, graph2 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= give_estimator(0),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )
    
    #  entropie de collision
    res3, graph3 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= give_estimator(2),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=np.inf,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )
    
    # 5
    res4, graph4 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= give_estimator(5),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=np.inf,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )
    
    # α=30
    res5, graph5 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= give_estimator(30),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=np.inf,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= 0
    )
    
    # show_error_graph(res2, θ_target_array, θ_initial_array, graph2, N, u, 0.2, 50)
    combine_error_graph({"KL" : res1, "α=0" : res2, "α=2" : res3, "α=5" : res4, "α=30" : res5}, θ_target_array, θ_initial_array, 
                        {"KL" : graph1, "α=0" : graph2, "α=2" : graph3, "α=5" : graph4, "α=30" : graph5}, 
                        {"KL" : "#6ab04c", "α=0" : "#be2edd", "α=2" : "#0abde3", "α=5" : "#ff9f43", "α=30" :  "#f368e0"},
                        N, u, 0.2, np.inf)
    
def renyi_vs_kullback_unknwon_var() -> None:

    N = 80
    u = 20
        
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    θ_target_array  = np.array([7,5])
    θ_initial_array = np.array([15,9])
    
    target_f = NormalFamily(*θ_target_array)
    intial_q = NormalFamily(*θ_initial_array)
    
    
    
    res1, graph1 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= K_grad_L,
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= None
    )    
    # show_error_graph(res1, θ_target_array, θ_initial_array, graph1, N, u, 0.2, 50)
    
    # Entropie de Hartley
    res2, graph2 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= renyi_gradL(0),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="ascent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= None
    )
    
    #  entropie de collision
    res3, graph3 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= renyi_gradL(2),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= None
    )
    
    # α=5
    res4, graph4 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= renyi_gradL(5),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= None
    )
    
    
    # α=30
    res5, graph5 = gradient_descent(
        f_target= target_f,
        q_init= intial_q,
        compute_grad_L_importance_sampling= renyi_gradL(30),
        nb_drawn_samples=N,
        nb_stochastic_choice=u,
        step= 0.2,
        method="descent",
        iter_limit=1000,
        adaptive=True,
        ɛ=1e-5,
        max_L_gradient_norm=50,
        benchmark=True,
        show_benchmark_graph=False,
        param_composante= None
    )
    
    
    # show_error_graph(res2, θ_target_array, θ_initial_array, graph2, N, u, 0.2, 50)
    combine_error_graph({"KL" : res1, "α=0" : res2, "α=2" : res3, "α=5" : res4, "α=30" : res5}, θ_target_array, θ_initial_array, 
                        {"KL" : graph1, "α=0" : graph2, "α=2" : graph3, "α=5" : graph4, "α=30" : graph5}, 
                        {"KL" : "#6ab04c", "α=0" : "#be2edd", "α=2" : "#0abde3", "α=5" : "#ff9f43", "α=30" :  "#f368e0"},
                        N, u, 0.2, 50)