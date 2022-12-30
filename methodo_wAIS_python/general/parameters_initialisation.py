from typing import Optional
from numpy.typing import NDArray
from custom_typing.custom_types import ParamsInitiaux
from distribution_family.distribution_family import DistributionFamily
# Debug
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str




def initialisation(q : DistributionFamily, 
                   ɛ : float, 
                   θ_0 : Optional[NDArray], 
                   N : int, 
                   𝛾 : float, 
                   η_0 : float, 
                   benchmark : bool, 
                   ) -> ParamsInitiaux:
    """initialisationdes paramètres pour la SGA de la fonction L
    
    
    Inputs : 
    
        q                               — original sampling policy : q(𝑥, θ)
        
        ɛ                               — threshold pour la norme du gradient
        
        θ_0                             — initialisation des paramètres
        
        nb_drawn_samples (N)            — Nombre de samples tirés par la distribution q à chaque itération
        
        nb_stochastic_choice (𝛾)        — nombre d'observations à tirer aléatoirement
        
        step (η_0)                      — initialisation du pas
        
        benchmark                       - if True, produces error graphs




    Return :
    
        η_t, θ_t, norm_grad_L, X, q_0, counter, benchmark_graph, state
        
        η_t             : float
        θ_t             : NDArray
        norm_grad_L     : float
        X               : list[float]
        q_0             : DistributionFamily
        benchmark_graph : list[list[float]] | None
        state           : list[bool]
    
    """
    
    """TYPES DEFINITION"""
    η_t             : float
    θ_t             : NDArray
    norm_grad_L     : float
    X               : list[float]
    q_0             : DistributionFamily
    counter         : int
    benchmark_graph : list[list[float]] | None
    state           : list[bool]
    
    # initialisation
    η_t = η_0
    if θ_0 is None :
        θ_0 = q.parameters_list()
        θ_t = q.parameters_list()
    else :
        θ_t = θ_0
        q.update_parameters(θ_0)
    
    # on s'assure de commencer la première itération
    norm_grad_L = (ɛ + 1)
    
    X = []
    
    debug(logstr(
        f"\nη_t = {η_t}\nθ_t = {θ_t}\n𝛾 = {𝛾}\nN = {N}\n"
                ))
    
    #! importance sampling selon q(θ_0)
    q_0 = q.copy()
        
    
    if benchmark is True :
        benchmark_graph = [ list([]) for k in range( len(θ_t) + 2)]
        #                                                     + 1
        #                                            if you don't want to include
        #                                            the graph (inter, L(θ)) 
    else :
        benchmark_graph = None

    state = [False for k in range(3)]
    
    return η_t, θ_t, norm_grad_L, X, q_0, benchmark_graph, state
