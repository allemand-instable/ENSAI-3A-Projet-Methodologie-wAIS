import numpy as np
from typing import Any, Callable

from utils.log import logstr
from logging import info, debug, warn, error, critical



def gradient_selon(arg_num : int ,f : Callable[[], Any], *args ,h = 1e-6):
    """_summary_

    Args:
        arg_num (int): starts at 1
        
        f (function): f :   R x R^p --> R
                            x    Θ      f(x , Θ)
        h (int, optional): _description_. Defaults to 1.

    Returns:


        for f(u,v,w) :
        u vector len p
        v vector len q
        w vector len r
        
        gradient_selon(1, f)
        [∂f/∂u_1](x) ... [∂f/∂u_p](x)
        
        gradient_selon(2, f)
        [∂f/∂v_1](x) ... [∂f/∂v_q](x)
        
        gradient_selon(2, f)
        [∂f/∂w_1](x) ... [∂f/∂w_r](x)

    """
    info(logstr("—————————— DEBUT GRADIENT DESCENT ——————————"))
    debug(logstr(f"Params :\n\narg_num = {arg_num}\nf = {f}\n\nargs = {args} ∈ {[type(obj) for obj in args]}"))
    
    # index
    index = arg_num-1
    debug(logstr(f"index = {index}"))
    
    
    argument_differencie : np.ndarray = np.array(args[index])
    #                                   on s'assure que on a bien un vecteur numpy
    #                                   si il l'est déjà, il le reste
    #                                   sinon il est transformé en ndarray ( notamment si c'est une liste )
    debug(logstr(f"argument_differencie = {argument_differencie}"))
    
    
    
    p = argument_differencie.size
    debug(f"p = {p}")
    
    
    
    gradient = np.zeros(p)
    
    
    # calcul du gradient
    
    debug(logstr("--- début de calcul de gradient composante par composante ---"))
    # we compute each partial derivative
    for composante_index in range(p):
        debug(logstr(f"pour la composante : {composante_index}"))
        # (u,v,w, ...)
        # on décide de modifier w, un vecteur de longueur p
        H = np.zeros(shape=p)
        # on ajoute h à une des composantes de w
        # ici : composante_index dans [1,p]
        H[composante_index] = h
        theta_plus_h = argument_differencie + H
        debug(logstr(f"theta_plus_h = {theta_plus_h}"))
        # on renvoie (u, v, w', ...)
        # si la composante modifié était w
        debug(logstr(f"args = {args}"))
        new_args = get_new_args(args, index, theta_plus_h)
        debug(logstr(f"new_args = {new_args}"))
        # calcul approché du gradient de f(u,v,w,...) selon w
        gradient_composante = (f(*new_args) - f(*args))/h
        debug(logstr(f"f(new_args) = {f(*new_args)}"))
        debug(logstr(f"f(args) = {f(*args)}"))
        debug(logstr(f"∂{index}_f[{composante_index}] = {gradient_composante}"))
        # grad_w f(u,v,w,...) 
        gradient[composante_index] = gradient_composante
    
    debug(logstr(f"\n[—————————— RÉSULTAT ——————————]\n\ngradient = {gradient}\n\n—————————— FIN GRADIENT DESCENT ——————————\n"))
    
    return(gradient)


def get_new_args(args, index, modified_vec):
    #debug(logstr("=== DEBUT DE GET_NEW_ARGS ==="))
    #debug(logstr(f"index = {index}"))
    #debug(logstr(f"modified vector = {modified_vec}"))
    if index == 0 :
        args_copy = list(args)
        args_copy.pop(0)
        res = [modified_vec] +  args_copy
    else :
        before = [args[k] for k in range(index)]
        after = [args[(index+1) + k] for k in range(len(args)-(index+1))]
        res = before + [modified_vec] + after
        #debug(logstr(f"before = {before}"))
        #debug(logstr(f"after = {after}"))
        #debug(logstr(f"res = {res}"))
    #debug(logstr("=== FIN DE GET_NEW_ARGS ==="))
    return res
