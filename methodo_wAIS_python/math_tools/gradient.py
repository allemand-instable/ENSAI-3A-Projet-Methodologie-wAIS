import numpy as np
from logging import debug, info
from typing import Any, Callable


def gradient_selon(arg_num : int ,f : Callable[[], Any], *args ,h = 1e-7)->Callable[[], Any]:
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
    info(f"=== DEBUT GRADIENT DESCENT ===")
    index = arg_num-1
    debug(f"index = {index}")
    argument_differencie : np.ndarray = args[index]
    debug(f"argument_differencie = {argument_differencie}")
    p = argument_differencie.size
    debug(f"p = {p}")
    gradient = np.zeros(p)
    
    debug("--- début de calcul de gradient composante par composante ---")
    # we compute each partial derivative
    for composante_index in range(p):
        debug(f"pour la composante : {composante_index}")
        # (u,v,w, ...)
        # on décide de modifier w, un vecteur de longueur p
        H = np.zeros(shape=p)
        # on ajoute h à une des composantes de w
        # ici : composante_index dans [1,p]
        H[composante_index] = h
        theta_plus_h = argument_differencie + H
        debug(f"theta_plus_h = {theta_plus_h}")
        # on renvoie (u, v, w', ...)
        # si la composante modifié était w
        debug(f"args = {args}")
        new_args = get_new_args(args, index, theta_plus_h)
        debug(f"new_args = {new_args}")
        # calcul approché du gradient de f(u,v,w,...) selon w
        gradient_composante = (f(*new_args) - f(*args))/h
        # grad_w f(u,v,w,...) 
        gradient[composante_index] = gradient_composante
        debug(f"gradient = {gradient}")
    
    info("=== FIN GRADIENT DESCENT ===")
    return(gradient)


def get_new_args(args, index, modified_vec):
    info("=== DEBUT DE GET_NEW_ARGS ===")
    debug(f"index = {index}")
    debug(f"modified vector = {modified_vec}")
    if index == 0 :
        args_copy = list(args)
        args_copy.pop(0)
        res = [modified_vec] +  args_copy
    else :
        before = [args[k] for k in range(index)]
        after = [args[(index+1) + k] for k in range(len(args)-(index+1))]
        res = before + [modified_vec] + after
        debug(f"before = {before}")
        debug(f"after = {after}")
        debug(f"res = {res}")
    info("=== FIN DE GET_NEW_ARGS ===")
    return res
