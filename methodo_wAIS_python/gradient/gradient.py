import numpy as np
from typing import Any, Callable, Optional
from numpy.typing import NDArray

from utils.log import logstr
from logging import info, debug, warn, error, critical

from numpy.typing import ArrayLike
from typing import Protocol, Union

vector_or_scalar = NDArray | float

# type hinting
class MultivariateFunction_to_R(Protocol):
    """
    function defined by the relation y = f( 𝑥ᵢ )₁,ₙ
    
    i.e
    
        Π ℝᴸᵉⁿ⁽ˣ-ⁱ⁾ ⟶   ℝ
    f : ( 𝑥ᵢ )₁,ₙ   ⟼   y
    """
    def __call__(self, *float_args : NDArray[np.float64] ) -> float: ...



def gradient_selon(arg_num : int ,f : MultivariateFunction_to_R, *args ,h = 1e-7, composante : Optional[int] = None) -> NDArray:
    """renvoie le gradient d'une fonction multivariée f(... 𝑥ᵢ ...)₁,ₙ selon  𝑥_{arg_num}  évalué en les arguments de f (*args)  |   où 𝑥ᵢ ∈ ℝ^p
    
    si composante = 𝑘   ⟶   renvoie : [𝛁_θₖ]f(x) = [ 0 , ..., [𝜕_θₖ]f(x) , ... , 0 ] ∈ ℝ^p

    Args:
        arg_num (int): starts at 1
        
        f (function): f :   R x R^p ⟶   R
                            (x , Θ)  ⟼   f(x , Θ)
        h (int, optional): finesse de la dérivée. Defaults to 1.
                            doit tendre vers 0 (h ≈ 0)
        
        composante (int, optional) : [𝛁_θ]f(x) ⇒ [𝛁_θ_composante]f(x)
                                     θ = [ θ₁ , θ₂ , ... , θₚ ]
                                     
                                     [𝛁_θ]f(x) = [ [𝜕_θ₁]f(x) , [𝜕_θ₂]f(x) , ... , [𝜕_θₚ]f(x) ]
                                     
                                     va donc renvoyer : [𝛁_θ_composante]f(x) = [ 0 , ..., [𝜕_θcₒₘₚₒₛₐₙₜₑ]f(x) , ... , 0 ]

    Returns:


        for f(u,v,w) :
        u vector len p
        v vector len q
        w vector len r
        
        gradient_selon(1, f)
        [ [∂f/∂u_1](x) ... [∂f/∂u_p](x) ]   ∈ ℝ^p
        
        gradient_selon(2, f)
        [ [∂f/∂v_1](x) ... [∂f/∂v_q](x) ]   ∈ ℝ^q
        
        gradient_selon(3, f)
        [ [∂f/∂w_1](x) ... [∂f/∂w_r](x) ]   ∈ ℝ^r
        
        gradient_selon(3, f, composante = 2)
        = [ 0  [∂f/∂w_2](x)  0  ...  0 ]    ∈ ℝ^r

    """
    debug(logstr(f"Params :\n\narg_num = {arg_num}\nf = {f}\n\nargs = {args} ∈ {[type(obj) for obj in args]}"))
    # index
    index = arg_num-1
    # au cas où quelqu'un donne en argument un float
    argument_differencie : np.ndarray = np.array(args[index])
    #                                   on s'assure que on a bien un vecteur numpy
    #                                   si il l'est déjà, il le reste
    #                                   sinon il est transformé en ndarray ( notamment si c'est une liste )
    # dimension du vecteur différentié [réutilisé un peu partout après]
    p = argument_differencie.size
    #
    gradient = np.zeros(p)
    # calcul du gradient
    # we compute each partial derivative
    # faire selon toutes les composantes du vecteur selon lequel on effectue le gradient
    if composante is None :
        for composante_index in range(p):
            # (u,v,w, ...)
            # on décide de modifier w, un vecteur de longueur p
            H = np.zeros(shape=p)
            # on ajoute h à une des composantes de w
            # ici : composante_index dans [1,p]
            H[composante_index] = h
            theta_plus_h = argument_differencie + H
            # on renvoie (u, v, w', ...)
            # si la composante modifié était w
            new_args = get_new_args(args, index, theta_plus_h)
            # calcul approché du gradient de f(u,v,w,...) selon w
            gradient_composante = (f(*new_args) - f(*args))/h
            # grad_w f(u,v,w,...) 
            gradient[composante_index] = gradient_composante
    
    # en sélectionnant une composante particulière du vecteur selon lequel on effectue le gradient
    else :
        # (u,v,w, ...)
        # on décide de modifier w, un vecteur de longueur p
        H = np.zeros(shape=p)
        # on ajoute h à la composante numéro [composante] de w
        H[composante] = h
        theta_plus_h = argument_differencie + H
        # on renvoie (u, v, w', ...)
        # si la composante modifié était w
        new_args = get_new_args(args, index, theta_plus_h)
        # calcul approché du gradient de f(u,v,w,...) selon w
        gradient_composante = (f(*new_args) - f(*args))/h
        # grad_w f(u,v,w,...) 
        gradient[composante] = gradient_composante    
    debug(logstr(f"∇f = {gradient}\n"))
    return(gradient)


def get_new_args(args, index, modified_vec):
    """donné une liste de vecteurs (potentiellement de dimensions différentes) 
    
    - args = (𝑥ᵢ)₁,ₙ
    - index = 𝑘
    - modified_vec = ⃗u
    
    ie avec un vec initial : [ 𝑥₁, ... , 𝑥ₖ , 𝑥ₖ₊₁, ... 𝑥ₙ  ]
                                         ↓
    retourne le vecteur    : [ 𝑥₁, ... , ⃗u , 𝑥ₖ₊₁, ... 𝑥ₙ   ]
    """
    # si c'est le premier vecteur que l'on remplace, il suffit d'ajouter les autres après
    if index == 0 :
        args_copy = list(args)
        args_copy.pop(0)
        res = [modified_vec] +  args_copy
    # sinon il faut concaténer dans le bon ordre : ceux avant, la modif, ceux après
    else :
        before = [args[k] for k in range(index)]
        after = [args[(index+1) + k] for k in range(len(args)-(index+1))]
        res = before + [modified_vec] + after
    return res
