import numpy as np
from typing import Callable
NumpyRandomSampler = Callable

def importance_sampling(sampler : NumpyRandomSampler, weight_function : Callable, h : Callable, n : int, params):
    
    X = np.array(sampler( *params ,size = n))
    
    term = lambda u : weight_function(u)*h(u, params)
    
    result = term(X).mean()

    
    return result

w = lambda u : 1
h = lambda u, p : p[0]*np.log(u) + p[1]*np.exp(u)

print(importance_sampling(np.random.normal, w, h, 100, np.array([3,1])))