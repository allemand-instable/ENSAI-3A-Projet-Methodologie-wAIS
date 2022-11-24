from math_tools.gradient import gradient_selon
import logging
import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike
from pprint import pprint

import test.integrale_polynome_ordre_3 as integrale_polynome_ordre_3

# def fcn(x : NDArray, y : NDArray, z : NDArray) -> float:
#     return np.log2(x) / np.exp(x)

def fcn(x : float, theta : list)-> float:
    μ = theta[0]
    Σ_carre = theta[1]
    π = np.pi
    return (np.exp(-((x - μ)**2)/(2*Σ_carre)))/(np.sqrt(2*π*Σ_carre))


fcn2 = lambda x, theta : (np.exp(-((x - theta[0])**2)/(2*theta[1])))/(np.sqrt(2*np.pi*theta[1]))


def test01():
    #gradient_selon()
    x = np.array([1,2,3,4,5,6])
    print(x)
    print(x.shape)
    y = x.transpose()
    print(x)
    print(np.sqrt(np.dot(x, x)))
    print(y.shape)
    A = np.matrix("1 2 3 ; 4 5 6 ; 7 89 986.268")
    print(A)
    print(A.shape)
    print(type(A))


    x = np.array(1)
    print(x) 
    theta = np.array( [2, 9] )

    arguments = [x, theta]

    dθ_f = gradient_selon(2, fcn, *arguments)
    print(dθ_f)
    
    dθ_f = gradient_selon(2, fcn2, *arguments)
    print(dθ_f)

def main():
    integrale_polynome_ordre_3.main()