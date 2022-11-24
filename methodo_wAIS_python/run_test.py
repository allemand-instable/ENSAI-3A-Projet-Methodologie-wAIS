from math_tools.gradient import gradient_selon
import logging
import numpy as np
from numpy.typing import NDArray
from pprint import pprint

# def fcn(x : NDArray, y : NDArray, z : NDArray) -> float:
#     return np.log2(x) / np.exp(x)

def main():
    #gradient_selon()
    x = np.array([1,2,3,4,5,6])
    print(x)
    print(x.shape)
    y = x.transpose()
    print(x)
    print(y)
    print(y.shape)
    A = np.matrix("1 2 3 ; 4 5 6 ; 7 89 986.268")
    print(A)
    print(A.shape)
    print(type(A))
