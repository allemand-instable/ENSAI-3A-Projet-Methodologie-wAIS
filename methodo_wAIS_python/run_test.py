from math_tools.gradient import gradient_selon
import logging
import numpy as np
from pprint import pprint
from utils.log import logstr
from logging import info, debug, warn, error


import test.integrale_polynome_ordre_3 as integrale_polynome_ordre_3
import test.SGA as SGA
import test.SGA_seq as SGAseq

# def test01():
#     fcn = lambda x : x
#     fcn2 = lambda x: x
#     #gradient_selon()
#     x = np.array([1,2,3,4,5,6])
#     print(x)
#     print(x.shape)
#     y = x.transpose()
#     print(x)
#     print(np.sqrt(np.dot(x, x)))
#     print(y.shape)
#     A = np.matrix("1 2 3 ; 4 5 6 ; 7 89 986.268")
#     print(A)
#     print(A.shape)
#     print(type(A))


#     x = np.array(1)
#     print(x) 
#     theta = np.array( [2, 9] )

#     arguments = [x, theta]

#     dθ_f = gradient_selon(2, fcn, *arguments)
#     print(dθ_f)
    
#     dθ_f = gradient_selon(2, fcn2, *arguments)
#     print(dθ_f)

def main():
    SGAseq.mean_seq()