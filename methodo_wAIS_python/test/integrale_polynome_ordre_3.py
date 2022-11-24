import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from math_tools.stochastic_gradient_descent import SGD_L
import numpy.random as nprd
from pprint import pprint
import matplotlib.pyplot

A = -2
B = 7

def P(x, a, b, c):
    return Polynomial( (0,c,b,a), domain=(A,B))

def P_eval(x, a, b, c):
    return polyval(x, [0,c,b,a])


# $$ax^3 + bx^2 + cx$$

def int_P_eval(A, B , a, b, c):
    A2, B2 = A*A , B*B
    A3, B3 =  A2*A , B2*B
    A4, B4 = A2*A2, B2*B2
    return (a/4) * (B4 - A4) + (b/3) * (B3 - A3) - (c/2) * (B2-A2)

def g(x, a, b, c):
    return np.exp(x**2/2) * (np.sqrt(2* np.pi)) * P_eval(x, a, b, c)

def h(x) :
    return np.exp(- x**2/2)/(np.sqrt(2*np.pi))
# h = 𝒩(0,1) est la vraie densité, on va tenter de l'approximer en faisant partir q de loin :

# famille normale 𝒩
def norm(x, θ):
    μ   = θ[0]
    Σ   = θ[1]
    cste_norm = 1/(np.sqrt(2*np.pi))
    

# 𝒩( 5, 9 )
θ_0 = {
    "loc" : 5,
    "scale" : 3
}

θ_0 = [5,3]

from math_tools.distribution_family import DistributionFamily


# q = DistributionFamily(nprd.normal, θ_0 )
# X = q.sample(2000)
# pprint(X)

from math_tools.normal_family import NormalFamily

q = NormalFamily(*θ_0)
X = q.sample(650)

import plotly.express as px
px.histogram(X, nbins=50).show()

def main():
    
    coeffs = [3, 2, 42]
    
    int_analytique = int_P_eval(A,B, *coeffs)
    print(int_analytique)
    
    # 1077.75
    
