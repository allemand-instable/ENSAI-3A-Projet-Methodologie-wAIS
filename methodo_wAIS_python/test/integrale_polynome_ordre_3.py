import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from math_tools.stochastic_gradient_descent_SEQUENTIAL import SGA_L
import numpy.random as nprd
from pprint import pprint
import matplotlib.pyplot
from math_tools.normal_family import NormalFamily
import plotly.express as px
from math_tools.distribution_family import DistributionFamily
from math_tools.gradient import gradient_selon

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


# q = DistributionFamily(nprd.normal, θ_0 )
# X = q.sample(2000)
# pprint(X)
# px.histogram(X, nbins=50).show()

def pre_calc_grad_f(x,y,z):
    a = (x[0] + x[1] + z[0])
    p = x[0]*x[1]
    s = (x[0]+z[0])
    return np.array( [p, s , 1/(a * (y[2])**2)] )

def f(x,y,z):
    a = (x[0] + x[1] + z[0])
    p = x[0]*x[1]
    s = (x[0]+z[0])
    return y[0]*p + y[1]*s - 1/(a*(y[2]))

def test_gradient():
    x = np.array([1,2])
    y = np.array([8,4,5])
    z = np.array([9])
    args = [ x, y, z  ]
    print(f"grad_selon = {gradient_selon(2, f, *args)}")
    print(f"pre_calc = {pre_calc_grad_f(*args)}")

def main():
        
    determiner_theta_opt()
    
    #test_gradient()
    
    # 1077.75
    
from math_tools.stochastic_gradient_ascent_POO import SGA_KullbackLeibler_Likelihood

def determiner_theta_opt():
    # 𝒩( 5, 9 )
    # θ_0 = {
    #     "loc" : 5,
    #     "scale" : 90
    # }

    θ_0 = [10,8]
    q = NormalFamily(*θ_0)
    f = NormalFamily(μ=7, Σ = 1)
    X = q.sample(650)
    
    # sga = SGA_KullbackLeibler_Likelihood(   f = f, 
    #                                         q = q, 
    #                                         nb_drawn_samples = 100, 
    #                                         𝛄 = 40, 
    #                                         η_0 = 1, 
                                            
    #                                         iter_limit=2000
    #                                         )
    # sga.execute()
    # pprint(sga.θ_t)

    SGA_L(f,q,100, 30, 1, iter_limit=2000)


# def calc_int_analytique():
#     coeffs = [3, 2, 42]
    
#     int_analytique = int_P_eval(A,B, *coeffs)
#     print(int_analytique)



