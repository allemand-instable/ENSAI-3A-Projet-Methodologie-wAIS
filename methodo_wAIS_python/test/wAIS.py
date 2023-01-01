from wAIS.wAIS import weighted_adaptive_importance_sampling, default_params_KL, default_params_R
from distribution_family.normal_family import NormalFamily
from distribution_family.uniform_family import UniformFamily
import numpy as np
import kullback_leibler.L_gradient.grad_importance_sampling as kl_grad_is
import renyi_alpha_divergence.renyi_importance_sampling_gradient_estimator as r_grad_is

q_init = NormalFamily(5, 2)

sqrt_2_pi = np.sqrt(2*np.pi)

def int_P(A, B, a, b, c):
    return (a/4)*(B**4 - A**4) + (b/3)*(B**3 - A**3) + (c/2)*(B**2-A**2)

def P(x, a, b, c):
    return a*(x**3) + b*(x**2) + c*x


from distribution_family.uniform_family import UniformFamily

from numpy.random import randint

def main():
    
    a = randint(-3, 3)
    b = randint(-2, 2)
    c = randint(-7, 5)
    
    A = randint(-1, 4)
    B = A + randint(4, 7)
    π = UniformFamily(A, B)
    def φ(x) : 
        return   P(x, a, b, c)
    
    print(f"∫P(u)du = {int_P(A,B,a,b,c)}")
    
    print(A, B)
    pi = UniformFamily(A,B)
    
    print(pi.sample(50))
    
    approx_int1 = weighted_adaptive_importance_sampling(lambda x : φ(x)/pi.density(x), pi, UniformFamily(-10,10), default_params_KL)
    
    
    approx_int2 = weighted_adaptive_importance_sampling(lambda x : φ(x)/pi.density(x), pi, UniformFamily(-10,15), default_params_R)
    
    print("———————————————————————————————————————")
    print(f"∫P(u)du = {int_P(A,B,a,b,c)}")
    print(f"approx_int = {approx_int1}")
    print("———————————————————————————————————————")
    print("———————————————————————————————————————")
    print(f"∫P(u)du = {int_P(A,B,a,b,c)}")
    print(f"approx_int = {approx_int2}")
    print("———————————————————————————————————————")