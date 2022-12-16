import unittest
import numpy as np
import numpy.typing as npt

from math_tools.gradient import gradient_selon
import numpy.polynomial.polynomial as npp 

def P(X : float,Y : float,Z : float, coeffs_x : npt.NDArray, coeffs_y : npt.NDArray, coeffs_z : npt.NDArray, interaction_x_y : float = 0, interaction_y_z : float = 0, interaction_x_z : float = 0, interaction_x2_y : float = 0, interaction_x_z2 : float = 0) -> float:
    
    return np.polyval(coeffs_x, X) + np.polyval(coeffs_y, Y) + np.polyval(coeffs_z, Z) + interaction_x_y*X*Y + interaction_y_z*Z*Y + interaction_x_z*X*Z + interaction_x2_y*X*X*Y + interaction_x_z2*X*Z*Z


def grad_z_P(X : float,Y : float,Z : float, coeffs_x : npt.NDArray, coeffs_y : npt.NDArray, coeffs_z : npt.NDArray, interaction_x_y : float = 0, interaction_y_z : float = 0, interaction_x_z : float = 0, interaction_x2_y : float = 0, interaction_x_z2 : float = 0):
    
    new_coeffs = [ (len(coeffs_z) - k)*coeffs_z[k] for k in range(len(coeffs_z)-1) ]
    
    return np.polyval( new_coeffs ,x= Z) + interaction_y_z*Y + interaction_x_z*X + 2*interaction_x_z2*X*Z

class TestGradient(unittest.TestCase):
    
    epsilon = 1e-4
    
    def test_gradient_polynome(self):
        
        
        
        
        self.assertLessEqual(true_gradient, gradient_algo_result)
        return
    def test_polynome_sin(self):
        return
    