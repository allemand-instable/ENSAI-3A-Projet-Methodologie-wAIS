import unittest
import numpy as np
from math_tools.stochastic_gradient_descent_SEQUENTIAL import SGA_L
from math_tools.stochastic_gradient_ascent_POO import SGA_KullbackLeibler_Likelihood

class TestGradientAscent(unittest.TestCase):
    def test_SGA_sequential_mean(self):
        
        magnitude : int = 3
        

        
        μ_target : float = magnitude * (0.5 - np.random.rand())
        θ_target = np.array([μ_target,1])
        
        print(f"θ_target = {θ_target}")
        
        μ_inital : float = magnitude * (0.5 - np.random.rand())
        θ_initial = np.array([μ_inital, 1])
        
        print(f"θ_initial = {θ_initial}")
        
        res = SGA_L()
        
        d = np.abs(θ_target - res)
        
        self.assertLessEqual(d, 0.3)
        
    
    def test_SGA_POO_mean(self):
        pass
    
if __name__ == '__main__':
    unittest.main()