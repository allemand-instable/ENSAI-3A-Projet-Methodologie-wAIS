import numpy as np
from typing import Any, Callable
from logging import debug, info
import matplotlib.pyplot as plt


class SamplingPolicy():
    """
    """
    def __init__(self, theta, q_expression) -> None:
        """Constructor

        Args:
            theta (numpy.ndarray):  Parameters Vector Theta
                                    (   example for normal distribution : np.array([μ, σ])  )
            q_expression (_type_): q(θ, x)
        """
        self.theta : np.ndarray = theta
        self.expression : Callable[[np.ndarray, float], float] = q_expression
        pass
    def eval(self, x):
        return self.expression(self.theta, x)
    def show_graph(self, start, stop):
        X = np.linspace(start, stop, num=1000)
        Y = [self.eval(x) for x in X]
        plt.plot(X, Y)
        plt.show()
    def generate_quantile(self):
        q = np.quantile(method=)
        return q

class GaussianSamplingPolicy(SamplingPolicy):
    def __init__(self, μ, σ) -> None:
        
        debug("creating gaussian sampling distrib")
        
        def q_gauss(theta, x):
            mu = theta[0]
            sigma = theta[1]
            return( np.exp(- ((x - mu)**2)/(2*sigma**2)) / (np.sqrt(2*np.pi) * sigma ) )
        
        debug(f"μ = {μ}")
        debug(f"σ = {σ}")
        super().__init__(theta = np.array([μ, σ]), q_expression =q_gauss)

        def generate_quantile(self):
            #normal_unbiased:
            # method 9 of H&F [1]. This method is probably the best method if the sample distribution function is known to be normal. This method gives continuous results using:
            # alpha = 3/8  
            # beta = 3/8

