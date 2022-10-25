import numpy as np
from proba.sampling_policy import SamplingPolicy, GaussianSamplingPolicy
import matplotlib.pyplot as plt

def main() :
    X = []
    q = GaussianSamplingPolicy(0,1)
    for k in range(500):
        u = np.random.uniform()
        x = q.eval(u)
        X.append(x)
        #print(x)
    plt.hist(X)
    plt.show()