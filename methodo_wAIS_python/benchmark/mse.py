from typing import List
from numpy import mean, nanmean

def mse(real_values : List, predicted : List):
    n = len(real_values)
    if n != len(predicted) :
        raise ValueError("not same length")
    else :
        return nanmean( [ (real_values[k] - predicted[k])**2 for k in range(n) ] )