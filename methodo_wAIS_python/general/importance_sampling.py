from typing import List, Callable
from distribution_family.distribution_family import DistributionFamily
import numpy as np


def importance_sampling_given_sample(
                                            ω : Callable,
                                            h : Callable,
                                            X_from_sampler : List[float]
                                            ) -> float:
    f = lambda x : ω(x) * h(x)
    terms = np.array([ f(x_i) for x_i in X_from_sampler])
    return terms.mean()
    