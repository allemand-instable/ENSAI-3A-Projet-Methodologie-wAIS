import numpy as np
from distribution_family.distribution_family import DistributionFamily
from numpy.typing import NDArray
from typing import List

def compute_likelihood(      f_target : DistributionFamily, 
                             q_t : DistributionFamily, 
                             q_importance_sampling : DistributionFamily,
                             theta_t : NDArray,
                             X_sampled_from_uniform : List[float],
                            ) -> float:
    terms = np.array(
        [
            (f_target.density(x_i) / q_importance_sampling.density(x_i)) * np.log(q_t.density_fcn(x_i, theta_t)/f_target.density(x_i)) for x_i in X_sampled_from_uniform
        ]
        )
    return terms.mean()