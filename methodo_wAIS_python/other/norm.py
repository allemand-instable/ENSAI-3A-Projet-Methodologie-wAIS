#import string
#import numpy as np




# c'était juste pour tester l'algo alphamax beta min


# def norm(X : np.ndarray, method : string = "numpy"):
#     """Returns the norm of a vector X

#     Args:
#         X (np.ndarray): vector as a numpy array
#         method (string, optional): how the norm is calculated. Defaults to "numpy".
#             -> numpy : default numpy method
#             -> alpha-max-beta-min : approximation for 2D and 3D vectors
#             cf : https://www.wikiwand.com/en/Alpha_max_plus_beta_min_algorithm
#             and https://math.stackexchange.com/questions/1282435/alpha-max-plus-beta-min-algorithm-for-three-numbers

#     Raises:
#         Exception: _description_

#     Returns:
#         _type_: _description_
#     """
#     if method == "numpy" :
#         norm = np.linalg.norm(X)
#     elif method == "alpha-max-beta-min":
#         if X.size > 3 :
#             norm = np.linalg.norm(X)
#         elif X.size == 3 :
#             gamma   = 0.2987061876143797
#             beta    = 0.38928148272372454
#             alpha   = 0.9398086351723256
#             norm = 0
#         elif X.size == 2 :
#             alpha   = 0.960433870103
#             # 2cos(pi/8) / [ 1 + cos(pi/8) ]
#             beta    = 0.397824734759
#             # 2sin(pi/8) / [ 1 + cos(pi/8) ]
#             norm  = alpha * X.max() + beta * X.min()
#     else :
#         raise Exception("please choose a norm method")
#     return norm


