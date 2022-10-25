import numpy as np
from math_tools.gradient import gradient_selon


def test_fcn(u : np.ndarray,theta :np.ndarray,v:np.ndarray):
    return np.sum( [ u[k] * v[k] for k in range(v.size) ] ) + u[-1] - theta[0] + theta[1] 

def main():
      
    u = np.array([1,2,3,4])
    v = np.array([6,7,8])
    theta = np.array([10,20])  
    
    
    
    arguments = [u, theta, v]
    print(f"v = {v}")
    print(gradient_selon(1, test_fcn, *arguments))
    # should be [v1, v2, v3, 1]
    print("\n\n")
    print(theta)
    print(gradient_selon(2, test_fcn, *arguments))
    # should be [-1, 1]
    print("\n\n")
    print(f"u = {u}")
    print(gradient_selon(3, test_fcn, *arguments))
    # should be [u1, u2, u3]

if __name__ == "__main__":
    main()