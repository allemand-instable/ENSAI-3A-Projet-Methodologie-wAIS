from other.norm import norm
import numpy as np
from time import perf_counter

def main() :
    vec = np.array([-2, 7])
    
    t1 = perf_counter()
    print(f"Numpy : {norm(vec, 'numpy')}")
    t2 = perf_counter()
    print(t2-t1)
    t1 = perf_counter()
    print(f"Numpy : {norm(vec, 'alpha-max-beta-min')}")
    t2 = perf_counter()
    print(t2-t1)
if __name__ == "__main__":
    main()