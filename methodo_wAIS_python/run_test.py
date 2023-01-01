import test.SGA_seq as SGAseq
import test.wAIS as wAIS
import test.target as target

def main():
    # SGAseq.generalized_code()
    # integral_test_polynomial()
    # target.main()
    wAIS.main()
    
def integral_test_polynomial():
    # wAIS.main()
    SGAseq.renyi_vs_kullback_knwon_var()
    SGAseq.renyi_vs_kullback_unknwon_var()
    
def normal_test():
    params = SGAseq.mean_seq(3,13, 500, 0.2)
    params = SGAseq.mean_seq(9,20, 500, 0.2)
    params = SGAseq.mean_seq(20,10, 500, 0.5)

def normal_test_known_variance():
    SGAseq.known_variance(3,13, 500, 0.2)
    SGAseq.known_variance(9,20, 500, 0.2)
    SGAseq.known_variance(20,10, 500, 0.5)
    
def other_distrib_test():
    SGAseq.other_distrib()
    
