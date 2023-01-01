from wAIS.get_target_density import get_target_density
from distribution_family.normal_family import NormalFamily

def main():
    target = get_target_density(
        NormalFamily(0,1),
        lambda x : x**2,
        1.25
    )

    X = [-1, 0, 0.5]

    a=[target.density(x) for x in X]
    b=[target.density_fcn(x, target.parameters_list()) for x in X]

    print(a)
    print(b)
    
    # 0.072591217355743
    # 0.518624964521863
    # 0.369668593102515
    
    # output :
    # [0.06049268112978584, 0.49867785050179086, 0.3520653267642995]
    # [0.06049268112978584, 0.49867785050179086, 0.3520653267642995]