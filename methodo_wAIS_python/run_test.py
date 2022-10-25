import test.test_gradient
import calcul_approx.importance_sampling
import proba.sampling_policy

#test.test_gradient.main()
#calcul_approx.importance_sampling.main()

q = proba.sampling_policy.GaussianSamplingPolicy(0, 1)

q.show_graph(-5, 5)

