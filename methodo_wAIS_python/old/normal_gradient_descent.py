import numpy as np
import numpy.random as nprd
import plotly.express as plx
import plotly.graph_objects as pgo

def sample(params : np.ndarray, n : int) -> np.ndarray:
    μ : float = params[0]
    # variance
    Σ : float = params[1]
    return nprd.normal(loc = μ, scale=np.sqrt(Σ), size= n)


def log_q_sur_f(x, sampler_params, true_params):
    μ = sampler_params[0]
    Σ = sampler_params[1]
    
    m = true_params[0]
    s = true_params[1]

    Z_μ = ((x-μ)**2)/(2*Σ)
    
    Z_m =((x-m)**2)/(2*s)

    return (np.log(s) - np.log(Σ))/2 + (Z_m - Z_μ)


def sample_function(fcn, params, N):
    return np.array(list(map(fcn, sample(params, N))))


def L(sampler_params : np.ndarray, true_params : np.ndarray, N : int) -> float:
    log_q_sur_f_given_params = lambda x : log_q_sur_f(x, sampler_params=sampler_params, true_params=true_params)
    
    list_log_q_sur_f = sample_function(log_q_sur_f_given_params, true_params, N)
    
    return 1/N * np.array( list_log_q_sur_f ).sum()
                          
                          
def grad_L_i(x, sampler_params : np.ndarray, true_params : np.ndarray) -> np.ndarray:
    μ = sampler_params[0]
    Σ = sampler_params[1]
    
    m = true_params[0]
    s = true_params[1]
    
    Σ_inverse = 1/Σ
    
    sqrt_Σ_sur_s = np.sqrt( Σ / s )
    
    centered_x_μ = x - μ
    centered_x_m = x-m
    
    Z_x_μ_Σ = centered_x_μ * Σ_inverse
    Z_x_m_s = centered_x_m * Σ_inverse
    
    exp_arg = ( Z_x_μ_Σ * centered_x_μ -  Z_x_m_s * centered_x_m )/2
    
    return sqrt_Σ_sur_s * np.exp(exp_arg) * np.array( [ Z_x_μ_Σ, (Z_x_μ_Σ**2)/2 + Σ_inverse ] )

def grad_L(sampler_params : np.ndarray, true_params : np.ndarray, N : int) -> float:
    
    
    grad_L_i_given_params = lambda x : grad_L_i(x, sampler_params, true_params)
    gradLi_list = sample_function(grad_L_i_given_params, sampler_params, N)
    
    return gradLi_list.sum()/N



true_params = np.array([1, 3])

μ_values = np.linspace(0, 2, 250)
Σ_values = np.linspace(1, 5, 200)
L_values = [[grad_L( np.array([x, y]) , true_params, 100) for y in Σ_values] for x in μ_values]
surface = pgo.Surface(x = μ_values, y = Σ_values, z = L_values)
fig = pgo.Figure(data = [surface])
scene = {
    "aspectratio": {"x": 3, "y": 2, "z": 1},
}
fig.update_layout(scene = scene,
                  xaxis_range =[-2, 2],
                  yaxis_range =[1, 5] )

import pprint


argmax_y_pour_x = [np.argmax([L_values[x][y] for y in range(len(Σ_values))]) for x in range(len(μ_values))]
L_max = [L_values[k][argmax_y_pour_x[k]] for k in range(len(μ_values))] 
argmax_x = np.argmax(L_max)
argmax = np.array([argmax_x, argmax_y_pour_x[argmax_x]])
print(argmax)



argmin_y_pour_x = [np.argmin([L_values[x][y] for y in range(len(Σ_values))]) for x in range(len(μ_values))]
L_max = [L_values[k][argmin_y_pour_x[k]] for k in range(len(μ_values))] 
argmin_x = np.argmin(L_max)
argmin = np.array([argmin_x, argmin_y_pour_x[argmin_x]])
print(argmin)

print((μ_values[argmin[0]], Σ_values[argmin[1]]))

#print( L_values[argmax_x], argmax_y_pour_x[argmax_x] )

#print(argmax_y_pour_x)

#print(len(L_max))

#print(L_max)

fig.show()