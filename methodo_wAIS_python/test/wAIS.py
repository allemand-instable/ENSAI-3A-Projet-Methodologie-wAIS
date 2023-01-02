from wAIS.wAIS import weighted_adaptive_importance_sampling, default_params_KL, default_params_R
from distribution_family.normal_family import NormalFamily
from distribution_family.uniform_family import UniformFamily
import numpy as np
import kullback_leibler.L_gradient.grad_importance_sampling as kl_grad_is
import renyi_alpha_divergence.renyi_importance_sampling_gradient_estimator as r_grad_is
import plotly.express as plx
import plotly.graph_objects as plgo
import pandas as pd
from numpy import mean, median, std, var

q_init = NormalFamily(5, 2)

sqrt_2_pi = np.sqrt(2*np.pi)

def int_P(A, B, a, b, c):
    return (a/4)*(B**4 - A**4) + (b/3)*(B**3 - A**3) + (c/2)*(B**2-A**2)

def P(x, a, b, c):
    return a*(x**3) + b*(x**2) + c*x


from distribution_family.uniform_family import UniformFamily

from numpy.random import randint
def produire_dict(method, one, two, three, four, five, six):
    return {
        # methode - T - n - freq - nb_pas 
        f"{method}-5-20 000-1-5"   : one,
        f"{method}-5-20 000-2-10"  : two ,
        f"{method}-20-5 000-4-6"   : three,
        f"{method}-20-5 000-2-3"   : four,
        f"{method}-50-2 000-4-6"   : five,
        f"{method}-50-2 000-2-3"   : six
    }


def kl_wais (T, n, freq, nb_pas, nb_estim, pi)  -> list[float]: 
    KL_updated_dict = default_params_KL.copy()
    u = dict(frequency=freq, gradient_descent__iter_limit=nb_pas, gradient_descent__nb_stochastic_choice=50)
    KL_updated_dict.update(u)
    print(KL_updated_dict)
    return [weighted_adaptive_importance_sampling(lambda x : x, pi, NormalFamily(-3,1), KL_updated_dict, lambda x : int(n), T=T) for j in range(nb_estim)]

def R_wais (T, n, freq, nb_pas, nb_estim, pi)  -> list[float]: 
    R_updated_dict = default_params_R.copy()
    u = dict(frequency=freq, gradient_descent__iter_limit=nb_pas, gradient_descent__nb_stochastic_choice=50)
    R_updated_dict.update(u)
    print(R_updated_dict)
    return [weighted_adaptive_importance_sampling(lambda x : x, pi, NormalFamily(-3,1), R_updated_dict, lambda x : n, T=T) for j in range(nb_estim)]

from benchmark.mse import mse

def main():
    mu = -7
    pi = NormalFamily(mu,1)

    int1 = []
    int2 = []    
    
    the_nombre_magic = 30
    
    KL_dict = produire_dict("KL", 
                  kl_wais(5, 20000,1,5,the_nombre_magic,pi),
                  kl_wais(5, 20000,2,10,the_nombre_magic,pi),
                  kl_wais(20, 5000,4,6,the_nombre_magic,pi),
                  kl_wais(20, 5000,2,3,the_nombre_magic,pi),
                  kl_wais(50, 2000,4,6,the_nombre_magic,pi),
                  kl_wais(50, 2000,2,3,the_nombre_magic,pi),
                  )
    
    R_dict = produire_dict("R", 
                  R_wais(5, 20000,1,5,the_nombre_magic,pi),
                  R_wais(5, 20000,2,10,the_nombre_magic,pi),
                  R_wais(20, 5000,4,6,the_nombre_magic,pi),
                  R_wais(20, 5000,2,3,the_nombre_magic,pi),
                  R_wais(50, 2000,4,6,the_nombre_magic,pi),
                  R_wais(50, 2000,2,3,the_nombre_magic,pi),
                  )
    
    mse_dict = {}
    for key, predict_list in KL_dict.items() :
        mse_dict.update( 
                        {key : mse([mu for k in range(len(predict_list))], predict_list)}
                        )
    for key, predict_list in R_dict.items() :
        mse_dict.update( 
                        {key : mse([mu for k in range(len(predict_list))], predict_list)}
                        )
    print(mse_dict)
    mse_df = pd.DataFrame.from_dict(mse_dict, orient='index', columns=["MSE"])
    mse_df.to_csv("./results/mse_results_wais.csv")
    mse_df.to_clipboard()
    
    
    def get_box_plot_KL():
        le_n = len(KL_dict["KL-5-20 000-1-5"])
        
        integral_df = pd.DataFrame(
            {
                'T' : ["5" for j in range(le_n)] + ["5" for j in range(le_n)] + ["20" for j in range(le_n)] +  ["20" for j in range(le_n)] + ["50" for j in range(le_n)] +  ["50" for j in range(le_n)],
                'predicted_mu' : KL_dict["KL-5-20 000-1-5"] + KL_dict["KL-5-20 000-2-10"] + KL_dict["KL-20-5 000-4-6"] + KL_dict["KL-20-5 000-2-3"] + KL_dict["KL-50-2 000-4-6"] + KL_dict["KL-50-2 000-2-3"],
                'frequency' : ["1" for j in range(le_n)] + ["1/2" for j in range(le_n)] + ["1/4" for j in range(le_n)] + ["1/2" for j in range(le_n)] + ["1/4" for j in range(le_n)] + ["1/2" for j in range(le_n)],
            }
        )
        integral_df.to_csv("./results/integral_df_KL.csv")
        fig = plx.box( integral_df,
                    x = "T",
                    y = "predicted_mu",
                    color="frequency",
                    points="all",
                    title="wAIS : Kullback-Leibler"
                    )
        fig.show()
    def get_box_plot_R():
        le_n = len(R_dict["R-5-20 000-1-5"])
        
        integral_df = pd.DataFrame(
            {
                'T' :  ["5" for j in range(le_n)] + ["5" for j in range(le_n)] + ["20" for j in range(le_n)] +  ["20" for j in range(le_n)] + ["50" for j in range(le_n)] +  ["50" for j in range(le_n)],
                'predicted_mu' : R_dict["R-5-20 000-1-5"] + R_dict["R-5-20 000-2-10"] + R_dict["R-20-5 000-4-6"] + R_dict["R-20-5 000-2-3"] + R_dict["R-50-2 000-4-6"] + R_dict["R-50-2 000-2-3"],
                'frequency' : ["1" for j in range(le_n)] + ["1/2" for j in range(le_n)] + ["1/4" for j in range(le_n)] + ["1/2" for j in range(le_n)] + ["1/4" for j in range(le_n)] + ["1/2" for j in range(le_n)],
            }
        )
        integral_df.to_csv("./results/integral_df_R.csv")
        fig = plx.box( integral_df,
                    x = "T",
                    y = "predicted_mu",
                    color="frequency",
                    points="all",
                    title="wAIS : Renyi"
                    )
        fig.show()
    
    print(mse_df)
    
    get_box_plot_KL()
    get_box_plot_R()
    
        
    # print(int1)
    # print(int2)
    
    # print("moyennes :")
    # print(np.array(int1).mean())
    # print(np.array(int2).mean())
    
    # print("std :")
    # print(np.array(int1).std())
    # print(np.array(int2).std())
    
    
    
def graph():
    T_list = [5,5,20,20,50,50]
    n_list = [2e4, 2e4, 5e3, 5e3, 2e3, 2e3]
    freq_list = [1,2,4,2,4,2]
    nb_pas_list = [5,10,6,3,6,3]
    mse_list = [[np.random.random() for l in range(6)] for k in range(6)]
    fig = plgo.Figure()
    surf=plgo.Scatter(x=freq_list , y=mse_list)
    # marker_color=color_dict[key]
    # surf = plgo.Surface(x=freq_list, y=nb_pas_list, z=mse_list, surfacecolor=T_list)
    fig.add_trace(surf)
    fig.update_xaxes(title_text='T')
    fig.update_yaxes(title_text='MSE')
    fig.update_layout(title=f"estimation de la moyenne d'une loi normale par wAIS (μ = {2})")
    fig.show()
    
def lol():
    def get_box_plot_KL():
        integral_df = pd.read_csv("./results/integral_df_KL.csv")
        fig = plx.box( integral_df,
                    x = "T",
                    y = "predicted_mu",
                    color="frequency",
                    points="all",
                    title="wAIS : Kullback-Leibler"
                    ).add_hline(-7, line_dash="dash", line_color="magenta", line_width=1,annotation_text="true μ = ∫xf(x)dx", 
              annotation_position="bottom right")
        fig.show()
    def get_box_plot_R():

        integral_df = pd.read_csv("./results/integral_df_R.csv")
        fig = plx.box( integral_df,
                    x = "T",
                    y = "predicted_mu",
                    color="frequency",
                    points="all",
                    title="wAIS : Renyi"
                    ).add_hline(-7, line_dash="dash", line_color="magenta", line_width=1, annotation_text="true μ= ∫xf(x)dx", 
              annotation_position="bottom right")
        fig.show()
    get_box_plot_KL()
    get_box_plot_R()