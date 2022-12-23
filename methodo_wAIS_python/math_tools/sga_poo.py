from math_tools.gradient import gradient_selon
import numpy as np
from typing import Optional
import numpy.random as nprd
from math_tools.distribution_family import DistributionFamily
from numpy.typing import ArrayLike, NDArray
from utils.log import logstr
from logging import info, debug, warn, error
from utils.print_array_as_vector import get_vector_str
from copy import deepcopy
import time
from typing import Tuple
import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as plgo

class SGA_KullbackLeibler_Likelihood():
    def __init__(self, 
                 f : DistributionFamily ,
                 q : DistributionFamily , 
                 nb_drawn_samples : int, 
                 𝛾 : int, 
                 η_0 : float, 
                 θ_0 : Optional[ArrayLike] = None, 
                 ɛ : float = - np.inf, 
                 iter_limit : int = 100, 
                 η_update_method : str = "fixed",
                 benchmark : bool = False) -> None:
        """effectue une stochastic gradient ascent pour le problème d'optimisation de θ suivant le critère de la vraissemblance de Kullback-Leibler
            
        f           — target density
                        ➤ va être utilisée pour la comparaison avec q dans la maximisation de la vraissemblance de Kullback-Leibler
                        
                        L(θ) = - KL( f || q )
        
        q           — original sampling policy : q(𝑥, θ)
        
                                        parametric family of sampling policies / distributions
                                        given as a (lambda) function of 𝑥, θ ∈ 𝘟 × Θ

                                        q = lambda x,θ : np.exp( - (x-θ[0])**2 / (2*θ[1]) )/(np.sqrt(2*np.pi*θ[1]))
                                        gives a normal law density

                    ➤ va être modifiée à chaque itération
        
        
        N           — Nombre de samples tirés par la distribution q à chaque itération
        
        𝛾           — nombre d'observations à tirer aléatoirement
        
        η_0         — initialisation du pas
        
        θ_0         — initialisation des paramètres
        
        ɛ           — threshold pour la norme du gradient
        
        iter_limit  - nombre d'itérations max du gradient descent avant l'arrêt
        
        """

        # hyper paramètres
        
        self.nb_drawn_samples = nb_drawn_samples    # nb_drawn_samples
        self.𝛾 = 𝛾                                  # learning_rate
        self.η_0 = η_0                              # initialisation du pas
        self.ɛ = ɛ                                  # norm threshold
        self.iter_limit = iter_limit                #


        #   ———————————————     Éviter explosion du gradient        ————————————
        self.norm_safety_coefficient = np.Inf
        
        # update specifications
        self.η_update_method = η_update_method      # façon d'update le pas

        # paramètres
        self.f = f                                  # target density
        
        # gérer les formats de θ_0
        if θ_0 is None :
            # non spécifié → prendre les paramètres de q₀
            self.θ_0 = q.parameters_list()     # initialisation des params de q
        elif type(θ_0) in [list, NDArray] :
            # si c'est une liste ou une ndarray
            # on s'assure que c'est au format numpy
            self.θ_0 = np.array(θ_0)
        else :
            # sinon on connait pas
            raise TypeError(f"mauvais type de θ_0 : {type(θ_0)}\ntype attendu : [int, NDArray, ...]")
        
        
        
        # ————————————————————————————————————————————————————————————————————————————————— #
        #                                 WORKING VAR                                       #
        # ————————————————————————————————————————————————————————————————————————————————— #
                
        # initialisation des working vars
        self.θ_t : NDArray = np.array(deepcopy(self.θ_0))
        self.η_t = deepcopy(self.η_0)
        self.q = deepcopy(q)                                  # sampler density
        
        # on s'assure de rentrer dans la boucle pour la première itération
        self.norm_grad_L : float = self.ɛ + 1
        
        # utilitaire
        self.counter : int = 0
        
        
        self.benchmark = benchmark
        # ⚙️ BENCHMARK
        if benchmark is True :
            self.benchmark_graph : list[list[float]] = [ list([]) for k in range( len(self.θ_t) + 1)]
            #                                [ list[counter], list[d_μ], list[d_Σ] ]
        
    def update_η(self) -> None:
        """modifie self.η_t selon la méthode choisie avec η_update_method
        """
        match self.η_update_method :
            case "fixed":
                self.update_η_fixed()
            case "fixed_decreasing_rate":
                self.update_η_fixed_decreasing_rate()
            case "hessian":
                self.update_η_hessian()
            case _ :
                # par défaut pa fixed
                err_str = f"/!\ {_} inconnu, utilisation du pas fixe..."
                print(err_str)
                debug(logstr(err_str))
                self.update_η_fixed()
    
    def update_η_fixed(self) -> None:
        # do nothing
        pass
    def update_η_fixed_decreasing_rate(self) -> None:
        #TODO
        pass
    def update_η_Hessian(self) -> None:
        #TODO
        pass
    def update_η_hessian(self) -> None:
        #TODO
        pass
    
    def get_current_theta(self) -> NDArray:
        return self.θ_t
    
    def execute(self) -> None:
        debug(logstr("executing gradient ascent :"))
        t = time.process_time()
        for counter in range(self.iter_limit) :
            debug(logstr(f"itération n°{counter}"))
            # si on atteint la précision voulue en terme de norme : fin
            if self.norm_grad_L < self.ɛ:
                debug(logstr(
                    f"||∇L|| = {self.norm_grad_L} ➤ arrêt"
                ))
                break
            # sinon on continue
            else :
                debug(logstr(
                    f"||∇L|| = {self.norm_grad_L} ➤ nouvelle itération"
                ))
                self.nouvelle_iteration()
        Δt = time.process_time() - t
        debug(logstr(f"⏱️ —  {Δt}"))
        
        
        # si on fait un benchmark
        if self.benchmark is True :
            
            n = len(self.θ_t)
            print(f"{self.θ_t}\n{n}")
            
            # 
            fig = make_subplots(
                                rows= n//2 + n%2 , cols=2,
                                subplot_titles= [ r"$\text{erreur relative : }" + "\\left| \\frac{" "θ_" + f"{k}" + "- θ^*_"f"{k}" +"}" + "{θ^*" + f"_{k}" + "}"  +"\\right|" "$" for k in range(n)]
                                )
            
            for k in range( n ):
                fig.add_trace(plgo.Scatter(x=self.benchmark_graph[0] , y=self.benchmark_graph[1+k]), row = 1 + k//2 , col = 1 + k%2)

            fig.show()
    
    
    def nouvelle_iteration(self):
        
        
        t = time.process_time()
        # état des lieux dans le log
        debug(logstr(
            f"état initial de l'itération :\nθ_t = {self.θ_t}\nη_t = {self.η_t}"
            ))
        
        
        # new sample from qₜ
        X_t = self.q.sample( self.nb_drawn_samples )
        
        if X_t is None :
            X_t = []        # comme ça le hinter me laisse tranquille
            raise ValueError("⚠️ new sample from qₜ is empty")
        
        debug(logstr(f"\nX = {X_t}\n\nlen(X) = {len(X_t)}"))
        
        
        #                           tirage de nouvelles observations selon le sampler q
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        
        # si le nombre d'obs tirées pour l'estimation du gradient est inférieur au nombre d'observations tirées dans le sample de qₜ il s'agit d'un SGA
        if self.𝛾 < self.nb_drawn_samples :
            # Stochastic : choose random observations
            obs_tirées = nprd.choice(range(len(X_t)), self.𝛾, replace=False)
            
            # 💡 NOTE : on pourrait s'intéresser au papier 
            #           Adaptive Sampling for Incremental Optimization using Stochastic Gradient Descent (2015) — Papa
            #           sur la façon dont on tire des observations dans l'échantillon samplé
            X_sampled_from_uniform = [  X_t[i] for i in obs_tirées  ]
            #                                             b inclu
            debug(logstr(f"\nX_sampled_from_uniform = {X_sampled_from_uniform}"))
        
        # si c'est = on a un GA classique
        elif self.𝛾 == self.nb_drawn_samples :
            debug(logstr("classic (non stochastic) gradient ascent"))
            X_sampled_from_uniform = X_t
        
        else :
            raise Exception(f"can't choose more values than generated : {self.𝛾} = 𝛾 > nb_drawn_samples = {self.nb_drawn_samples}")
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        
        
        
        
        #                   on update la valeur du gradient de L selon la méthode de la SGD
        # ————————————————————————————————————————————————————————————————————————————————————————————————————

        grad_L_list = self.grad_L_list(X_sampled_from_uniform, self.θ_t, known_len_X=self.𝛾)
        
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        #                             calcul de ∇L ⋍ 1/𝛾 Σ ∇L(X_i | θ)
        #                                             i ∈ {obs tirées}
        # 
        #                               rappel : card {obs tirées} = 𝛾
        #  ————————————————————————————————————————————————————————————————————————————————————————————————————
        
        
        grad_L_estimator : NDArray = np.add.reduce( grad_L_list )/self.𝛾
        
        debug(logstr(f"calcul de ∇L ⋍ 1/𝛾 Σ ∇L(X_i | θ)\n\ngrad_L_estimator = {grad_L_estimator}"))
        
        
        
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        #                                       update des (hyper) params
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        
        # paramètre de distributoin q
        self.θ_t = self.θ_t + self.η_t * grad_L_estimator
        
        
        #?  ———————————        DEBUG        —————————————————
        str_theta = f"θ_{self.counter} = {self.θ_t}"
        #! TOGGLE ON/OFF if needed or not
        print(str_theta)    #!          |
        #! —————————————————————————————
        #?  —————————————————————————————————————————————————
        
        
                
        # sampling policy
        self.q.update_parameters(self.θ_t)
        
        # pas
        self.update_η()
        debug(logstr(f"{str_theta}\nη_t+1 = {self.η_t}"))
        
        # ————————————————————————————————————————————————————————————————————————————————————————————————————
        
        self.counter += 1
        self.norm_grad_L = np.linalg.norm(self.η_t * grad_L_estimator)
        
        Δt = time.process_time() - t
        debug(logstr(f"⏱️ —  {Δt}"))
        
        
        if self.benchmark is True :
            target = self.f.parameters_list()
            self.benchmark_graph[0].append(self.counter)
            for k in range(len(self.θ_t)) :
                d_k = np.abs((self.θ_t[k] - target[k])/(target[k] + 1e-4))
                self.benchmark_graph[1+k].append(d_k)
            
        
        # on retourne aussi le dernier sample pour être efficace en calcul pour l'algo qui va être utilisé
        return X_t
    
    
    
    def ω(self, x : float,θ : NDArray) -> float: 
        
        # indépendant de θ
        f_val = self.f.density(x)
        # dépendant de θ
        q_val = self.q.density_fcn(x, θ)
                
        res = f_val/q_val
        
        debug(logstr(f"θ = {θ}\nf(x) = {f_val}\nq(x, theta) = {q_val}\n⇒ ω(x,θ) = {res}"))
        return res
    
    
    
    
    def ω_norm_list(self, X : list,θ : NDArray, normalize : bool = True) -> NDArray: 
        
        t = time.process_time()
        
        ω_list = np.array([ self.ω(x, θ) for x in X ])
        if normalize is True :
            ω_list = 1/(ω_list.sum()) * ω_list
        
        Δt = time.process_time() - t
        debug(logstr(f"⏱️ —  {Δt}"))
        
        return ω_list
    
    
    
    
    def h(self,x : float,θ : NDArray) -> NDArray:
        log_q = lambda u, v : np.log(self.q.density_fcn(u, v))
        res = gradient_selon(2, log_q , *[x, θ] )
        debug(logstr(f"h(x,θ) = {get_vector_str(res)}"))
        return res
    
    
    
    
    def grad_L_list(self, X : list[float], θ : NDArray, known_len_X : int | None = None) -> NDArray:
        
        t = time.process_time()
        
        if known_len_X is None :
            n = len(X)
        else :
            n = known_len_X
        
        ω_list = self.ω_norm_list(X, θ, True)
        #                   on renormalise les poids après les avoir calculés
        #                                           |
        #                                           ↓
        #gradL_list = np.array([self.h(X[i], θ) * ω_list[i] for i in range(len(X))])
        gradL_list = np.array([self.grad_L(X[i], self.θ_t, self.norm_safety_coefficient, ω_list[i]) for i in range(n)])
        
        Δt = time.process_time() - t
        debug(logstr(f"calcul de ∇L_list : {gradL_list[0:2]} ... {gradL_list[-1]}"))
        debug(logstr(f"⏱️ —  {Δt}"))
        
        return gradL_list        

    
    def grad_L(self, x_i : float, θ : NDArray, norm_safety_coefficient : float, ω_i : float) -> NDArray:
        
        t = time.process_time()
                
        #                   importance sampling selon q(θ)
        #                                  |
        #                                  ↓
        #? changement
        #res = self.h(x_i, θ) * self.ω(x_i, θ)
        res = self.h(x_i, θ) * ω_i
        
        debug(logstr(f"∇L_i(θ) = \n{get_vector_str(res)}"))
        
        norm_res = np.linalg.norm(res)
        norm_theta = np.linalg.norm(np.array(θ))
        
        # avec les ω, si on a un ω ~ 10 000 lorsque q << f 
        # on va avoir la norme de la direction qui explose
        # on essaye d'éviter cela
        
        
        # si le gradient est aberrant en i selon le coefficient de sécurité
        # on ne le compte pas
        if norm_res > norm_safety_coefficient * norm_theta :
            debug(logstr(f"{norm_res} = || res || > {norm_safety_coefficient} x || θ || = {norm_safety_coefficient*norm_theta}\n\nreturning zeros..."))
            return np.zeros(θ.shape)
        
        Δt = time.process_time() - t
        debug(logstr(f"⏱️ —  {Δt}"))
        
        return res
