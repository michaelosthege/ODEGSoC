from scipy.integrate import odeint
import numpy as np
from ode_api import *
from ode_funcs import *
import matplotlib.pyplot as plt
import scipy
import pymc3 as pm

#odefunc, n_states, n_ivs, n_odeparams, y0 = None

n_states = 2
n_ivs = 2
n_odeparams = 1
model = ODEModel(odefunc = test_ode_func_3, 
                    n_states = n_states,
                    n_ivs = n_ivs,
                    n_odeparams = n_odeparams)


class solveCached(object):
    def __init__(self, times, n_params, n_outputs):
      
        self._times = times
        self._n_params = n_params
        self._n_outputs = n_outputs
        self._cachedParam = np.zeros(n_params)
        self._cachedSens = np.zeros((len(times), n_outputs, n_params))
        self._cachedState = np.zeros((len(times),n_outputs))
        
    def __call__(self, x):
        
        if np.all(x==self._cachedParam):
            state, sens = self._cachedState, self._cachedSens
            
        else:
            state, sens = model.simulate(x, times)
        
        return state, sens


times = np.arange(0,10) # number of measurement points (see below)   
cached_solver=solveCached(times, n_odeparams + n_ivs, n_states)

def state(x):
    State, Sens = cached_solver(np.array(x,dtype=np.float64))
    cached_solver._cachedState, cached_solver._cachedSens, cached_solver._cachedParam = State, Sens, x
    return State.reshape((2*len(State),))

def numpy_vsp(x, g):    
    numpy_sens = cached_solver(np.array(x,dtype=np.float64))[1].reshape((n_states*len(times),len(x)))
    return numpy_sens.T.dot(g)


sims, sens = model.simulate([8,0.99,0.01], times)
S = scipy.stats.lognorm.rvs(s = 0.1, scale = sims[:,0])
I = scipy.stats.lognorm.rvs(s = 0.1, scale = sims[:,1])

# Define the data matrix
Y = np.vstack((S,I)).T

# Now instantiate the theano custom ODE op
my_ODEop = ODEop(state,numpy_vsp)

# The probabilistic model
with pm.Model() as LV_model:

    # Priors for unknown model parameters
    beta = pm.Normal('beta',6,2)
    # S0 = pm.Deterministic('S0',0.99)
    # I0 = pm.Deterministic('I0',0.01)
    
    sigma = pm.HalfCauchy('sigma',1)

    # Forward model
    all_params = pm.math.stack([beta, 0.99,0.01],axis=0)
    ode_sol = my_ODEop(all_params)
    forward = ode_sol.reshape(Y.shape)

    # Likelihood 
    Y_obs = pm.Lognormal('Y_obs', mu=pm.math.log(forward), sd=sigma, observed=Y)
    
    trace = pm.sample(1500, tune=1000, init='adapt_diag')

pm.traceplot(trace)

plt.show()