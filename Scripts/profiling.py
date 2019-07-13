from scipy.integrate import odeint
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pymc3 as pm
import pprint

from ode_api import *
from ode_funcs import *

THEANO_FLAGS = 'exception_verbosity=high'
THEANO_FLAGS='optimizer=fast_compile'
theano.config.exception_verbosity= 'high'
theano.config.floatX = 'float64'


def run():
    #Specify how many stats, initial values, and ode parameters there are
    n_states = 2
    n_odeparams = 2

    data = make_test_data_4()
    #Times to evaluate the solution
    times = data['t']

    #Instantiate the ODEModel
    ode_model = ODEModel(func = test_ode_func_4,
                        t0 = 0,
                        times = times,
                        n_states = n_states,
                        n_odeparams = n_odeparams)

    Ytrue = data['y']
    Y = data['yobs']

    my_ODEop = ODEop(ode_model)

    # The probabilistic model
    with pm.Model() as first_model:

        # Priors for unknown model parameters
        beta = pm.HalfNormal('beta',2)
        gamma = pm.HalfNormal('gamma',2)

        sigma = pm.HalfCauchy('sigma',1, shape = 2)

        # Forward model
        #[ODE Parameters, initial condition]
        all_params = pm.math.stack([beta, gamma,0.99, 0.01],axis=0)


        ode_sol = my_ODEop(all_params)

        forward = ode_sol.reshape(Y.shape)

        ode = pm.Deterministic('ode',ode_sol)

        # Likelihood
        Y_obs = pm.Lognormal('Y_obs', mu=pm.math.log(forward), sd=sigma, observed=Y)


        trace = pm.sample(50, tune=50, init='adapt_diag', target_accept = 0.99, chains=1)


if __name__ == '__main__':
    run()
