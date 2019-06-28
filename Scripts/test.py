from scipy.integrate import odeint
from scipy.stats import lognorm
import numpy as np
from ode_api import *
from ode_funcs import *
import matplotlib.pyplot as plt
import scipy
import pymc3 as pm

THEANO_FLAGS = 'exception_verbosity=high'


#Specify how many stats, initial values, and ode parameters there are
n_states = 1
n_ics = 1
n_odeparams = 1

#Times to evaluate the solution
times = np.arange(0.25,8,1)

#Instantiate the ODEModel
model_1 = ODEModel(odefunc = test_ode_func_1, 
                   y0 = 0,
                   t0 = 0,
                    times = times,
                    n_states = n_states,
                    n_ics = n_ics,
                    n_odeparams = n_odeparams)

#Simulate the data and create data to learn from
sims, sens = model_1.simulate([0.4,0])


Y = scipy.stats.lognorm.rvs(s = 0.1, scale = sims)

# Now instantiate the theano custom ODE op
my_ODEop = ODEop(model_1)


# The probabilistic model
with pm.Model() as first_model:

    # Priors for unknown model parameters
    alpha = pm.HalfNormal('alpha',1)
#     y0 = pm.HalfNormal('y0',1)
    sigma = pm.HalfCauchy('sigma',1)

    # Forward model
    all_params = pm.math.stack([alpha, 0],axis=0)
    pprint = tt.printing.Print('all_params')(all_params)
    
    ode_sol = my_ODEop(all_params)
    oprint = tt.printing.Print('ode_sol')(ode_sol)
    
    forward = ode_sol.reshape(Y.shape)
    fprint = tt.printing.Print('forward')(pm.math.log(forward))
    
    

    # Likelihood 
    Y_obs = pm.Lognormal('Y_obs', mu=fprint, sd=sigma, observed=Y)
#     Y_obs = pm.Normal('Y_obs', mu=forward, sd=sigma, observed=Y)
#     Y_obs = pm.Lognormal('Y_obs', mu=forward, sd=sigma, observed=Y)
    
    trace = pm.sample(10, tune=0, chains = 1, init='adapt_diag', target_accept = 0.99)
