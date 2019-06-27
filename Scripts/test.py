from scipy.integrate import odeint
import numpy as np
from ode_api import *
from ode_funcs import *
import matplotlib.pyplot as plt
import scipy
import pymc3 as pm

#edefunc, y0, t0, times, n_states, n_ics, n_odeparams

n_states = 1
n_ivs = 1
n_odeparams = 1
times = np.arange(0.5, 12, 0.5)


model = ODEModel(odefunc = test_ode_func_1, 
                    y0 = 0,
                    t0 = 0,
                    times = times,
                    n_states = n_states,
                    n_ivs = n_ivs,
                    n_odeparams = n_odeparams)




sims, sens = model.simulate([0.4,0])

Y = scipy.stats.lognorm.rvs(s = 0.1, scale = sims)

# Now instantiate the theano custom ODE op
my_ODEop = ODEop(model)

# The probabilistic model
with pm.Model() as LV_model:

    # Priors for unknown model parameters
    alpha = pm.Lognormal('alpha',1,1)
    
    sigma = pm.HalfCauchy('sigma',1)

    # Forward model
    all_params = pm.math.stack([alpha, 0],axis=0)
    ode_sol = my_ODEop(all_params)
    forward = ode_sol.reshape(Y.shape)

    # Likelihood 
    Y_obs = pm.Lognormal('Y_obs', mu=pm.math.log(forward), sd=sigma, observed=Y)
    
    trace = pm.sample(1500, tune=1000, init='adapt_diag')

pm.traceplot(trace)

plt.show()
