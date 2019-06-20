from scipy.integrate import odeint
import numpy as np
from ode_api import *
from ode_funcs import *
import matplotlib.pyplot as plt


n_states = 2
n_params = 2
y0 = np.array([0.99,.01])
p0 = np.zeros(n_states*n_params)

#odefunc, n_states, n_ivs, n_odeparams, y0 = None
model = ODEModel(test_ode_func_4, 
                2,
                2,
                2,
                y0 = np.concatenate([y0,p0]))


times = np.linspace(0,5,1001)
parameters = [8,1, 0.99, 0.01]

y,sens = model.simulate(times, parameters)

print(sens[0])
