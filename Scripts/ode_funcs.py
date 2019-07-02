import numpy as np
import scipy

def test_ode_func_1(y,t,p):

    #Scalar ODE, one parameter
    #Inspired by first order pharmacokineitc models
    #y' = exp(-t)-p*y
    return np.exp(-t) - p[0]*y[0]

def make_test_data_1():

    t = np.arange(0,8,0.5)
    p = np.array([0.4])
    y = scipy.integrate.odeint(func = test_ode_func_1, 
                                 y0 = 0,
                                 t = t, 
                                 args = tuple([p]))

    yobs = np.random.lognormal(mean = np.log(y), sigma = 0.1)

    data = {'t':t[1:], 'p':p, 'y': y[1:], 'yobs': yobs[1:], 'sigma': 0.1}

    return data


def test_ode_func_2(y,t,p):

    #Scalar ODE, two parameters
    #Inspired by first order pharmacokineitc models
    #y' = D/V*k_a*exp(-k_a*t)-k*y

    return p[0]*np.exp(-p[0]*t)-p[1]*y[0]


def make_test_data_2():

    t = np.arange(0, 8, 0.5)
    p = np.array([0.4, 1])
    y = scipy.integrate.odeint(func=test_ode_func_2,
                               y0=0,
                               t=t,
                               args=tuple([p]))

    yobs = np.random.lognormal(mean=np.log(y), sigma=0.1)

    data = {'t': t[1:], 'p': p, 'y': y[1:], 'yobs': yobs[1:], 'sigma': 0.1}

    return data


def test_ode_func_3(y,t,p):

    #Vector ODE, scalar parameter
    #non-dimensionalized SIR model
    #S' = -R_0*S*I, I' = R_0*S*I - I


    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - y[1]

    return [ds,di]


def make_test_data_3():

    t = np.linspace(0,8,51)
    p = np.array([4])
    y = scipy.integrate.odeint(func=test_ode_func_3,
                               y0=[0.99, 0.01],
                               t=t,
                               args=tuple([p]))

    S,I = y.T
    yobs = np.c_[np.random.lognormal(mean=np.log(S), sigma=0.1), np.random.lognormal(mean=np.log(I), sigma=0.25)]


    data = {'t': t, 'p': p, 'y': y, 'yobs': yobs, 'sigma': [0.1, 0.25]}

    return data

def test_ode_func_4(y,t,p):

    #Vector ODE, vector paramter
    #Inspired by SIR model
    #S' = -beta*S*I, I' = beta*S*I - gamma*I


    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - p[1]*y[1]

    return [ds,di]


def make_test_data_4():

    t = np.linspace(0, 8, 51)
    p = np.array([4,1])
    y = scipy.integrate.odeint(func=test_ode_func_4,
                               y0=[0.99, 0.01],
                               t=t,
                               args=tuple([p]))

    S, I = y.T
    yobs = np.c_[np.random.lognormal(mean=np.log(
        S), sigma=0.1), np.random.lognormal(mean=np.log(I), sigma=0.25)]

    data = {'t': t, 'p': p, 'y': y, 'yobs': yobs, 'sigma': [0.1, 0.25]}

    return data
