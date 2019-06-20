import numpy as np


def test_ode_func_1(y,t,p):

    #Scalar ODE, one parameter
    #Inspired by first order pharmacokineitc models
    #y' = exp(-t)-p*y
    return np.exp(-t) - p[0]*y[0]

def test_ode_func_2(y,t,p):

    #Scalar ODE, two parameters
    #Inspired by first order pharmacokineitc models
    #y' = D/V*k_a*exp(-k_a*t)-k*y

    return p[0]*np.exp(-p[0]*t)-p[1]*y[0]


def test_ode_func_3(y,t,p):

    #Vector ODE, scalar parameter
    #non-dimensionalized SIR model
    #S' = -R_0*S*I, I' = R_0*S*I - I


    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - y[1]

    return [ds,di]

def test_ode_func_4(y,t,p):

    #Vector ODE, vector paramter
    #Inspired by SIR model
    #S' = -beta*S*I, I' = beta*S*I - gamma*I


    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - p[1]*y[1]

    return [ds,di]
