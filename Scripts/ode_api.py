import numpy as np
import theano
import theano.tensor as tt
import scipy
THEANO_FLAGS='optimizer=fast_compile'


class ODEModel(object):

    def __init__(self, odefunc, n_states, n_ivs, n_odeparams, y0 = None):

        self.odefunc = odefunc
        self.n_states = n_states
        self.n_ivs = n_ivs
        self.n_odeparams = n_odeparams
        self.y0 = y0
        self.augmented_system = augment_system(self.odefunc)

    def system(self,Y,t,p):

        """
        This is the function that wull be passed to odeint.
        Solves both ODE and sensitivities

        Args:
            Y (vector): current state and current gradient state
            t (scalar): current time
            p (vector): parameters

        Returns:
            derivatives (vector): derivatives of state and gradient
        """


        dydt, sens = self.augmented_system(Y[:self.n_states], 
                                            t,
                                            p,
                                            Y[self.n_states:],
                                            self.n_states,
                                            self.n_odeparams + self.n_ivs
                                            )

        derivatives =  np.concatenate([dydt,sens])

        return derivatives

    def simulate(self, times, parameters):

        '''
        This function returns solutions and sensitivities of the ODE,
        evaluated at times and parameterized by parameters.

        Inputs:
            times(array): Times to evaluate the solution of the ODE
            parameters(array): Parameters for ODE.  Last entries should be
                               initial conditions

        Returns:
            sol(array): Solution of ODE
        '''


        sens_ic = np.zeros((self.n_states, self.n_odeparams + self.n_ivs))

        sens_ic[:, -self.n_states:] = np.eye(self.n_states)

        y_ic = parameters[-self.n_states:]

        y0 = np.concatenate([y_ic, sens_ic.ravel()])


        soln = scipy.integrate.odeint(self.system,
                    y0=y0,
                    t = times,
                    args = tuple([parameters]))

        y = soln[:, :self.n_states]
        sens = soln[:, self.n_states:].reshape((len(times),self.n_states, self.n_odeparams + self.n_ivs) )

        return [y,sens]




def augment_system(ode_func):


    '''Function to create augmented system.

    Take a function which specifies a set of differential equations and return
    a compiled function which allows for computation of gradients of the
    differential equation's solition with repsect to the parameters.

    Args:
        ode_func (function): Differential equation.  Returns array-like

    Returns:
        system (function): Augemted system of differential equations.

    '''

    #Shapes for the dydp dmatrix
    #TODO: Should this be int64 or other dtype?
    t_n = tt.scalar('n', dtype = 'int32')
    t_m = tt.scalar('m', dtype = 'int32')

    #Present state of the system
    t_y = tt.dvector('y')

    #Parameter(s).  Should be vector to allow for generaliztion to multiparameter
    #systems of ODEs
    t_p = tt.dvector('p')

    #Time.  Allow for non-automonous systems of ODEs to be analyzed
    t_t = tt.dscalar('t')

    #Present state of the gradients:
    #Will always be 0 unless the parameter is the inital condition
    #Entry i,j is partial of y[i] wrt to p[j]
    dydp_vec= tt.dvector('dydp')

    dydp = dydp_vec.reshape((t_n,t_m))

    #Stack the results of the ode_func
    #TODO: Does this behave the same of ODE is scalar?
    f_tensor = tt.stack(ode_func(t_y, t_t, t_p))

    #Now compute gradients
    J = tt.jacobian(f_tensor,t_y)

    Jdfdy = tt.dot(J, dydp)

    grad_f = tt.jacobian(f_tensor, t_p)

    #This is the time derivative of dydp
    ddt_dydp = (Jdfdy + grad_f).flatten()

    #TODO: Ensure system is reshaped on the odeint call?
    system = theano.function(
            inputs=[t_y, t_t, t_p, dydp_vec, t_n, t_m],
            outputs=[f_tensor, ddt_dydp],on_unused_input='ignore')

    return system
