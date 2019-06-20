import numpy as np
import theano
import theano.tensor as tt
import scipy
THEANO_FLAGS='optimizer=fast_compile'


class ODEModel(object):

    def __init__(self, odefunc, n_states, n_ivs, n_odeparams):

        self.odefunc = odefunc
        self.n_states = n_states
        self.n_ivs = n_ivs
        self.n_odeparams = n_odeparams
        self.augmented_system = _augment_system(self.odefunc)

        self.n = self.n_states
        self.m = self.n_odeparams + self.n_ivs

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


        dydt, sens = self.augmented_system(Y[:self.n], 
                                            t,
                                            p,
                                            Y[self.n:],
                                            self.n,
                                            self.m
                                            )

        derivatives =  np.concatenate([dydt,sens])

        return derivatives

    def simulate(self, parameters, times):

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

        #Set up the inital condition for the sensitivities
        sens_ic = np.zeros((self.n, self.m))

        #Last n columns correspond to senstivity for inital condition
        #So last n columns form identity matrix
        sens_ic[:, -self.n:] = np.eye(self.n)

        #Create an initial condition for the system
        y_ic = parameters[-self.n:]

        #Concatenate the two inital conditions to form the initial condition
        #for the augmented system (ODE + sensitivity ODe)
        y0 = np.concatenate([y_ic, sens_ic.ravel()])

        #Integrate
        soln = scipy.integrate.odeint(self.system,
                    y0=y0,
                    t = times,
                    args = tuple([parameters]))

        #Reshaoe the sensitivities so that there is an nxm matrix for each 
        #timestep
        y = soln[:, :self.n]
        sens = soln[:, self.n:].reshape((len(times),self.n, self.m) )

        return [y,sens]

class ODEGradop(theano.Op):
    def __init__(self, numpy_vsp):
        self._numpy_vsp = numpy_vsp

    def make_node(self, x, g):
        x = theano.tensor.as_tensor_variable(x)
        g = theano.tensor.as_tensor_variable(g)
        node = theano.Apply(self, [x, g], [g.type()])
        return node

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]

        g = inputs_storage[1]
        out = output_storage[0]
        out[0] = self._numpy_vsp(x, g)       # get the numerical VSP
        
class ODEop(theano.Op):

    def __init__(self, state, numpy_vsp):
        self._state = state
        self._numpy_vsp = numpy_vsp

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)

        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]
        out = output_storage[0]
        
        out[0] = self._state(x)               # get the numerical solution of ODE states

    def grad(self, inputs, output_grads):
        x = inputs[0]
        g = output_grads[0]

        grad_op = ODEGradop(self._numpy_vsp)  # pass the VSP when asked for gradient 
        grad_op_apply = grad_op(x, g)
        
        return [grad_op_apply]





def _augment_system(ode_func):
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


    system = theano.function(
            inputs=[t_y, t_t, t_p, dydp_vec, t_n, t_m],
            outputs=[f_tensor, ddt_dydp],
            on_unused_input='ignore')

    return system
