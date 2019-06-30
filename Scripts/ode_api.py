import numpy as np
import theano
import theano.tensor as tt
import scipy
THEANO_FLAGS='optimizer=fast_compile'


class ODEModel(object):


    def __init__(self, func, t0, times, n_states, n_odeparams):
        
        self.func = func
        self.t0 = t0
        self.times = times
        self.n_states = n_states
        self.n_odeparams = n_odeparams

        self.n = n_states
        self.m = n_odeparams + n_states

        self.augmented_times = np.insert(times, t0, 0)
        self.augmented_func = augment_system(func)
        self.sens_ic = self.make_sens_ic()

        self.cached_y = None
        self.cached_sens = None
        self.cached_parameters = None

    def make_sens_ic(self):

        #The sensitivity matrix will always have consistent form.
        #If the first n_odeparams entries of the parameters vector in the simulate call
        #correspond to ode paramaters, then the first n_odeparams columns in the sensitivity matrix will be 0
        sens_matrix = np.zeros((self.n, self.m))

        #If the last n_states entrues of the paramters vector in the simulate call
        #correspond to initial conditions of the system,
        #then the last n_states columns of the sensitivity matrix should form an identity matrix
        sens_matrix[:,-self.n_states] = np.eye(self.n_states)

        #We need the sensitivity matrix to be a vector (see augmented_function)
        #Ravel and return
        dydp = sens_matrix.ravel()
        return dydp

    def make_state_ic(self, parameters):

        #Convience function
        return parameters[self.n_odeparams:]

    def system(self, Y,t,p):
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

        dydt, ddt_dydp = self.augmented_func(Y[:self.n], t, p, Y[self.n:], self.n, self.m)
        derivatives = np.concatenate([dydt,ddt_dydp])
        return derivatives


    def simulate(self, parameters):

        #Initial condition comprised of state initial conditions and raveled sensitivity matrix
        y0 = np.concatenate([self.make_state_ic(parameters),self.sens_ic])

        #perform the integration
        sol= scipy.integrate.odeint(func = self.system,
                                    y0 = y0,
                                    t = self.augmented_times,
                                    args = tuple([parameters]))
        #The solution
        y = sol[1:,:self.n_states]
        
        #The sensitivities, reshaped to be a sequence of matrices
        sens = sol[1:, self.n_states:].reshape(len(self.times), self.n, self.m)

        return y, sens

    def cached_simulate(self, parameters):

        if np.all(np.array(parameters) == self.cached_parameters):
            y = self.cached_y
            sens = self.cached_sens
        else:
            y,sens = self.simulate(np.array(parameters))

        return y,sens

    def state(self,x):
        y, sens = self.cached_simulate(np.array(x,dtype=np.float64))
        self.cached_y, self.cached_sens, self.cached_parameters = y, sens, x
        return y.ravel()

    def numpy_vsp(self,x, g):    
        numpy_sens = self.cached_simulate(np.array(x,dtype=np.float64))[1].reshape((self.n_states*len(self.times),len(x)))
        return numpy_sens.T.dot(g)


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

    def __init__(self, ode_model):
        self._state = ode_model.state
        self._numpy_vsp = ode_model.numpy_vsp

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
    t_n = tt.scalar('n', dtype = 'int64')
    t_m = tt.scalar('m', dtype = 'int64')

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

