import numpy as np
import pandas as pd

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random
from jax.experimental.ode import odeint

# import MCMC library
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

from tqdm import tqdm

### Basic usage ###

'''
# import ODE
from autode.autode import ODE

# instantiate ODE fit
model = ODE(system, df, params)

# fit to data
params = model.fit()

where df has columns [Time, Treatments, S1, ..., SN]
'''

# define function that returns model sensitivity vector
def runODE(t_eval, x, params, ctrl_params, dX_dt):
    # solve ODE model
    y = odeint(dX_dt, x, t_eval, params, ctrl_params)
    return y

# function to compute ODE gradients
def dZdt(system, Z, t, x, params, ctrl_params):

    # compute Jacobian (gradient of model w.r.t. x)
    Jx = jacfwd(system, 1)(t, x, params, ctrl_params)

    # compute gradient of model w.r.t. parameters
    Jp = jacfwd(system, 2)(t, x, params, ctrl_params)

    return Jx@Z + Jp

# define function that returns model sensitivity vector
def runODEZ(t_eval, x, params, ctrl_params, dXZ_dt):
    # check dimensions
    dim_x = len(x)
    n_params = len(params)
    dim_z = dim_x*n_params

    # set initial condition to z equal to zeros
    xz = jnp.concatenate((x, np.zeros(dim_z)))

    # solve ODE model
    y = odeint(dXZ_dt, xz, t_eval, params, ctrl_params)

    return y

### Function to process dataframes ###

def process_df(df, sys_vars, measured_vars, controls):
    # store treatment names
    all_treatments = df.Treatments.values
    unique_treatments = np.unique(all_treatments)

    # store measured datasets for quick access
    data = []
    for i,treatment in enumerate(unique_treatments):

        # pull community trajectory
        comm_inds = np.in1d(df['Treatments'].values, treatment)
        comm_data = df.iloc[comm_inds].copy()

        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull initial condition
        Y_init = np.array(comm_data[sys_vars].values[0], float)
        
        # pull system data
        Y_measured = np.array(comm_data[measured_vars].values, float)

        # pull control params
        ctrl_params = np.array(comm_data[controls].values, float)

        # append t_eval and Y_measured to data list
        data.append([t_eval, Y_init, Y_measured, ctrl_params])
    return data

class ODE:
    def __init__(self, system, dataframes, compressors, params, sys_vars, measured_vars, controls = [],
                 alpha_0=1., prior=None, verbose=True):
        '''
        system: a system of differential equations

        dfs: dataframes each with columns
        [Treatment], [Time], [x_1], ..., [x_n], [control_1], ..., [control_m]

        sys_vars: List of variable names of all model outputs as they appear in
                  dataframe (df). (Includes measured and unobserved outputs)

        params: initial guess of model parameters

        measured_sys_vars: List of observed (measured) model outputs

        control_param

        '''

        # make sure params are 1-dimensional
        self.params = np.array(params).ravel()
        if prior is not None:
            self.prior = np.array(prior).ravel()
        else:
            self.prior = np.zeros_like(params)

        # initial degree of regularization
        self.alpha_0 = alpha_0

        # number of parameters
        self.n_params = len(params)

        # dimension of model output
        self.sys_vars = sys_vars
        self.measured_vars = measured_vars
        self.n_sys_vars = len(sys_vars)

        # control input
        self.controls = controls
        self.n_ctrls = len(controls)

        # set compressors 
        self.compressors = []
        for compressor in compressors:
            # vmap over all time points 
            self.compressors.append(jit(vmap(compressor)))
        
        # store derivative of compressors 
        self.compressor_primes = []
        for compressor in compressors:
            # vmap over all time points 
            self.compressor_primes.append(jit(vmap(jacfwd(compressor))))
            
        # set up data
        self.datasets = []
        for i,(df, measured_var) in enumerate(zip(dataframes, measured_vars)):
            self.datasets.append(process_df(df, sys_vars, measured_var, controls))
            
        # for additional output messages
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-4

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        # jit compile differential equation
        def dX_dt(x, t, params, ctrl_params):
            # concatentate x and z
            return system(t, x, params, ctrl_params)
        self.dX_dt = jit(dX_dt)

        # if not vectorized, xz will be 1-D
        dim_z = len(sys_vars)*len(params)
        def dXZ_dt(xz, t, params, ctrl_params):
            # split up x and z
            x = xz[:len(sys_vars)]
            Z = jnp.reshape(xz[len(sys_vars):], [len(sys_vars), len(params)])
            dzdt = jnp.reshape(dZdt(system, Z, t, x, params, ctrl_params), dim_z)

            # concatentate x and z
            dXZdt = jnp.concatenate([system(t, x, params, ctrl_params), dzdt])
            return dXZdt
        self.dXZ_dt = jit(dXZ_dt)

        # jit compile function to integrate ODE
        self.runODE  = jit(lambda t_eval, x, params, ctrl_params: runODE(t_eval, x, params, ctrl_params, self.dX_dt))
        self.runODEZ = jit(lambda t_eval, x, params, ctrl_params: runODEZ(t_eval, x, params, ctrl_params, self.dXZ_dt))

        # jit compile matrix operations
        def SSE_next(Y_error, G, Ainv):
            return jnp.sum(Y_error**2) + jnp.einsum('tki,ij,tlj->', G, Ainv, G)
        self.SSE_next = jit(SSE_next)

        def yCOV_next(Y_error, G, Ainv):
            return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.einsum('tki,ij,tlj->kl', G, Ainv, G)
        self.yCOV_next = jit(yCOV_next)

        def A_next(G, Beta):
            return jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
        self.A_next = jit(A_next)

        def GAinvG(G, Ainv):
            return jnp.einsum('tki,ij,tlj->tkl', G, Ainv, G)
        self.GAinvG = jit(GAinvG)

        def NewtonStep(A, g):
            return jnp.linalg.solve(A,g)
        self.NewtonStep = jit(NewtonStep)

        def eval_grad_NLP(Y_error, Beta, G):
            return jnp.einsum('tk,kl,tli->i', Y_error, Beta, G)
        self.eval_grad_NLP = jit(eval_grad_NLP)

    def fit(self, evidence_tol=1e-3, beta_tol=1e-3):
        # estimate parameters using gradient descent
        convergence = np.inf
        previdence  = -np.inf

        while convergence > evidence_tol:
            # update Alpha and Beta hyper-parameters
            self.update_precision()
            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective, x0=self.params,
                       jac=True, hess=self.hessian, tol=beta_tol,
                       method='Newton-CG',
                       callback=self.callback)
            if self.verbose:
                print(self.res)
            self.params = self.res.x
            # update covariance
            self.update_covariance()
            # update evidence
            self.update_evidence()
            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1.,np.abs(self.evidence)])
            # update evidence
            previdence = np.copy(self.evidence)

    # EM algorithm to update hyper-parameters
    def update_precision(self):
        print("Updating precision...")

        # initialize precision parameters
        self.N = []
        self.beta = []
        self.Beta = []

        # loop over datasets
        for i,dataset in enumerate(self.datasets):
            # loop over each sample in dataset
            SSE = 0.
            yCOV = 0.
            N = 0
            for t_eval, Y_init, Y_compressed, ctrl_params in dataset:
                
                # count number of observations
                N += len(t_eval[1:]) * np.sum(np.sum(Y_compressed, 0) > 0) / self.n_sys_vars

                # run model using current parameters
                if self.A is None:
                    # run ODE on initial condition 
                    output = self.runODE(t_eval, Y_init, self.params, ctrl_params)

                    # Determine SSE
                    Y_error = self.compressors[i](output) - Y_compressed
                    SSE  += np.sum(Y_error[1:]**2)
                    yCOV += np.einsum('tk,tl->kl', Y_error[1:], Y_error[1:])
                else:
                    # run model using current parameters, output = [n_time, n_sys_vars]
                    output = self.runODEZ(t_eval, Y_init, self.params, ctrl_params)
                    Y_predicted = output[1:, :self.n_sys_vars]

                    # collect gradients and reshape
                    G = np.reshape(output[1:, self.n_sys_vars:],
                                  [output[1:].shape[0], self.n_sys_vars, self.n_params])
                    
                    # compress model output and gradient 
                    G = np.einsum('tck,tki->tci', self.compressor_primes[i](Y_predicted), G) 
                    Y_predicted = self.compressors[i](Y_predicted)
                    
                    # Determine SSE
                    Y_error = Y_predicted - Y_compressed[1:]
                    SSE  += self.SSE_next(Y_error, G, self.Ainv)
                    yCOV += self.yCOV_next(Y_error, G, self.Ainv)

            ### M step: update hyper-parameters ###
            self.N.append(N)
            if self.A is None:
                # target precision
                self.beta.append(N*Y_compressed.shape[-1]/(SSE + 2.*self.b))
                self.Beta.append(N*np.linalg.inv(yCOV + 2.*self.b*np.eye(Y_compressed.shape[-1])))
            else:
                # maximize complete data log-likelihood w.r.t. beta
                self.beta.append(N*Y_compressed.shape[-1]/(SSE + 2.*self.b))
                Beta = N*np.linalg.inv(yCOV + 2.*self.b*np.eye(Y_compressed.shape[-1]))
                Beta = (Beta + Beta.T)/2.
                self.Beta.append(Beta)
                # self.Beta.append(N/(SSE + 2.*self.b)*np.eye(Y_compressed.shape[-1]))

        ### M step: update hyper-parameters ###
        if self.A is None:
            # initial guess of parameter precision
            self.alpha = self.alpha_0
            self.Alpha = self.alpha_0*np.ones(self.n_params)
        else:
            # maximize complete data log-likelihood w.r.t. alpha and beta
            self.alpha = self.n_params/(np.dot(self.params-self.prior, self.params-self.prior) + np.trace(self.Ainv) + 2.*self.a)
            #self.Alpha = self.alpha*np.ones(self.n_params)
            self.Alpha = 1./((self.params-self.prior)**2 + np.diag(self.Ainv) + 2.*self.a)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(np.sum(self.N), self.alpha/self.beta[-1]))

    def objective(self, params):
        # compute residuals
        self.RES = 0.
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * (params-self.prior)**2) / 2.
        # compute gradient of negative log posterior
        self.grad_NLP = self.Alpha*(params-self.prior)

        # compute Hessian, covariance of y, sum of squares error
        self.A = np.diag(self.Alpha)

        for i, dataset in enumerate(self.datasets):
            # loop over each sample in dataset
            for t_eval, Y_init, Y_compressed, ctrl_params in dataset:            
                
                # run model using current parameters, output = [n_time, n_sys_vars]
                output = self.runODEZ(t_eval, Y_init, params, ctrl_params)
                Y_predicted = output[1:, :self.n_sys_vars]

                # collect gradients and reshape
                G = np.reshape(output[1:, self.n_sys_vars:],
                              [output[1:].shape[0], self.n_sys_vars, self.n_params])

                # compress model output and gradient 
                G = np.einsum('tck,tki->tci', self.compressor_primes[i](Y_predicted), G) 
                Y_predicted = self.compressors[i](Y_predicted)
                
                # Determine error
                Y_error = Y_predicted - Y_compressed[1:]

                # compute Hessian
                self.A += self.A_next(G, self.Beta[i])

                # Determine SSE and gradient of SSE
                self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta[i], Y_error)/2.
                self.RES += np.sum(Y_error)/self.N[i]

                # sum over time and outputs to get gradient w.r.t params
                self.grad_NLP += self.eval_grad_NLP(Y_error, self.Beta[i], G)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # return NLP and gradient of NLP
        return self.NLP, self.grad_NLP

    def update_covariance(self):
        # update parameter covariance matrix given current parameter estimate
        if self.A is None:
            self.A = self.alpha_0*np.eye(self.n_params)
        else:
            self.A = np.diag(self.Alpha)

        # loop over datasets
        for i, dataset in enumerate(self.datasets):
            # loop over each sample in dataset
            for t_eval, Y_init, Y_compressed, ctrl_params in dataset:

                # run model using current parameters, output = [n_time, n_sys_vars]
                output = self.runODEZ(t_eval, Y_init, self.params, ctrl_params)
                Y_predicted = output[1:, :self.n_sys_vars]

                # collect gradients and reshape
                G = np.reshape(output[1:, self.n_sys_vars:],
                              [output[1:].shape[0], self.n_sys_vars, self.n_params])

                # compress model output and gradient 
                G = np.einsum('tck,tki->tci', self.compressor_primes[i](Y_predicted), G) 
                
                # compute Hessian
                self.A += self.A_next(G, self.Beta[i])

        # Laplace approximation of posterior covariance
        self.A = (self.A + self.A.T)/2.
        self.Ainv = np.linalg.inv(self.A)
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

    def update_evidence(self):
        # compute evidence
        self.evidence = np.sum(np.log(self.Alpha))/2. - \
                        np.sum(np.log(np.linalg.eigvalsh(self.A)))/2. - \
                        self.NLP

        # loop over precision matrices from each dataset
        for N, Beta in zip(self.N, self.Beta):
            self.evidence += N*np.sum(np.log(np.linalg.eigvalsh(Beta)))/2.
        print("Evidence {:.3f}".format(self.evidence))

    def jacobian(self, params):
        # compute gradient of cost function
        return self.grad_NLP

    def hessian(self, params):
        # compute hessian of NLP
        return self.A

    def callback(self, xk, res=None):
        if self.verbose:
            print("Total weighted fitting error: {:.3f}".format(self.NLP))
        return True

    def predict(self, x_test, teval, ctrl_params=[], compressor=-1):
        # check if precision has been computed
        if self.A is None:
            self.update_covariance()

        # make predictions given initial conditions and evaluation times
        output = self.runODEZ(teval, x_test, self.params, ctrl_params)

        # reshape gradient
        G = np.reshape(output[:, self.n_sys_vars:],
                       [output.shape[0], self.n_sys_vars, self.n_params])

        # compress model output
        Y_predicted = output[:, :self.n_sys_vars]

        # calculate variance of each output (dimension = [steps, outputs])
        covariance = np.linalg.inv(self.Beta[compressor]) + self.GAinvG(G, self.Ainv)
        get_diag = vmap(jnp.diag, (0,))
        stdv = np.sqrt(get_diag(covariance))

        return Y_predicted, stdv, covariance
