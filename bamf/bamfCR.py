import numpy as np
import pandas as pd

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random, vjp
from jax.experimental.ode import odeint
from jax.scipy.linalg import block_diag

# import MCMC library
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

from tqdm import tqdm

import matplotlib.pyplot as plt

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
def runODE(t_eval, s0, r0, cr_params, s_present, dX_dt):
    # solve ODE model
    x0 = jnp.concatenate((s0, r0))
    y = odeint(dX_dt, x0, t_eval, cr_params, s_present)
    # jac = jit(jacfwd(dX_dt, 1))
    # soln = solve_ivp(dX_dt, t_span=(t_eval[0], t_eval[-1]), y0=x0,
    #                  args=(cr_params,), t_eval=t_eval, method='LSODA', jac=jac)

    return y

# define function to integrate adjoint sensitivity equations backwards
def runODEA(t_eval, zt, at, cr_params, s_present, dXA_dt):
    # check dimensions
    n_params = len(cr_params)
    lt = np.zeros(n_params)

    # concatenate final condition
    xal = (zt, at, lt)

    # solve ODE model
    x0, a0, l0 = odeint(dXA_dt, xal, t_eval, cr_params, s_present)

    return jnp.concatenate([a0[-1], l0[-1]])

# define function that returns model sensitivity vector
def runODEZ(t_eval, s0, r0, cr_params, s_present, dXZ_dt):
    # check dimensions
    dim_x = len(s0) + len(r0)
    n_params = len(cr_params)

    # set initial condition of z0 equal to I
    z0 = np.eye(dim_x)[:,len(s0):]
    xz = (jnp.concatenate([s0, r0]), z0, np.zeros([dim_x, n_params]))

    # solve ODE model
    y, Z0, Z = odeint(dXZ_dt, xz, t_eval, cr_params, s_present)

    return y, Z0, Z

### Function to process dataframes ###
def process_df(df, species):

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data = comm_data.sort_values(by='Time', ascending=True).copy()

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull system data
        Y_measured = np.array(comm_data[species].values, float)

        # pull system data
        s_present  = np.array(Y_measured[0]>0, int)
        Y_measured = np.einsum("ij,j->ij", np.log(Y_measured+1e-8), s_present)

        # append t_eval and Y_measured to data list
        data.append([treatment, t_eval, Y_measured, s_present])

    return data

class ODE:
    def __init__(self, system, dataframe, cr_params, r0, species,
                 alpha_0=1e-5, log_tol=1e-5, batch_size=None, prior=None, verbose=True):
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
        self.params = np.concatenate((r0, np.array(cr_params).ravel()))
        self.prior  = np.concatenate((-5.*np.ones_like(r0), np.array(prior).ravel()))

        # initial degree of regularization
        self.alpha_0 = alpha_0

        # batch_size
        self.batch_size = batch_size

        # number of parameters
        self.n_s = len(species)
        self.n_r = len(r0)
        self.n_params = len(self.params)

        # dimension of model output
        self.species = species
        self.n_sys_vars = self.n_s + self.n_r

        # set up data
        self.dataset = process_df(dataframe, species)

        # for additional output messages
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-4

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        # jit compile differential equation
        def dX_dt(x, t, params, s_present):
            # concatentate x and z
            return system(x, t, params, s_present)
        self.dX_dt = jit(dX_dt)

        # adjoint sensitivity derivative
        def dXA_dt(xa, t, params, s_present):

            # unpack state, adjoint, and gradient of loss w.r.t. parameters
            x, a, _ = xa
            # vjp returns system evaluated at x,t,params and
            # vjpfun, which evaluates a^T Jx, a^T Jp
            # where Jx is the gradient of the system w.r.t. x
            # and   Jp is the gradient of the system w.r.t. parameters
            y_dot, vjpfun = jax.vjp(lambda x, params: system(x,t,params,s_present), x, params)
            vjps = vjpfun(a)

            return (-y_dot, *vjps)
        self.dXA_dt = jit(dXA_dt)

        # if not vectorized, xz will be 1-D
        def dXZ_dt(xZ, t, params, s_present):
            # split up x, z, and z0
            x, Z0, Z = xZ

            # compute derivatives
            dxdt  = system(x, t, params, s_present)

            # time derivative of initial condition sensitivity
            # Jacobian-vector-product approach is surprisingly slow
            # JxV  = vmap(lambda z: jax.jvp(lambda x: system(x,t,params), (x,), (z,))[1], (1), (1))
            Jx = jacfwd(system, 0)(x, t, params, s_present)
            JxZ0 = Jx@Z0 # JxV(Z0)

            # time derivative of parameter sensitivity
            JxZ = Jx@Z # JxV(Z)

            # compute gradient of model w.r.t. parameters
            Jp = jacfwd(system, 2)(x, t, params, s_present)

            # return derivatives
            return (dxdt, JxZ0, JxZ + Jp)
        self.dXZ_dt = jit(dXZ_dt)

        # jit compile function to integrate ODE
        self.runODE  = jit(lambda t_eval, x, r0, params, s_present: runODE(t_eval, x[0], r0, params, s_present, self.dX_dt))
        # self.batchODE = jit(vmap(self.runODE, (None, 0, None, None)))

        # jit compile function to integrate forward sensitivity equations
        self.runODEZ = jit(lambda t_eval, x, r0, params, s_present: runODEZ(t_eval, x[0], r0, params, s_present, self.dXZ_dt))
        # self.batchODEZ = jit(vmap(self.runODEZ, (None, 0, None, None)))

        # jit compile function to integrate adjoint sensitivity equations
        self.adjoint = jit(vmap(jacfwd(lambda zt, yt, B: jnp.einsum("i,ij,j", yt-zt[:self.n_s], B, yt-zt[:self.n_s])/2.), (0, 0, None)))
        self.runODEA = jit(lambda t, xt, at, cr_params, s_present: runODEA(jnp.array([0., t]), xt, at, cr_params, s_present, self.dXA_dt)[self.n_s:])
        # running in batches is slower for some reason...
        #self.batchODEA = jit(vmap(lambda t, out, a, p: self.runODEA(jnp.array([0., t]), out, a, p)[self.n_s:], (0, 0, 0, None)))

        # JIT compile matrix operations
        def GAinvG(G, Ainv):
            return jnp.einsum("tij,jl,tml->tim", G, Ainv, G)
        self.GAinvG = jit(GAinvG)

        def yCOV_next(Y_error, G, Ainv):
            # sum over time dimension
            return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.sum(self.GAinvG(G, Ainv), 0)
        self.yCOV_next = jit(yCOV_next)

        def A_next(G, Beta):
            A_n = jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
            A_n = (A_n + A_n.T)/2.
            return A_n
        self.A_next = jit(A_next)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = jit(Ainv_next)

        # jit compile inverse Hessian computation step
        def Ainv_prev(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(GAinv@G.T - BetaInv)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_prev = jit(Ainv_prev)

        # jit compile function to compute log of determinant of a matrix
        def log_det(A):
            L = jnp.linalg.cholesky(A)
            return 2*jnp.sum(jnp.log(jnp.diag(L)))
        self.log_det = jit(log_det)

        # log transform
        def log_y(y):
            return jnp.log(y+log_tol)
        self.log_y = jit(log_y)

        def log_err(f, y):
            return jnp.log((f+log_tol)/(y+log_tol))
        self.log_err = jit(log_err)

        # pad zeros
        def pad_zeros(y):
            y_copy = np.copy(y)
            y_copy[y_copy==0] = 1.
            return np.clip(y_copy, log_tol, np.inf)
        self.pad_zeros = pad_zeros

        # approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
        def compute_Ainv(A):
            Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
            Ainv = Linv.T@Linv
            return Ainv
        self.compute_Ainv = jit(compute_Ainv)

        def NewtonStep(A, g):
            return jnp.linalg.solve(A,g)
        self.NewtonStep = jit(NewtonStep)

        def eval_grad_NLP(Y_error, Beta, G):
            return jnp.einsum('tk,kl,tli->i', Y_error, Beta, G)
        self.eval_grad_NLP = jit(eval_grad_NLP)

        # jit compile prediction covariance computation
        def compute_searchCOV(Beta, G, Ainv):
            # dimensions of sample
            n_t, n_y, n_theta = G.shape
            # stack G over time points [n, n_t, n_out, n_theta]--> [n, n_t*n_out, n_theta]
            Gaug = jnp.concatenate(G, 0)
            return jnp.eye(n_t*n_y) + jnp.einsum("kl,li,ij,mj->km", block_diag(*[Beta]*n_t), Gaug, Ainv, Gaug)
        self.compute_searchCOV = jit(compute_searchCOV)

        # jit compile prediction covariance computation
        def compute_forgetCOV(Beta, G, Ainv):
            # dimensions of sample
            n_t, n_y, n_theta = G.shape
            # stack G over time points [n, n_t, n_out, n_theta]--> [n, n_t*n_out, n_theta]
            Gaug = jnp.concatenate(G, 0)
            return jnp.eye(n_t*n_y) - jnp.einsum("kl,li,ij,mj->km", block_diag(*[Beta]*n_t), Gaug, Ainv, Gaug)
        self.compute_forgetCOV = jit(compute_forgetCOV)

    def fit_BFGS(self, evidence_tol=1e-3, nlp_tol=None, patience=1, max_fails=2, beta=1e-3):
        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence  = -np.inf

        # initialize hyper parameters
        self.init_hypers()

        while passes < patience and fails < max_fails:
            # update Alpha and Beta hyper-parameters
            if self.itr>0: self.update_hypers()

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian, #hess=self.hessian,
                                x0=self.params,
                                tol=nlp_tol,
                                method='BFGS',
                                callback=self.callback)
            # if self.verbose:
            #     print(self.res)
            self.params = self.res.x

            # BFGS approximation
            self.Ainv = self.res.hess_inv

            # use cholesky decomposition to check positive-definiteness of A
            while jnp.isnan(jnp.linalg.cholesky(self.Ainv)).any():

                # increase precision of prior until posterior precision is positive definite
                self.Ainv += beta*np.eye(self.n_params)

                # increase prior precision
                beta *= 2.

    def fit(self, evidence_tol=1e-3, nlp_tol=None, patience=1, max_fails=2, beta=1e-3):
        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence  = -np.inf

        # initialize hyper parameters
        self.init_hypers()

        while passes < patience and fails < max_fails:
            # update Alpha and Beta hyper-parameters
            if self.itr>0: self.update_hypers()

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian_fwd,
                                hess=self.hessian,
                                x0=self.params,
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            # if self.verbose:
            #     print(self.res)
            self.params = self.res.x

            # update precision
            self.update_precision()

            # make sure that precision is positive definite (algorithm 3.3 in Numerical Optimization)
            if np.min(np.diag(self.A)) > 0:
                tau = 0.
            else:
                tau = beta - np.min(np.diag(self.A))

            # use cholesky decomposition to check positive-definiteness of A
            while jnp.isnan(jnp.linalg.cholesky(self.A)).any():

                # increase precision of prior until posterior precision is positive definite
                self.A += tau*np.diag(self.Alpha)

                # increase prior precision
                tau = np.max([2*tau, beta])
                self.A += tau*np.diag(self.Alpha)

            # update evidence
            self.update_evidence()
            assert not np.isnan(self.evidence), "Evidence is NaN! Something went wrong."

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1.,np.abs(self.evidence)])

            # update pass count
            if convergence < evidence_tol:
                passes += 1
                print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                print("Fail count ", fails)
            else:
                fails = 0

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

        # finally compute covariance (Hessian inverse)
        self.final_covariance()

    def init_hypers(self):

        # count number of samples
        self.N = 0

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:
            # count effective number of uncorrelated observations
            # self.N += len(t_eval[1:]) * np.sum(np.sum(Y_measured, 0) > 0) / self.C.shape[0]
            k = 0 # number of outputs
            N = 0 # number of samples
            for series in Y_measured.T:
                # check if there is any variation in the series
                if np.std(series) > 0:
                    # count number of outputs that vary over time
                    k += 1

                    # determine lag between uncorrelated samples
                    # lag = [i for i,j in enumerate(acf(series) < .5) if j][0]

                    # count number of uncorrelated samples in series
                    # N += len(series[::lag]) - 1

                    # Evidence optimization collapses with reduced N, so using full N instead
                    N += len(series) - 1
            assert k > 0, f"There are no time varying outputs in sample {treatment}"
            self.N += N / k

        # init output precision
        self.n_total = self.N*self.n_s
        self.Beta = np.eye(self.n_s)
        self.BetaInv = np.eye(self.n_s)

        # initial guess of parameter precision
        self.alpha = self.alpha_0
        self.Alpha = self.alpha_0*np.ones(self.n_params)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    # EM algorithm to update hyper-parameters
    def update_hypers(self):
        print("Updating hyper-parameters...")

        # compute inverse of cholesky decomposed precision matrix
        Ainv = self.compute_Ainv(self.A)

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            # run model using current parameters, output = [n_time, self.n_sys_vars]
            # outputs = np.nan_to_num(self.batchODEZ(t_eval, Y_batch[batch_inds], self.params[:self.n_r], self.params[self.n_r:]))

            # for each output
            output, Z0, Z = self.runODEZ(t_eval, Y_measured, self.params[:self.n_r], self.params[self.n_r:], s_present)

            # stack gradient matrices
            G = np.concatenate((Z0, Z), axis=-1)[1:, :self.n_s]

            # Determine SSE
            Y_error = output[1:, :self.n_s] - Y_measured[1:]
            yCOV += self.yCOV_next(Y_error, G, Ainv)

        ### M step: update hyper-parameters ###

        # maximize complete data log-likelihood w.r.t. alpha and beta
        Ainv_ii = np.diag(Ainv)
        self.alpha = self.n_params/(np.sum((self.params-self.prior)**2) + np.sum(Ainv_ii) + 2.*self.a)
        # self.Alpha = self.alpha*np.ones(self.n_params)
        self.Alpha = 1./((self.params-self.prior)**2 + Ainv_ii + 2.*self.a)

        # update output precision
        self.Beta = self.N*np.linalg.inv(yCOV + 2.*self.b*np.eye(self.n_s))
        self.Beta = (self.Beta + self.Beta.T)/2.
        self.BetaInv = np.linalg.inv(self.Beta)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    def objective(self, params):
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * (params-self.prior)**2) / 2.
        # compute residuals
        self.RES = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            # run model in batches
            # for batch_inds in np.array_split(np.arange(n_samples), n_samples//self.batch_size):
            # outputs = np.nan_to_num(self.batchODE(t_eval, Y_batch[batch_inds], params[:self.n_r], params[self.n_r:]))

            # for each output
            output = np.nan_to_num(self.runODE(t_eval, Y_measured, params[:self.n_r], params[self.n_r:], s_present))

            # Determine error
            Y_error = output[1:, :self.n_s] - Y_measured[1:]

            # Determine SSE and gradient of SSE
            self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error)/2.
            self.RES += np.sum(Y_error)/self.n_total

        # return NLP
        return self.NLP

    def jacobian(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            ### Using vmap to integrate ODEs in parallel seems to be slower?
            # for batch_inds in np.array_split(np.arange(n_samples), n_samples//self.batch_size):
            # outputs = np.nan_to_num(self.batchODEZ(t_eval, Y_batch[batch_inds], self.params[:self.n_r], self.params[self.n_r:]))

            ### Adjoint sensitivity method can be slower depending on stiffness of ODE
            output = np.nan_to_num(self.runODE(t_eval, Y_measured, params[:self.n_r], params[self.n_r:], s_present))

            # adjoint at measured time points
            at = self.adjoint(output, Y_measured, self.Beta)

            # gradient of NLP
            for t, out, a in zip(t_eval[1:], output[1:], at[1:]):
                grad_NLP += self.runODEA(t, out, a, params[self.n_r:], s_present)

            if np.any(np.isnan(grad_NLP)):
                print("NaN gradient, switching to forward sensitivity method!")
                return self.jacobian_fwd(params)

        # return gradient of NLP
        return grad_NLP

    def jacobian_fwd(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            # for each output
            output, Z0, Z = self.runODEZ(t_eval, Y_measured, params[:self.n_r], params[self.n_r:], s_present)

            # stack gradient matrices
            G = np.concatenate((Z0, Z), axis=-1)[1:, :self.n_s]

            # Determine error
            Y_error = output[1:, :self.n_s] - Y_measured[1:]

            # sum over time and outputs to get gradient w.r.t params
            grad_NLP += self.eval_grad_NLP(Y_error, self.Beta, G)

        # return gradient of NLP
        return grad_NLP

    def hessian(self, params):

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            # run model in batches
            # for batch_inds in np.array_split(np.arange(n_samples), n_samples//self.batch_size):
            # outputs = np.nan_to_num(self.batchODEZ(t_eval, Y_batch[batch_inds], params[:self.n_r], params[self.n_r:]))

            # for each output
            output, Z0, Z = self.runODEZ(t_eval, Y_measured, params[:self.n_r], params[self.n_r:], s_present)

            # stack gradient matrices
            G = np.concatenate((Z0, Z), axis=-1)[1:, :self.n_s]

            # compute Hessian
            self.A += self.A_next(G, self.Beta)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # return Hessian
        return self.A

    def update_precision(self):
        # update parameter covariance matrix given current parameter estimate
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, s_present in self.dataset:

            # run model in batches
            # for batch_inds in np.array_split(np.arange(n_samples), n_samples//self.batch_size):
            # outputs = np.nan_to_num(self.batchODEZ(t_eval, Y_batch[batch_inds], self.params[:self.n_r], self.params[self.n_r:]))

            # for each output
            output, Z0, Z = self.runODEZ(t_eval, Y_measured, self.params[:self.n_r], self.params[self.n_r:], s_present)

            # stack gradient matrices
            G = np.concatenate((Z0, Z), axis=-1)[1:, :self.n_s]

            # compute Hessian
            self.A += self.A_next(G, self.Beta)

        # Laplace approximation of posterior precision
        self.A = (self.A + self.A.T)/2.

    def final_covariance(self):
        ### Approximate / fast method to compute inverse ###
        self.Ainv = self.compute_Ainv(self.A)

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = self.N/2*self.log_det(self.Beta)  + \
                        1/2*np.nansum(np.log(self.Alpha)) - \
                        1/2*self.log_det(self.A) - self.NLP

        # print evidence
        if self.verbose:
            print("Evidence {:.3f}".format(self.evidence))

    def callback(self, xk, res=None):
        if self.verbose:
            print("Total weighted fitting error: {:.3f}".format(self.NLP))
        return True

    def predict_point(self, x_test, teval):

        # make predictions given initial conditions and evaluation times
        s_present = np.array(x_test > 0, int)
        Y_predicted = np.nan_to_num(self.runODE(teval, np.atleast_2d(np.log(x_test+1e-8)), self.params[:self.n_r], self.params[self.n_r:], s_present))

        return np.einsum('ij,j->ij', np.exp(Y_predicted), s_present)

    def predict(self, x_test, t_eval, n_std=1.):
        # check if precision has been computed

        # for each output
        s_present = np.array(x_test > 0, int)
        output, Z0, Z = self.runODEZ(t_eval, np.atleast_2d(np.log(x_test+1e-8)), self.params[:self.n_r], self.params[self.n_r:], s_present)

        # stack gradient matrices
        G = np.concatenate((Z0, Z), axis=-1)[:, :self.n_s]

        # calculate covariance of each output (dimension = [steps, outputs])
        covariance = self.BetaInv + self.GAinvG(G, self.Ainv)

        # predicted stdv of log y
        get_diag = vmap(jnp.diag, (0,))
        stdv = np.sqrt(get_diag(covariance))

        # determine confidence interval for species
        ls_out = output[:, :self.n_s]
        lr_out = output[:, self.n_s:]

        # determine confidence interval for species
        L = np.einsum("ij,j->ij", np.exp(ls_out - n_std*stdv), s_present)
        U = np.einsum("ij,j->ij", np.exp(ls_out + n_std*stdv), s_present)

        return np.einsum("ij,j->ij", np.exp(ls_out), s_present), L, U, np.exp(lr_out)

    # function to predict from posterior samples
    def predict_MC(self, x_test, t_eval, n_samples=100):

        # monte carlo draws from posterior
        z = np.random.randn(n_samples, self.n_params)
        Linv = np.linalg.cholesky(self.Ainv)
        posterior_params = self.params + np.einsum('jk,ik->ij', Linv, z)

        # make point predictions of shape [n_mcmc, n_samples, n_time, n_outputs]
        preds = vmap(lambda params: jnp.nan_to_num(self.runODE(t_eval, jnp.atleast_2d(x_test), params[:self.n_r], params[self.n_r:])), (0,))(posterior_params)

        return preds
