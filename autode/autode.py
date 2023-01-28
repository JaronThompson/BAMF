import numpy as np
import pandas as pd

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random
from jax.experimental.ode import odeint
from jax.scipy.linalg import block_diag

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

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull initial condition
        Y_init = np.array(comm_data[sys_vars].values[0], float)

        # pull system data
        Y_observed = np.array(comm_data[measured_vars].values, float)

        # pull control params
        ctrl_params = np.array(comm_data[controls].values, float)

        # append t_eval and Y_observed to data list
        data.append([treatment, t_eval, Y_init, Y_observed, ctrl_params])

    return data

class ODE:
    def __init__(self, system, df, params, sys_vars, measured_vars=None, prior=None,
                 controls = [], compressor=None, alpha_0=1e-5, verbose=True):
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
        if measured_vars is None:
            self.measured_vars = sys_vars
        else:
            self.measured_vars = measured_vars
        self.n_sys_vars = len(sys_vars)

        # control input
        self.controls = controls
        self.n_ctrls = len(controls)

        # set compressors (observability function, y = c(x))
        if compressor is None:
            # set default observability function to identity
            self.compressor = jit(vmap(lambda x: x ))
        else:
            self.compressor = jit(vmap(compressor))

        # store derivative of compressors
        self.compressor_prime = jit(vmap(jacfwd(self.compressor)))

        # set up data
        self.dataset = process_df(df, self.sys_vars, self.measured_vars, self.controls)

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
            return jnp.sum(Y_error**2) + jnp.clip(jnp.einsum('tki,ij,tlj->', G, Ainv, G), 0., jnp.inf)
        self.SSE_next = jit(SSE_next)

        def yCOV_next(Y_error, G, Ainv):
            return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.clip(jnp.einsum('tki,ij,tlj->kl', G, Ainv, G), 0., jnp.inf)
        self.yCOV_next = jit(yCOV_next)

        def A_next(G, Beta):
            return jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
        self.A_next = jit(A_next)

        def GAinvG(G, Ainv):
            return jnp.clip(jnp.einsum('tki,ij,tlj->tkl', G, Ainv, G), 0., jnp.inf)
        self.GAinvG = jit(GAinvG)

        def NewtonStep(A, g):
            return jnp.linalg.solve(A,g)
        self.NewtonStep = jit(NewtonStep)

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

    def fit(self, evidence_tol=1e-3, nlp_tol=None, patience=1, max_fails=2):
        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence  = -np.inf

        while passes < patience and fails < max_fails:
            # update Alpha and Beta hyper-parameters
            self.update_precision()

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=True,
                                hess=self.hessian,
                                x0=self.params,
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            if self.verbose:
                print(self.res)
            self.params = self.res.x

            # update covariance
            self.update_covariance()

            # make sure that precision is positive-definite
            p_eigs = np.linalg.eigvalsh(self.A)
            gap = np.abs(np.clip(np.min(p_eigs), -np.inf, 0.))
            eps = 0.
            if gap > 0: print("Hessian not positive definite, increasing regularization...")
            while gap > 0:
                # increase prior precision
                eps += .5
                p_eigs += eps*self.Alpha
                gap = np.abs(np.clip(np.min(p_eigs), -np.inf, 0.))
            if eps > 0.:
                self.Alpha += eps*self.Alpha
                self.update_covariance()

            # update evidence
            self.update_evidence()

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

    # EM algorithm to update hyper-parameters
    def update_precision(self):
        if self.verbose:
            print("Updating precision...")

        # loop over each sample in dataset
        SSE = 0.
        yCOV = 0.
        self.N = 0
        for treatment, t_eval, Y_init, Y_observed, ctrl_params in self.dataset:

            # count number of observations
            self.N += len(t_eval[1:]) * np.sum(np.sum(Y_observed, 0) > 0) / self.n_sys_vars

            # run model using current parameters
            if self.A is not None:

                # run model using current parameters, output = [n_time, n_sys_vars]
                output = np.nan_to_num(self.runODEZ(t_eval, Y_init, self.params, ctrl_params))
                Y_predicted = output[1:, :self.n_sys_vars]

                # collect gradients and reshape
                G = np.reshape(output[1:, self.n_sys_vars:],
                              [output[1:].shape[0], self.n_sys_vars, self.n_params])

                # compress model output and gradient
                G = np.einsum('tck,tki->tci', self.compressor_prime(Y_predicted), G)
                Y_predicted = self.compressor(Y_predicted)

                # Determine SSE
                Y_error = Y_predicted - Y_observed[1:]
                SSE  += self.SSE_next(Y_error, G, self.Ainv)
                yCOV += self.yCOV_next(Y_error, G, self.Ainv)

        ### M step: update hyper-parameters ###
        if self.A is None:
            # init output precision
            self.beta = 1.
            self.Beta = np.eye(Y_observed.shape[-1])
            self.BetaInv = np.eye(Y_observed.shape[-1])

            # initial guess of parameter precision
            self.alpha = self.alpha_0
            self.Alpha = self.alpha_0*np.ones(self.n_params)
        else:
            # update precision
            self.beta = self.N*Y_observed.shape[-1]/(SSE + 2.*self.b)
            self.Beta = self.N*np.linalg.inv(yCOV + 2.*self.b*np.eye(Y_observed.shape[-1]))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)

            # maximize complete data log-likelihood w.r.t. alpha and beta
            self.alpha = self.n_params/(np.dot(self.params-self.prior, self.params-self.prior) + np.trace(self.Ainv) + 2.*self.a)
            #self.Alpha = self.alpha*np.ones(self.n_params)
            self.Alpha = 1./((self.params-self.prior)**2 + np.clip(np.diag(self.Ainv), 0., np.inf) + 2.*self.a)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha/self.beta))

    def objective(self, params):
        # compute residuals
        self.RES = 0.
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * (params-self.prior)**2) / 2.
        # compute gradient of negative log posterior
        self.grad_NLP = self.Alpha*(params-self.prior)

        # loop over each sample in dataset
        for treatment, t_eval, Y_init, Y_observed, ctrl_params in self.dataset:

            # run model using current parameters, output = [n_time, n_sys_vars]
            output = np.nan_to_num(self.runODEZ(t_eval, Y_init, params, ctrl_params))
            Y_predicted = output[1:, :self.n_sys_vars]

            # collect gradients and reshape
            G = np.reshape(output[1:, self.n_sys_vars:],
                          [output[1:].shape[0], self.n_sys_vars, self.n_params])

            # compress model output and gradient
            G = np.einsum('tck,tki->tci', self.compressor_prime(Y_predicted), G)
            Y_predicted = self.compressor(Y_predicted)

            # Determine error
            Y_error = Y_predicted - Y_observed[1:]

            # Determine SSE and gradient of SSE
            self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error)/2.
            self.RES += np.sum(Y_error)/self.N

            # sum over time and outputs to get gradient w.r.t params
            self.grad_NLP += self.eval_grad_NLP(Y_error, self.Beta, G)

        # return NLP and gradient of NLP
        return self.NLP, self.grad_NLP

    def hessian(self, params):

        # compute Hessian, covariance of y, sum of squares error
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_init, Y_observed, ctrl_params in self.dataset:

            # run model using current parameters, output = [n_time, n_sys_vars]
            output = np.nan_to_num(self.runODEZ(t_eval, Y_init, params, ctrl_params))
            Y_predicted = output[1:, :self.n_sys_vars]

            # collect gradients and reshape
            G = np.reshape(output[1:, self.n_sys_vars:],
                          [output[1:].shape[0], self.n_sys_vars, self.n_params])

            # compress model output and gradient
            G = np.einsum('tck,tki->tci', self.compressor_prime(Y_predicted), G)

            # compute Hessian
            self.A += self.A_next(G, self.Beta)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # return NLP and gradient of NLP
        return self.A

    def update_covariance(self):
        # update parameter covariance matrix given current parameter estimate
        if self.A is None:
            self.A = self.alpha_0*np.eye(self.n_params)
            self.Ainv = np.eye(self.n_params)/self.alpha_0
        else:
            self.A = np.diag(self.Alpha)
            self.Ainv = np.diag(1./self.Alpha)

        # loop over datasets
        for treatment, t_eval, Y_init, Y_observed, ctrl_params in self.dataset:

            # run model using current parameters, output = [n_time, n_sys_vars]
            output = np.nan_to_num(self.runODEZ(t_eval, Y_init, self.params, ctrl_params))
            Y_predicted = output[1:, :self.n_sys_vars]

            # collect gradients and reshape
            G = np.reshape(output[1:, self.n_sys_vars:],
                          [output[1:].shape[0], self.n_sys_vars, self.n_params])

            # compress model output and gradient
            G = np.einsum('tck,tki->tci', self.compressor_prime(Y_predicted), G)

            # compute Hessian
            self.A += self.A_next(G, self.Beta)

            # compute covariance
            for Gt in G:
                self.Ainv -= self.Ainv_next(Gt, self.Ainv, self.BetaInv)

        # Laplace approximation of posterior covariance
        self.A = (self.A + self.A.T)/2.
        # self.Ainv = np.linalg.inv(self.A)
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

    def update_evidence(self):
        # eigenvalues of precision (hessian of NLP)
        p_eigs = np.linalg.eigvalsh(self.A)

        # compute evidence
        self.evidence = np.sum(np.log(self.Alpha))/2. - \
                        np.sum(np.log(p_eigs[p_eigs>0]))/2. - \
                        self.NLP + self.N*np.sum(np.log(np.linalg.eigvalsh(self.Beta)))/2.

        # loop over precision matrices from each dataset
        if self.verbose:
            print("Evidence {:.3f}".format(self.evidence))

    def callback(self, xk, res=None):
        if self.verbose:
            print("Total weighted fitting error: {:.3f}".format(self.NLP))
        return True

    def predict(self, x_test, teval, ctrl_params=[]):
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

        # calculate covariance of each output (dimension = [steps, outputs])
        covariance = self.BetaInv + self.GAinvG(G, self.Ainv)

        # predicted stdv
        get_diag = vmap(jnp.diag, (0,))
        stdv = np.sqrt(get_diag(covariance))

        return Y_predicted, stdv, covariance

    ### MCMC Code ###
    def fit_MCMC(self,
                 num_warmup=1000,
                 num_samples=1000,
                 num_subsample=10,
                 num_chains=5,
                 scale=1.,
                 seed=0):

        # adjusting the scale adjusts the acceptance ratio
        # ideally the acceptance ratio is between .2 and .3
        # if the acceptance ratio is too high, increase the scale
        # if the acceptance ratio is too low, reduce the scale
        # increasing the scale increases the stdv. of the proposal distribution

        # generate random key
        key = random.PRNGKey(seed)

        # total number of samples = number of chains * num_samples
        self.num_chains = num_chains
        num_samples *= num_chains*num_subsample
        num_samples += num_warmup

        # scaled Cholesky decomposition of Laplace estimated parameter covariance
        L = scale*np.linalg.cholesky(self.Ainv)/np.sqrt(self.n_params)

        # compile functions for sampling and evaluating samples
        print("Compiling...")

        # use Laplace approximated posterior as proposal distribution
        @jit
        def sample_proposal(subkey, current_theta):

            # sample from Gaussian
            z_theta = random.normal(subkey, shape=(self.n_params,))
            theta = current_theta + L@z_theta

            # return samples
            return theta

        # compile function to compute acceptance probability
        @jit
        def acceptance_prob(new_sse, old_sse, new_theta, old_theta):

            # log of acceptance probability
            log_ratio = (-new_sse
                         -self.alpha*jnp.sum(new_theta**2)
                         +old_sse
                         +self.alpha*jnp.sum(old_theta**2))

            # take exponent and return
            return jnp.exp(log_ratio/2.)

        # function to make predictions
        @jit
        def sse(theta):
            sse_val = 0.
            for treatment, t_eval, Y_init, Y_observed, ctrl_params in self.dataset:
                pred = self.compressor(self.runODE(t_eval, Y_init, theta, ctrl_params)[1:, :self.n_sys_vars])
                sse_val += jnp.einsum('tk,kl,tl', Y_observed[1:]-pred, self.Beta, Y_observed[1:]-pred)
            return sse_val

        # run MCMC and store samples
        acc_theta = [np.copy(self.params)]
        old_sse = sse(acc_theta[-1])
        n_acc = 0
        n_try = 0
        for _ in tqdm(range(num_samples), desc='Run'):
            # generate new random key
            key, subkey = random.split(key)

            # sample from proposal
            new_theta = sample_proposal(subkey, acc_theta[-1])

            # make predictions
            new_sse = sse(new_theta)

            # compute acceptance probability
            acc_prb = acceptance_prob(new_sse, old_sse, new_theta, acc_theta[-1])

            # decide whether to accept new parameters
            if np.min([1., acc_prb]) > random.uniform(subkey):
                acc_theta.append(new_theta)
                old_sse = new_sse
                n_acc += 1
            else:
                acc_theta.append(acc_theta[-1])

            # count number of trials
            n_try += 1

        # store posterior samples
        self.posterior_params = np.stack(acc_theta)[num_warmup+1::num_subsample]

        # acceptance probability
        print("Acceptance ratio: {:.2f}\n".format(n_acc / n_try))

        # compute potential scale reduction
        chains = np.stack(np.split(self.posterior_params, self.num_chains))
        m,n,p = chains.shape
        B = n/(m-1) * np.sum((np.mean(chains, 1) - np.mean(chains, (0,1)))**2, 0)
        W = np.mean( 1/(n-1)*np.sum((np.transpose(chains, (1,0,2)) - np.mean(chains, 1))**2 , 0) , 0)
        var_plus = (n-1)/n*W + B/n
        R = np.round(np.sqrt(var_plus/W),2)

        # save statistics
        self.mcmc_summary = pd.DataFrame()
        self.mcmc_summary['Params'] = ['w'+str(i+1) for i in range(self.n_params)]
        self.mcmc_summary['mean'] = np.mean(self.posterior_params, 0)
        self.mcmc_summary['median'] = np.median(self.posterior_params, 0)
        self.mcmc_summary['stdv'] = np.std(self.posterior_params, 0)
        self.mcmc_summary['r_hat'] = R

    def compute_effective_samples():

        # compute potential scale reduction
        chains = np.stack(np.split(self.posterior_params, self.num_chains))
        m,n,p = chains.shape
        B = n/(m-1) * np.sum((np.mean(chains, 1) - np.mean(chains, (0,1)))**2, 0)
        W = np.mean( 1/(n-1)*np.sum((np.transpose(chains, (1,0,2)) - np.mean(chains, 1))**2 , 0) , 0)
        var_plus = (n-1)/n*W + B/n

        # variogram of t
        Vt = lambda t: 1/(m*(n-t))*np.sum(np.sum(np.stack([(chains[:, i] - chains[:, i-t])**2 for i in range(t+1,n)]), 0), 0)
        # lag of t
        pt = lambda t: 1 - Vt(t) / (2*var_plus)

        # compute effective number of samples for each feature
        n_eff = []
        for i in range(self.n_params):
            t = 1
            pt_sum = 0
            while pt(t+1)[i]+pt(t+2)[i] > 0 and t < n-4:
                pt_sum += pt(t)[i]
                t += 1
            n_eff.append(m*n / (1 + 2*pt_sum))

        # save statistics
        self.mcmc_summary['n_eff'] = n_eff
        return self.mcmc_summary

    # function to predict from posterior samples
    def predict_MCMC(self, x_test, t_eval, ctrl_params=[]):

        # make point predictions of shape [n_mcmc, n_samples, n_time, n_outputs]
        preds = vmap(lambda params: self.runODE(t_eval, x_test, params, ctrl_params), (0,))(self.posterior_params)

        return preds

    def search(self, df_main, N, batch_size=512):
        # process dataframe
        if self.verbose:
            print("Processing design dataframe...")
        design_space = process_df(df_main, self.sys_vars, self.measured_vars, self.controls)

        # total number of possible experimental conditions
        n_samples = len(design_space)
        batch_size = np.min([batch_size, n_samples])

        # init parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # store sensitivity to each condition
        if self.verbose:
            print("Computing sensitivies...")
        Gs = {}
        exp_names = []
        for treatment, t_eval, Y_init, Y_observed, ctrl_params in design_space:
            exp_names.append(treatment)

            # run model using current parameters, output = [n_time, n_sys_vars]
            output = np.nan_to_num(self.runODEZ(t_eval, Y_init, self.params, ctrl_params))
            Y_predicted = output[1:, :self.n_sys_vars]

            # collect gradients and reshape
            G = np.reshape(output[1:, self.n_sys_vars:],
                          [output[1:].shape[0], self.n_sys_vars, self.n_params])

            # compress model output and gradient
            G = np.einsum('tck,tki->tci', self.compressor_prime(Y_predicted), G)

            # store in hash table of sensitivies
            Gs[treatment] = G

        # # randomly select experiments
        # best_experiments = list(np.random.choice(exp_names, 10, replace=False))
        # for exp in best_experiments:
        #     # update parameter covariance given selected experiment
        #     for Gt in Gs[exp]:
        #         Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # search for new experiments
        best_experiments = []
        N_selected = 0
        while N_selected < N:

            # compute information content of each observation
            f_I = []
            for treatment in exp_names:
                # predCOV has shape [n_time, n_out, n_out]
                searchCOV = self.compute_searchCOV(self.Beta, Gs[treatment], Ainv_q)
                f_I.append(self.fast_utility(searchCOV))

            # concatenate utilities
            utilities = np.array(f_I).ravel()
            # print("Top 5 utilities:, ", np.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                treatment, t_eval, Y_init, Y_observed, ctrl_params = design_space[exp]
                if treatment not in best_experiments:
                    best_experiments.append(treatment)
                    # number of selected observations is the evaluation time
                    # minus 1 to ignore initial condition
                    N_selected += len(t_eval) - 1

                    # update parameter covariance given selected experiment
                    for Gt in Gs[treatment]:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

                    # select next sample
                    print(f"Picked {treatment}")
                    break
                else:
                    print("Picked duplicate!")

        ### Algorithm to continue improving design ###
        while True:
            # Find point that, when dropped, results in smallest decrease in EIG
            f_L = []
            for treatment in best_experiments:
                # compute impact of losing this point
                # | A - G' B G |
                forgetCOV = self.compute_forgetCOV(self.Beta, Gs[treatment], Ainv_q)
                f_L.append(self.fast_utility(forgetCOV))
            worst_exp = best_experiments[np.argmax(f_L)]

            # update parameter covariance given selected experiment
            for Gt in Gs[worst_exp]:
                Ainv_q -= self.Ainv_prev(Gt, Ainv_q, self.BetaInv)
            print(f"Dropped {worst_exp}")

            # Find next most informative point
            f_I = []
            for treatment in exp_names:
                # compute impact of gaining new point
                # | A + G' B G |
                searchCOV = self.compute_searchCOV(self.Beta, Gs[treatment], Ainv_q)
                f_I.append(self.fast_utility(searchCOV))
            best_exp = design_space[np.argmax(f_I)][0]
            # update parameter covariance given selected experiment
            for Gt in Gs[best_exp]:
                Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
            print(f"Picked {best_exp}")

            # If the dropped point is the same as the added point,
            # or if the same point selected again, stop
            if worst_exp == best_exp or best_exp in best_experiments:
                return best_experiments
            else:
                best_experiments.remove(worst_exp)
                best_experiments.append(best_exp)

        return best_experiments

    # compute utility of each experiment
    def fast_utility(self, searchCOV):
        # predicted objective + log det of prediction covariance over time series
        # searchCOV has shape [n_out, n_out]
        # log eig predCOV has shape [n_out]
        # det predCOV is a scalar
        return jnp.nansum(jnp.log(jnp.linalg.eigvalsh(searchCOV)))
