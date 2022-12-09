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
    def __init__(self, system, dataframes, C, params, sys_vars, measured_vars, controls = [],
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

        # set compressors (observability functions)
        self.C = C

        # set up data
        self.datasets = []
        for i,(df, measured_var) in enumerate(zip(dataframes, measured_vars)):
            self.datasets.append(process_df(df, sys_vars, measured_var, controls))

        # for additional output messages
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-5

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
        def SSE_next(Y_error, CCTinv, G, Ainv):
            return jnp.einsum('tk, kl, tl', Y_error, CCTinv, Y_error) + jnp.einsum('tki,ij,tlj->', G, Ainv, G)
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

        # init sse and n
        SSE = 0.
        n = 0
        self.N = 0

        # loop over datasets
        for i, (dataset, Ci) in enumerate(zip(self.datasets, self.C)):
            # count number of measured output variables
            n += Ci.shape[0]

            # compute observability precision
            CCTinv = np.linalg.inv(Ci@Ci.T)

            # loop over each sample in dataset
            N = 0
            for t_eval, Y_init, Y_measured, ctrl_params in dataset:

                # count number of observations
                N += len(t_eval[1:]) * np.sum(np.sum(Y_measured, 0) > 0) / Ci.shape[0]

                # run model using current parameters
                if self.A is None:
                    # run ODE on initial condition
                    output = self.runODE(t_eval, Y_init, self.params, ctrl_params)

                    # Determine SSE
                    Y_error = np.einsum("ck,tk->tc", Ci, output) - Y_measured
                    SSE  += np.sum(Y_error[1:]**2)
                    # yCOV += np.einsum('tk,tl->kl', Y_error[1:], Y_error[1:])
                else:
                    # run model using current parameters, output = [n_time, n_sys_vars]
                    output = self.runODEZ(t_eval, Y_init, self.params, ctrl_params)
                    Y_predicted = np.einsum('ck,tk->tc', Ci, output[1:, :self.n_sys_vars])

                    # collect gradients and reshape
                    G = np.reshape(output[1:, self.n_sys_vars:],
                                  [output[1:].shape[0], self.n_sys_vars, self.n_params])

                    # compress model output and gradient
                    G = np.einsum('ck,tki->tci', Ci, G)

                    # Determine SSE
                    Y_error = Y_predicted - Y_measured[1:]
                    SSE  += self.SSE_next(Y_error, CCTinv, G, self.Ainv)
                    # yCOV += self.yCOV_next(Y_error, G, self.Ainv)

            # update sample count
            self.N += N
            SSE /= N

        ### M step: update hyper-parameters ###

        # update target precision
        self.beta = n / (SSE + 2.*self.b)

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
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha/self.beta))

    def objective(self, params):
        # compute residuals
        self.RES = 0.
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * (params-self.prior)**2) / 2.
        # compute gradient of negative log posterior
        self.grad_NLP = self.Alpha*(params-self.prior)

        # compute Hessian, covariance of y, sum of squares error
        self.A = np.diag(self.Alpha)

        for i, (dataset, Ci) in enumerate(zip(self.datasets, self.C)):
            # compute observability precision
            Beta = self.beta*np.linalg.inv(Ci@Ci.T)

            # loop over each sample in dataset
            for t_eval, Y_init, Y_measured, ctrl_params in dataset:

                # run model using current parameters, output = [n_time, n_sys_vars]
                output = self.runODEZ(t_eval, Y_init, params, ctrl_params)
                Y_predicted = np.einsum('ck,tk->tc', Ci, output[1:, :self.n_sys_vars])

                # collect gradients and reshape
                G = np.reshape(output[1:, self.n_sys_vars:],
                              [output[1:].shape[0], self.n_sys_vars, self.n_params])

                # compress model output and gradient
                G = np.einsum('ck,tki->tci', Ci, G)

                # Determine error
                Y_error = Y_predicted - Y_measured[1:]

                # compute Hessian
                self.A += self.A_next(G, Beta)

                # Determine SSE and gradient of SSE
                self.NLP += np.einsum('tk,kl,tl->', Y_error, Beta, Y_error)/2.
                self.RES += np.sum(Y_error)/self.N

                # sum over time and outputs to get gradient w.r.t params
                self.grad_NLP += self.eval_grad_NLP(Y_error, Beta, G)

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
        for i, (dataset, Ci) in enumerate(zip(self.datasets, self.C)):

            # y = f(x) + eps , eps ~ N(0, 1/beta I)
            # p(y) = N(f(x), 1/beta I)
            # g = C@y
            # p(g) = N(C@f(x), 1/beta C@C.T)

            # compute observability precision
            Beta = self.beta*np.linalg.inv(Ci@Ci.T)

            # loop over each sample in dataset
            for t_eval, Y_init, Y_measured, ctrl_params in dataset:

                # run model using current parameters, output = [n_time, n_sys_vars]
                output = self.runODEZ(t_eval, Y_init, self.params, ctrl_params)
                Y_predicted = np.einsum('ck,tk->tc', Ci, output[1:, :self.n_sys_vars])

                # collect gradients and reshape
                G = np.reshape(output[1:, self.n_sys_vars:],
                              [output[1:].shape[0], self.n_sys_vars, self.n_params])

                # compress model output and gradient
                G = np.einsum('ck,tki->tci', Ci, G)

                # compute Hessian
                self.A += self.A_next(G, Beta)

        # Laplace approximation of posterior covariance
        self.A = (self.A + self.A.T)/2.
        self.Ainv = np.linalg.inv(self.A)
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

    def update_evidence(self):
        # compute evidence
        self.evidence = np.sum(np.log(self.Alpha))/2. - \
                        np.sum(np.log(np.linalg.eigvalsh(self.A)))/2. - \
                        self.NLP + self.N*np.log(self.beta)/2.
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
        covariance = 1/self.beta + self.GAinvG(G, self.Ainv)

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

        # measured values for each fidelity
        true = []
        for i, data in enumerate(self.datasets):
            true_vals = []
            for t_eval, Y_init, Y_measured, ctrl_params in data:
                true_vals.append(Y_measured[1:])
            true.append(true_vals)

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
            for i, (data, Ci) in enumerate(zip(self.datasets, self.C)):
                Beta = self.beta*jnp.linalg.inv(Ci@Ci.T)
                preds = []
                for t_eval, Y_init, Y_measured, ctrl_params in data:
                    preds.append(jnp.einsum('ck,tk->tc', Ci, self.runODE(t_eval, Y_init, theta, ctrl_params)[1:, :self.n_sys_vars]))
                sse_val += jnp.sum(jnp.array([jnp.einsum('tk,kl,tl',t-p,Beta,t-p) for t,p in zip(true[i], preds)]))
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
