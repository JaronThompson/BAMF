import numpy as np
import pandas as pd

# from scipy.integrate import odeint, solve_ivp
from torchdiffeq import odeint
from scipy.optimize import minimize

from tqdm import tqdm

import matplotlib.pyplot as plt

# import pytorch libraries to compute gradients
import functorch
import torch
import torch.nn as nn
from torchdiffeq import odeint

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

### Function to process dataframes ###
def process_df(df, species, device):

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = torch.tensor(comm_data['Time'].values, dtype=torch.float32, device=device)

        # pull system data
        Y_measured = torch.tensor(comm_data[species].values, dtype=torch.float32, device=device)

        # append t_eval and Y_measured to data list
        data.append([treatment, t_eval, Y_measured])

    return data

class CRNN:
    def __init__(self, dataframe, species, n_r=2, n_h=10, device='cuda', verbose=True):

        # set hardware
        self.device = device

        # dimensions
        self.n_s = len(species)
        self.n_r = n_r
        self.n_x = self.n_s + self.n_r
        self.n_h = n_h

        # initialize consumer resource parameters

        # death rate
        d = -3.*np.ones(self.n_s)
        # [C]_ij = rate that species j consumes resource i
        C = np.random.uniform(-1., 0., [self.n_r, self.n_s])
        # [P]_ij = rate that species j produces resource i
        P = np.random.uniform(-5., -1., [self.n_r, self.n_s])
        # carrying capacity of resource
        k = np.ones(self.n_r)

        # initialize neural network parameters
        p_std = 1./np.sqrt(self.n_x)
        # map state to hidden dimension
        W1 = p_std*np.random.randn(self.n_h, self.n_x)
        b1 = np.random.randn(self.n_h)
        # map hidden dimension to efficiencies
        p_std = 1./np.sqrt(self.n_h)
        W2 = p_std*np.random.randn(self.n_r+2*self.n_s, self.n_h)
        b2 = np.random.randn(self.n_r+2*self.n_s)

        # initial resource concentration (log)
        self.r0 = np.random.uniform(-2, 0, self.n_r)

        # concatenate parameter initial guess
        self.params = (d, C, P, k, W1, b1, W2, b2)

        # determine shapes of parameters
        self.shapes = []
        self.k_params = []
        self.n_params = 0
        for param in self.params:
            self.shapes.append(param.shape)
            self.k_params.append(self.n_params)
            self.n_params += param.size
        self.k_params.append(self.n_params)

        # set prior so that C is sparse
        r0 = -5.*np.ones(self.n_r)
        C0 = -5.*np.ones([self.n_r, self.n_s])
        P0 = -5.*np.ones([self.n_r, self.n_s])
        W10 = np.zeros_like(W1)
        b10 = np.zeros_like(b1)
        W20 = np.zeros_like(W2)
        b20 = np.zeros_like(b2)

        # concatenate prior
        prior = [r0, d, C0, P0, k, W10, b10, W20, b20]
        self.prior = np.concatenate([p.ravel() for p in prior])

        # set up data
        self.dataset = process_df(dataframe, species, device)

        # for additional output messages
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-4

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        # define neural consumer resource model
        def system(t, x, params):

            # species
            s = x[:self.n_s]

            # resources
            r = torch.exp(x[self.n_s:])

            # compute state
            state = torch.concatenate((s, r))

            # unpack params
            d, Cmax, Pmax, k, W1, b1, W2, b2 = params

            # take exp of strictly positive params
            d = torch.exp(d)
            Cmax = torch.exp(Cmax)
            Pmax = torch.exp(Pmax)
            k = torch.exp(k)

            # map to hidden layer
            h1 = torch.tanh(W1@state + b1)
            h2 = torch.sigmoid(W2@h1 + b2)

            # divide hidden layer into resource availability, species growth efficiency, resource production efficiency
            f = h2[:self.n_r]
            g = h2[self.n_r:self.n_r+self.n_s]
            h = h2[self.n_r+self.n_s:]

            # update consumption matrix according to resource attractiveness
            C = torch.einsum("i,ij->ij", f, Cmax)

            # scaled production rate
            P = torch.einsum("ij,j->ij", Pmax, h)

            # rate of change of species
            dsdt = s*(g*(C.T@r) - d)

            # rate of change of log of resources
            dlrdt = (1. - r/k)*(P@s) - C@s

            return torch.concatenate((dsdt, dlrdt))

        # define augmented system with forward sensitivity equations where
        # dZ/dt = Jx@Z + Jp
        Jx = functorch.jacfwd(system, 1)
        Jp = functorch.jacfwd(system, 2)

        def aug_system(t, aug_x, params):

            # unpack augmented state
            x = aug_x[0]
            Y = aug_x[1]
            Z = aug_x[2:]

            # time derivative of state
            dxdt = system(t, x, params)

            # system jacobian
            Jx_i = Jx(t, x, params)

            # time derivative of grad(state, initial condition)
            dYdt = Jx_i@Y

            # time derivative of parameter sensitivity
            dZdt = [torch.einsum("ij,j...->i...", Jx_i, Z_i) + Jp_i for Z_i, Jp_i in zip(Z, Jp(t, x, params))]

            return (dxdt, dYdt, *dZdt)

        # function to integrate ODE
        self.runODE  = lambda t_eval, x, r0, params: odeint(lambda t, x: system(t, x, params), torch.concatenate((x[0], r0)), t_eval)

        # function to integrate forward sensitivity equations
        Y0 = torch.eye(self.n_x, dtype=torch.float32, device=device)[:, self.n_s:]
        Z0 = [torch.zeros([self.n_x] + list(param.shape), dtype=torch.float32, device=device) for param in self.params]
        self.runODEZ = lambda t_eval, x, r0, params: odeint(lambda t, x: aug_system(t, x, params), (torch.concatenate((x[0], r0)), Y0, *Z0), t_eval)

        # define useful matrix functions
        def GAinvG(G, Linv):
            return np.einsum("tij,kj,kl,tml->tim", G, Linv, Linv, G)
        self.GAinvG = GAinvG

        def yCOV_next(Y_error, G, Linv):
            return np.einsum('tk,tl->kl', Y_error, Y_error) + np.sum(self.GAinvG(G, Linv), 0)
        self.yCOV_next = yCOV_next

        def A_next(G, Beta):
            A_n = np.einsum('tki, kl, tlj->ij', G, Beta, G)
            A_n = (A_n + A_n.T)/2.
            return A_n
        self.A_next = A_next

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@np.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = Ainv_next

        # jit compile inverse Hessian computation step
        def Ainv_prev(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@np.linalg.inv(GAinv@G.T - BetaInv)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_prev = Ainv_prev

        # jit compile function to compute log of determinant of a matrix
        def log_det(A):
            L = np.linalg.cholesky(A)
            return 2*np.sum(np.log(np.diag(L)))
        self.log_det = log_det

        # compute inverse of L where A = LL^T
        def compute_Linv(A):
            return np.linalg.inv(np.linalg.cholesky(A))
        self.compute_Linv = compute_Linv

        # jit compile function to approximate diagonal of covariance
        def Ainv_diag(Linv):
            return np.diag(Linv.T@Linv)
        self.Ainv_diag = Ainv_diag

        def NewtonStep(A, g):
            return np.linalg.solve(A,g)
        self.NewtonStep = NewtonStep

        def eval_grad_NLP(Y_error, Beta, G):
            return np.einsum('tk,kl,tli->i', Y_error, Beta, G)
        self.eval_grad_NLP = eval_grad_NLP

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        return [torch.tensor(np.reshape(params[k1:k2], shape), dtype=torch.float32, device=self.device) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    def fit(self, evidence_tol=1e-3, nlp_tol=None, alpha_0=1e-5, patience=2, max_fails=2, beta=1e-3):
        # estimate parameters using gradient descent
        self.alpha_0 = alpha_0
        self.itr = 0
        passes = 0
        fails = 0
        convergence = torch.inf
        previdence  = -torch.inf

        # scipy minimize works with a numpy vector of parameters
        params = np.concatenate([self.r0]+[p.ravel() for p in self.params])

        # initialize hyper parameters
        self.init_hypers()

        while passes < patience and fails < max_fails:

            # update Alpha and Beta hyper-parameters
            if self.itr>0: self.update_hypers()

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian,
                                hess=self.hessian,
                                x0 = params,
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            # if self.verbose:
            #     print(self.res)
            params = self.res.x
            self.r0 = torch.tensor(params[:self.n_r], dtype=torch.float32, device=self.device)
            self.params = self.reshape(params[self.n_r:])

            # update covariance
            self.update_precision()

            # make sure that precision is positive definite (algorithm 3.3 in Numerical Optimization)
            if np.min(np.diag(self.A)) > 0:
                tau = 0.
            else:
                tau = beta - np.min(np.diag(self.A))

            # increase precision of prior until posterior precision is positive definite
            self.A += tau*np.diag(self.Alpha)
            while True:
                # try taking cholesky decomposition
                try:
                    np.linalg.cholesky(self.A)
                    break
                except:
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
        self.update_covariance()

    def init_hypers(self):

        # count number of samples
        self.N = 0

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:
            k = 0 # number of outputs
            N = 0 # number of samples
            for series in Y_measured.T:
                # check if there is any variation in the series

                if torch.std(series) > 0:
                    # count number of outputs that vary over time
                    k += 1

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
        self.Alpha = self.alpha_0*np.ones(self.n_params+self.n_r)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    # EM algorithm to update hyper-parameters
    def update_hypers(self):
        print("Updating precision...")

        # compute inverse of cholesky decomposed precision matrix
        Linv = self.compute_Linv(self.A)

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, self.r0, self.params)
            output = torch.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = torch.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = torch.concatenate((Y, Z), axis=-1)[1:, :self.n_s, :]

            # Determine SSE
            Y_error = output[1:, :self.n_s] - Y_measured[1:]
            yCOV += self.yCOV_next(Y_error.cpu().numpy(), G.cpu().numpy(), Linv)

        ### M step: update hyper-parameters ###

        # maximize complete data log-likelihood w.r.t. alpha and beta
        Ainv_ii = self.Ainv_diag(Linv)
        params  = torch.concatenate([self.r0]+[p.ravel() for p in self.params]).cpu().numpy()
        self.alpha = self.n_params/(np.sum((params-self.prior)**2) + np.sum(Ainv_ii) + 2.*self.a)
        # self.Alpha = self.alpha*torch.ones(self.n_params)
        self.Alpha = 1./((params-self.prior)**2 + Ainv_ii + 2.*self.a)

        # update output precision
        self.Beta = self.N*np.linalg.inv(yCOV + 2.*self.b*np.eye(self.n_s))
        self.Beta = (self.Beta + self.Beta.T)/2.
        self.BetaInv = np.linalg.inv(self.Beta)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    def objective(self, params):
        # initial resource concentration
        r0 = torch.tensor(params[:self.n_r], dtype=torch.float32, device=self.device)
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha*(params-self.prior)**2) / 2.
        # compute residuals
        self.RES = 0.

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output = torch.nan_to_num(self.runODE(t_eval, Y_measured, r0, params))

            # Determine error
            Y_error = output[1:, :self.n_s] - Y_measured[1:]

            # convert back to numpy
            Y_error = Y_error.cpu().numpy()

            # Determine SSE and gradient of SSE
            self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error)/2.
            self.RES += np.sum(Y_error)/self.n_total

        # return NLP
        return self.NLP

    def jacobian(self, params):
        # initial resource concentration
        r0 = torch.tensor(params[:self.n_r], dtype=torch.float32, device=self.device)

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, r0, params)
            output = torch.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = torch.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = torch.concatenate((Y, Z), axis=-1)[1:, :self.n_s, :]

            # Determine error
            Y_error = output[1:,:self.n_s] - Y_measured[1:]

            # sum over time and outputs to get gradient w.r.t params
            grad_NLP += self.eval_grad_NLP(Y_error.cpu().numpy(), self.Beta, G.cpu().numpy())

        # return gradient of NLP
        return grad_NLP

    def hessian(self, params):
        # initial resource concentration
        r0 = torch.tensor(params[:self.n_r], dtype=torch.float32, device=self.device)

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, r0, params)
            output = torch.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = torch.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = torch.concatenate((Y, Z), axis=-1)[1:, :self.n_s, :]

            # compute Hessian
            self.A += self.A_next(G.cpu().numpy(), self.Beta)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # return Hessian
        return self.A

    def update_precision(self):

        # update parameter covariance matrix given current parameter estimate
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, self.r0, self.params)
            output = torch.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = torch.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = torch.concatenate((Y, Z), axis=-1)[1:, :self.n_s, :]

            # compute Hessian
            self.A += self.A_next(G.cpu().numpy(), self.Beta)

        # Laplace approximation of posterior precision
        self.A = (self.A + self.A.T)/2.

    def update_covariance(self):
        ### Approximate / fast method ###
        self.Linv = self.compute_Linv(self.A)

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

    def predict_point(self, x_test, t_eval):

        # convert to torch tensors
        t_eval = torch.tensor(t_eval, dtype=torch.float32, device=self.device)
        x_test = torch.atleast_2d(torch.tensor(x_test, dtype=torch.float32, device=self.device))

        # make predictions given initial conditions and evaluation times
        output = torch.nan_to_num(self.runODE(t_eval, x_test, self.r0, self.params))

        return output.cpu().numpy()

    # Define function to make predictions on test data
    def predict(self, x_test, t_eval):

        # convert to torch tensors
        t_eval = torch.tensor(t_eval, dtype=torch.float32, device=self.device)
        x_test = torch.atleast_2d(torch.tensor(x_test, dtype=torch.float32, device=self.device))

        # integrate forward sensitivity equations
        xYZ = self.runODEZ(t_eval, x_test, self.r0, self.params)
        Y_predicted = torch.nan_to_num(xYZ[0]).cpu().numpy()
        Y = xYZ[1]
        Z = xYZ[2:]

        # collect gradients and reshape
        Z = torch.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

        # stack gradient matrices
        G = torch.concatenate((Y, Z), axis=-1)

        # calculate covariance of each output (dimension = [steps, outputs])
        BetaInv = np.zeros([self.n_x, self.n_x])
        BetaInv[:self.n_s, :self.n_s] = self.BetaInv
        covariance = BetaInv + self.GAinvG(G.cpu().numpy(), self.Linv)

        # predicted stdv
        get_diag = functorch.vmap(torch.diag, (0,))
        stdv = torch.sqrt(get_diag(torch.tensor(covariance))).cpu().numpy()

        return Y_predicted, stdv, covariance
