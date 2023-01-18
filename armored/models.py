import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from functools import partial
import time

# used for optimizing GP hyper-parameters
from scipy.optimize import minimize

# class that implements standard RNN
class RNN():

    def __init__(self, n_species, n_metabolites, n_controls, n_hidden,
                 alpha_0=1., param_0=1., rng_key=123):

        # set rng key
        rng_key = random.PRNGKey(rng_key)

        # store dimensions
        self.n_species = n_species
        self.n_metabolites = n_metabolites
        self.n_controls = n_controls
        self.n_hidden = n_hidden
        self.alpha_0 = alpha_0
        self.param_0 = param_0

        # determine indeces of species, metabolites and controls
        self.n_out = n_species + n_metabolites
        self.s_inds = np.array([False]*self.n_out)
        self.m_inds = np.array([False]*self.n_out)
        self.s_inds[:n_species] = True
        self.m_inds[n_species:n_species+n_metabolites] = True

        # determine shapes of weights/biases = [Whh,bhh,Wih, Who,bho, h0]
        self.shapes = [[n_hidden, n_hidden], [n_hidden], [n_hidden, n_species+n_metabolites+2*n_controls],
                  [n_species+n_metabolites, n_hidden], [n_species+n_metabolites], [n_hidden]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)

        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1,k2,shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            if len(shape)>1:
                stdv = self.param_0/np.sqrt(np.prod(shape))
            self.params[k1:k2] = random.uniform(rng_key, shape=(k2-k1,), minval=0., maxval=stdv)
        self.Ainv = None
        self.a = 1e-4
        self.b = 1e-5

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0, 0)))

        # jit compile gradient w.r.t. params
        self.G  = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile function to compute observed output
        def Clout(Cl, out):
            return jnp.einsum('ck,ntk->ntc', Cl, out)
        self.Clout = jit(Clout)

        # jut compile Hessian computation step
        def A_next(Gl, Beta):
            return jnp.einsum('ntki,kl,ntlj->ij', Gl, Beta, Gl)
        self.A_next = jit(A_next)

        # jit compile Newton update direction computation
        def NewtonStep(C, G, g, alpha, beta):
            # compute Hessian of negative log posterior density function
            A = jnp.diag(alpha)
            for Cl, Gl in zip(C, G):
                Beta = beta*jnp.linalg.inv(Cl@Cl.T)
                A += self.A_next(Gl, Beta)
            # solve for Newton step direction
            d = jnp.linalg.solve(A, g)
            return d
        self.NewtonStep = jit(NewtonStep)

        # jit compile function to compute observed gradient
        def CGl(Cl, Gl):
            return jnp.einsum('ck,ntki->ntci', Cl, Gl)
        self.CGl = jit(CGl)

        # jit compile function to compute gradient of NLL
        def g(error, Beta, Gl):
            return jnp.einsum('ntk,kl,ntli->i', error, Beta, Gl)
        self.g = jit(g)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = jit(Ainv_next)

        # jit compile measurement covariance computation
        def compute_SSE(errors, G, C, Ainv):
            SSE = 0.
            for error, Cl, Gl in zip(errors, C, G):
                CCTinv = jnp.linalg.inv(Cl@Cl.T)
                SSE += jnp.einsum('ntk,kl,ntl->', error, CCTinv, error) + jnp.einsum('ntki,ij,ntlj->', Gl, Ainv, Gl)
            return SSE
        self.compute_SSE = jit(compute_SSE)

        # jit compile prediction covariance computation
        def compute_predCOV(beta, G, Ainv):
            return 1/beta + jnp.einsum("ntki,ij,ntlj->ntkl", G, Ainv, G)
        self.compute_predCOV = jit(compute_predCOV)

        # jit compile prediction covariance computation
        def epistemic_COV(G, Ainv):
            return jnp.einsum("ntki,ij,ntlj->ntkl", G, Ainv, G)
        self.epistemic_COV = jit(epistemic_COV)

        # jit compile prediction covariance computation
        def compute_searchCOV(Beta, G, Ainv):
            return jnp.eye(Beta.shape[0]) + jnp.einsum("kl,ntli,ij,ntmj->ntkm", Beta, G, Ainv, G)
        self.compute_searchCOV = jit(compute_searchCOV)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Whh,bhh,Wih,Who,bho,h0]
        return [np.reshape(params[k1:k2], shape) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    def forward(self, params, sample, control):
        return self.output(params, sample[self.s_inds], sample[self.m_inds], control)

    def output(self, params, s, m, u):
        # reshape params
        Whh,bhh,Wih,Who,bho,h0 = self.reshape(params)
        params = [Whh,bhh,Wih,Who,bho]

        # define rnn
        rnn_ctrl = partial(self.rnn_cell, params, u)

        # define initial value
        init = (0, h0, s, m)

        # per-example predictions
        carry, out = lax.scan(rnn_ctrl, init, xs=u[1:])
        return out

    # RNN cell
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, h, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        Whh, bhh, Wih, Who, bho = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t], u[t+1]))

        # update hidden vector
        h = nn.leaky_relu(Whh@h + Wih@i + bhh)

        # predict output
        o = Who@h + bho
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t+1, h, s, m), o

    # fit to data
    def fit(self, data, C, lr=1e-2, map_tol=1e-3, evd_tol=1e-3,
            patience=3, max_fails=3):
        passes = 0
        fails  = 0
        # fit until convergence of evidence
        previdence = -np.inf
        evidence_converged = False
        epoch = 0
        best_evidence_params = np.copy(self.params)
        best_params = np.copy(self.params)

        while not evidence_converged:

            # update hyper-parameters
            self.update_hypers(data, C)

            # use Newton descent to determine parameters
            prev_loss = np.inf

            # fit until convergence of NLP
            converged = False
            while not converged:

                # loop through each data set
                loss = np.dot(self.alpha*self.params, self.params)/2.
                errors = []
                for (x, u, y), Cl in zip(data, C):
                    # x is a [n x (n_s+n_m)] matrix
                    # u is a [n x n_t x n_u] matrix
                    # y is a [n x n_t x n_obs] matrix

                    # forward passs
                    outputs = self.forward_batch(self.params, x, u)

                    # compute prediction error
                    error = np.nan_to_num(self.Clout(Cl, outputs) - y[:,1:])
                    errors.append(error)
                    residuals = np.sum(error)/self.n_total

                    # compute convergence of loss function
                    Beta  = self.beta*np.linalg.inv(Cl@Cl.T)
                    loss += self.compute_NLL(error, Beta)

                # compute convergence
                convergence = (prev_loss - loss) / max([1., loss])
                if epoch%1==0:
                    print("Epoch: {}, Loss: {:.5f}, Residuals: {:.5f}, Convergence: {:5f}".format(epoch, loss, residuals, convergence))

                # stop if less than tol
                if abs(convergence) <= map_tol:
                    # set converged to true to break from loop
                    converged = True
                else:
                    # lower learning rate if convergence is negative
                    if convergence < 0:
                        lr /= 2.
                        # re-try with the smaller step
                        self.params = best_params - lr*d
                    else:
                        # update best params
                        best_params = np.copy(self.params)

                        # update previous loss
                        prev_loss = loss

                        # compute gradients
                        G = []
                        g = self.alpha*self.params
                        for (x, u, y), Cl, error in zip(data, C, errors):
                            Beta = self.beta*np.linalg.inv(Cl@Cl.T)
                            G.append(self.CGl(Cl, self.G(self.params, x, u)))
                            g += self.g(error, Beta, G[-1])

                        # determine Newton update direction
                        d = self.NewtonStep(C, G, g, self.alpha, self.beta)

                        # update parameters
                        self.params -= lr*d

                        # update epoch counter
                        epoch += 1

            # Update gradient
            G = [self.CGl(Cl, self.G(self.params, x, u)) for (x,u,y), Cl in zip(data,C)]

            # Update Hessian estimation
            self.A, self.Ainv = self.compute_precision(G, C)

            # compute evidence
            evidence = self.compute_evidence(loss)

            # determine whether evidence is converged
            evidence_convergence = (evidence - previdence) / max([1., abs(evidence)])
            print("\nEpoch: {}, Evidence: {:.5f}, Convergence: {:5f}".format(epoch, evidence, evidence_convergence))

            # stop if less than tol
            if abs(evidence_convergence) <= evd_tol:
                passes += 1
                lr *= 2.
            else:
                if evidence_convergence < 0:
                    # reset :(
                    fails += 1
                    self.params = np.copy(best_evidence_params)
                    G = [self.CGl(Cl, self.G(self.params, x, u)) for (x,u,y), Cl in zip(data,C)]

                    # Update Hessian estimation
                    self.A, self.Ainv = self.compute_precision(G, C)

                    # reset evidence back to what it was
                    evidence = previdence
                    # lower learning rate
                    lr /= 2.
                else:
                    passes = 0
                    # otherwise, update previous evidence value
                    previdence = evidence
                    # update measurement covariance
                    self.SSE = self.compute_SSE(errors, G, C, self.Ainv)
                    # update best evidence parameters
                    best_evidence_params = np.copy(self.params)

            # If the evidence tolerance has been passed enough times, return
            if passes >= patience or fails >= max_fails:
                evidence_converged = True

    # compute loss
    def compute_NLL(self, errors, Beta):
        return np.einsum('ntk,kl,ntl->', errors, Beta, errors)/2.

    # update hyper-parameters alpha and Beta
    def update_hypers(self, data, C):
        if self.Ainv is None:
            SSE = 0.
            self.N = []
            self.n = []
            for (x, u, y), Cl in zip(data, C):
                self.n.append(Cl.shape[0])
                N = 0
                for yi in y:
                    # N = time points * number of measured system variables / shape
                    N += (yi.shape[0]-1) * np.sum(np.sum(yi, 0) > 0) / Cl.shape[0]
                    SSE += np.einsum('tk,tk->', np.nan_to_num(yi), np.nan_to_num(yi))
                self.N.append(N)
                # SSE /= N
            # update alpha
            self.alpha = self.alpha_0*jnp.ones_like(self.params)
            # update Beta
            self.n_total = 0
            for n, N in zip(self.n, self.N):
                self.n_total += n*N
            self.beta = self.n_total / (SSE + 2.*self.b)
        else:
            # update alpha
            self.alpha = 1. / (self.params**2 + np.diag(self.Ainv) + 2.*self.a)
            # update beta
            self.beta = self.n_total / (self.SSE + 2.*self.b)

    # compute Precision and Covariance matrices
    def compute_precision(self, G, C):
        # compute Hessian (precision Matrix)
        A = np.diag(self.alpha)
        for Cl,Gl in zip(C, G):
            Beta = self.beta*np.linalg.inv(Cl@Cl.T)
            A += self.A_next(Gl, Beta)
        A = (A + A.T)/2.

        # compute inverse precision (covariance Matrix)
        Ainv = np.diag(1./self.alpha)
        for Cl,Gl in zip(C,G):
            BetaInv = (Cl@Cl.T)/self.beta
            for Gn in Gl:
                for Gt in Gn:
                    Ainv -= self.Ainv_next(Gt, Ainv, BetaInv)
        Ainv = (Ainv + Ainv.T)/2.
        return A, Ainv

    # compute the log marginal likelihood
    def compute_evidence(self, loss):
        # compute evidence
        Hessian_eigs = np.linalg.eigvalsh(self.A)
        evidence = self.n_total/2*np.nansum(np.log(self.beta)) + \
                   1/2*np.nansum(np.log(self.alpha)) - \
                   1/2*np.nansum(np.log(Hessian_eigs[Hessian_eigs>0])) - loss
        return evidence

    # function to predict metabolites and variance
    def predict(self, X, U, batch_size=512):
        # determine number of samples to search over
        n_samples, n_time, _ = U.shape
        batch_size = min([n_samples, batch_size])

        # init prediction array
        preds = np.zeros([n_samples, n_time, self.n_out])
        stdvs = np.zeros([n_samples, n_time, self.n_out])
        COV = np.zeros([n_samples, n_time, self.n_out, self.n_out])

        # function to get diagonal of a tensor
        get_diag = vmap(vmap(jnp.diag, (0,)), (0,))

        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):

            # keep initial condition
            preds[batch_inds, 0] = np.array(X[batch_inds])

            # make predictions
            preds[batch_inds, 1:] = np.array(nn.relu(self.forward_batch(self.params, X[batch_inds], U[batch_inds])))

            # compute sensitivities
            G = self.G(self.params, X[batch_inds], U[batch_inds])

            # compute covariances
            COV[batch_inds, 1:] = np.array(self.compute_predCOV(self.beta, G, self.Ainv))

            # pull out standard deviations
            stdvs[batch_inds] = np.sqrt(get_diag(COV[batch_inds]))

        return preds, stdvs, COV

    # function to predict metabolites and variance
    def predict_point(self, X, U):
        # make point predictions
        preds = nn.relu(self.forward_batch(self.params, X, U))

        # include known initial conditions
        preds = np.concatenate((X, preds), 1)

        return preds

    # compute utility of each experiment
    def fast_utility(self, predCOV):
        # predicted objective + log det of prediction covariance over time series
        # predCOV has shape [n_time, n_out, n_out]
        # log eig predCOV has shape [n_time, n_out]
        # det predCOV has shape [n_time]
        return jnp.sum(jnp.nansum(jnp.log(jnp.linalg.eigvalsh(predCOV)), -1))

    # return indeces of optimal samples
    def search(self, data, objective, scaler, N,
                    P=None, batch_size=512, explore = 1e-4, max_explore = 1e4):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute profit function (f: R^[n_t, n_o] -> R) in batches
        if P is not None:
            objective_batch = jit(vmap(lambda pred, obj_params: objective(scaler.inverse_transform(pred), obj_params)))
        else:
            objective_batch = jit(vmap(lambda pred, obj_params: objective(scaler.inverse_transform(pred))))
            P = jnp.zeros(n_samples)
        f_P = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            preds = self.predict_point(data[batch_inds])
            f_P.append(objective_batch(preds, P[batch_inds]))
        f_P = jnp.concatenate(f_P).ravel()
        print("Top 5 profit predictions: ", jnp.sort(f_P)[::-1][:5])

        # if explore <= 0, return pure exploitation search
        if explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(f_P)[::-1][:N])

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(f_P).item()]
        print(f"Picked experiment {len(best_experiments)} out of {N}")

        # init and update parameter covariance
        Ainv_q = jnp.copy(self.Ainv)
        Gi = self.Gi(self.params, data[best_experiments[-1]])
        for Gt in Gi:
            Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # define batch function to compute utilities over all samples
        utility_batch = jit(vmap(self.fast_utility))

        # search for new experiments until find N
        while len(best_experiments) < N:

            # compute information acquisition function
            f_I = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                # predCOV has shape [n_samples, n_time, n_out, n_out]
                predCOV = self.compute_searchCOV(self.Beta, self.G(self.params, data[batch_inds]), Ainv_q)
                f_I.append(utility_batch(predCOV))
            f_I = jnp.concatenate(f_I).ravel()

            # select next point
            w = 0.
            while jnp.argmax(f_P + w*f_I) in best_experiments and w < max_explore:
                w += explore
            utilities = f_P + w*f_I
            print("Exploration weight set to: {:.4f}".format(w))
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]

                    # update parameter covariance given selected condition
                    Gi = self.Gi(self.params, data[best_experiments[-1]])
                    for Gt in Gi:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
                    print(f"Picked experiment {len(best_experiments)} out of {N}")

                    # if have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments
                    else:
                        break

                else:
                    # if the same experiment was picked twice at the max exploration rate
                    print("WARNING: Did not select desired number of conditions")
                    return best_experiments

    # return indeces of optimally informative samples
    def explore(self, data, N, batch_size=512):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])
        best_experiments = []

        # init parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # define batch function to compute utilities over all samples
        utility_batch = jit(vmap(self.fast_utility))

        # search for new experiments until find N
        while len(best_experiments) < N:

            # compute information acquisition function
            f_I = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                # predCOV has shape [n_samples, n_time, n_out, n_out]
                predCOV = self.compute_searchCOV(self.Beta, self.G(self.params, data[batch_inds]), Ainv_q)
                f_I.append(utility_batch(predCOV))
            f_I = jnp.concatenate(f_I).ravel()

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(f_I)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    # update parameter covariance given selected condition
                    Gi = self.Gi(self.params, data[best_experiments[-1]])
                    for Gt in Gi:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    break

        return best_experiments

# class that implements microbiome RNN (miRNN) inherits the RNN class
class miRNN(RNN):

    # RNN cell
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, h, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        Whh, bhh, Wih, Who, bho = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t], u[t+1]))

        # update hidden vector
        # h = nn.leaky_relu(Whh@h + Wih@i + bhh)
        h = nn.tanh(Whh@h + Wih@i + bhh)

        # predict output
        zeros_mask = jnp.concatenate((jnp.array(s>0, float), jnp.ones(m.shape)))
        o = zeros_mask*(Who@h + bho)
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t+1, h, s, m), o
