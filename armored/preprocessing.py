import numpy as np
import pandas as pd
import jax.numpy as jnp

def format_data(df, species, metabolites, controls, observed):
    '''
    Format data so that all experiments have same time length and time steps with
    NaNs filled in missing entries

    df is a dataframe with columns
    ['Experiments', 'Time', 'S_1', ..., 'S_ns', 'M_1', ..., 'M_nm', 'U_1', ..., 'U_nu']

    species := 'S_1', ..., 'S_ns'
    metabolites := 'M_1', ..., 'M_nm'
    controls := 'U_1', ..., 'U_nu'

    '''
    # concatenate all sytem variable names
    sys_vars = np.concatenate((species, metabolites))

    # get experiment names
    experiments = df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, counts = np.unique(experiments, return_counts=True)

    # determine time vector corresponding to longest sampled experiment
    exp_longest = unique_exps[np.argmax(counts)]
    exp_longest_inds = np.in1d(experiments, exp_longest)
    t_eval = df.iloc[exp_longest_inds]['Time'].values
    n_t = len(t_eval)

    # initialize data matrix with NaNs
    X = np.empty([len(unique_exps), len(sys_vars)])
    X[:] = np.nan
    U = np.empty([len(unique_exps), len(t_eval), len(controls)])
    U[:] = np.nan
    Y = np.empty([len(unique_exps), len(t_eval), len(observed)])
    Y[:] = np.nan

    # fill in data for each experiment
    k = 0
    for i,exp in enumerate(unique_exps):
        # isolate particular condition dataframe
        comm_data = df.iloc[k*n_t:(k+1)*n_t].copy()
        k += 1

        # store initial condition data
        X[i] = np.array(comm_data[sys_vars].values, float)[0]

        # store controls and observed data
        U[i] = np.array(comm_data[controls].values, float)
        Y[i] = np.array(comm_data[observed].values, float)

    return jnp.array(X), jnp.array(U), jnp.array(Y), unique_exps

# define scaling functions
class ZeroMaxScaler():

    def __init__(self):
        pass

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.X_min = 0.
        self.X_max = np.nanmax(X, axis=0)
        self.X_range = self.X_max
        self.X_range[self.X_range==0.] = 1.
        return self

    def transform(self, X):
        # convert to 0-1 scale
        X_scaled = X / self.X_range[:,:X.shape[-1]]
        return X_scaled

    def inverse_transform(self, X_scaled):
        X = X_scaled*self.X_range[:,:X_scaled.shape[-1]]
        return X

    def inverse_transform_stdv(self, X_scaled):
        X = X_scaled*self.X_range[:,:X_scaled.shape[-1]]
        return X

    def inverse_transform_cov(self, COV_scaled):
        # transform covariance of scaled values to original scale based on:
        # var ( A x ) = A var (x) A^T
        # where COV_scaled is the var (x) and A is diag( range(x) )
        return np.einsum('tk,ntkl,tl->ntkl',self.X_range[1:], COV_scaled, self.X_range[1:])
