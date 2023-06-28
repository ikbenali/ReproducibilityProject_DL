
"""
DataGenerator

Description: Some datapoint generators for the Burger's equation
Author : 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable
from scipy.stats import qmc

def plot_x(data, ax, plt_label, linecolor='gx'):

    if isinstance(data, np.ndarray):
        xplot = data.copy()
        ax.plot(xplot[:,0], xplot[:,1], linecolor, markersize=3, label=plt_label)
    elif isinstance(data, torch.Tensor):
        xplot = data.detach().numpy()
        ax.plot(xplot[:,0], xplot[:,1], linecolor, markersize=3, label=plt_label)
    elif isinstance(data, Iterable):
        for i, x in enumerate(data):
            if isinstance(x, np.ndarray):
                xplot = x.copy()
            elif isinstance(x, torch.Tensor):
                xplot = x.detach().numpy()

            if i == 0:
                label = plt_label
            else:
                label = None

            if label == None:
                ax.plot(xplot[:,0], xplot[:,1], linecolor, markersize=3)
            else:
                ax.plot(xplot[:,0], xplot[:,1], linecolor, markersize=3, label=plt_label)
    return ax

def plot_dataset(X, X_bc=None, X_ic=None):


    fig, ax  = plt.subplots(1,1)
    
    ax = plot_x(X,    ax, r'$x$', 'go')
    if X_bc != None:
        ax = plot_x(X_bc, ax, r'$x_{bc}$', 'bo')
    if X_ic != None:
        ax = plot_x(X_ic, ax, r'$x_{ic}$', 'ro')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.5, 0, 0.6))

    return fig, ax

###############################################################################
# Data sample types
###############################################################################
def latin_hypercube(X_0, X_N, N, dtype=np.float32):
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=N).reshape(-1,1)
    sample = qmc.scale(sample, X_0, X_N)

    return sample.astype(dtype)

def grid_spaced_points(x0, xn, t0, tn, N, Nb = None, Ni = None, batch_type='grid', dtype=np.float32):

    if dtype == torch.float32:
        dtype = np.float32

    X = np.linspace(x0, xn, N, dtype=dtype)
    T = np.linspace(t0, tn, N, dtype=dtype)

    XX, TT = np.meshgrid(X, T)
    # create 2D array of all points
    lu = np.vstack([XX.ravel(), TT.ravel()]).T

    if batch_type == 'grid':
        bc = np.vstack(  [np.stack([XX[:N //2,0], TT[0:-1:2 ,0]], 1), np.stack([XX[:N //2, -1], TT[0:-1:2 ,0]], 1)])
        ic = np.stack((XX[0,:], TT[0,:]), 1)
    elif batch_type == 'uniform':
        if Nb == None:
            Nb = N
        elif Ni == None:
            Ni = N
        bc_x  = np.concatenate( [x0*np.ones(Nb//2, dtype=dtype), xn*np.ones(Nb//2, dtype=dtype)])
        bc_t  = np.concatenate( [np.linspace(t0, tn, Nb//2, dtype=dtype), np.linspace(t0, tn, Nb//2, dtype=dtype)])

        bc    = np.stack([bc_x, bc_t], 1)

        ic    = np.stack((np.linspace(x0, xn, Ni, dtype=dtype), np.zeros(Ni, dtype=dtype) ),1)
    return [XX, TT], lu, bc, ic


###############################################################################
# Data loaders
###############################################################################

# Prepare datasets
class Loader(torch.utils.data.Sampler):
    def __init__(self, X_coords, N, dim, B, datapoint_type=None, sample_type='random', shuffle=False, single_batch=False, dtype=np.float32):
        self.X_coords   = X_coords  # coords
        self.N          = int(N)    # Number of samples
        self.dim        = dim       # dimension
        self.B          = int(B)    # Batch size
        self.batches    = self.N // B

        # arguments
        self.shuffle    = shuffle
        self.single_batch = single_batch
        if dtype == torch.float32:
            dtype = np.float32
        self.dtype   = dtype

        if sample_type == 'random':
            # Sample once and randomly fetch
            if datapoint_type =='residual':
                ## interior points
                self.data = X_coords[0,:] + (X_coords[1,:] - X_coords[0,:])*np.random.rand(N, dim).astype(np.float32)
            elif datapoint_type == 'bc':
                ## boundary condition
                X_bc1 = X_coords[0,0]*np.ones((N // 2, dim)).astype(np.float32)
                X_bc2 = X_coords[1,0]*np.ones((N // 2, dim)).astype(np.float32)
                # randomize along each axis
                X_bc1[:,-1] = X_coords[0,-1] + (X_coords[1,-1] - X_coords[0,-1])*np.random.rand(N // 2).astype(np.float32)
                X_bc2[:,-1] = X_coords[0,-1] + (X_coords[1,-1] - X_coords[0,-1])*np.random.rand(N // 2).astype(np.float32)
                self.data = np.vstack([X_bc1, X_bc2])
            elif datapoint_type == 'ic':
                ## initial condition
                self.data = X_coords[0,:] + (X_coords[1,:] - X_coords[0,:])*np.random.rand(N, dim).astype(np.float32)
                self.data[:,1] = 0.

        elif sample_type == 'latin':
            if datapoint_type == 'residual':
                self.data  = np.hstack([latin_hypercube(X_coords[0,0], X_coords[1,0], N), latin_hypercube(X_coords[0,1], X_coords[1,1], N)])
            elif datapoint_type == 'bc':
                X_bc1   = np.vstack([X_coords[0,0]*np.ones((N // 2,1), dtype=dtype), X_coords[1,0]*np.ones((N // 2,1), dtype=dtype)] )
                X_bc2   = np.vstack([latin_hypercube(X_coords[0,1], X_coords[1,1], N // 2), latin_hypercube(X_coords[0,1], X_coords[1,1], N//2)])
                self.data = np.hstack([X_bc1, X_bc2])
            elif datapoint_type == 'ic':
                self.data = np.hstack([latin_hypercube(X_coords[0,0], X_coords[1,0], N), np.zeros((N,1), dtype=dtype)])

        self.data = torch.from_numpy(self.data).requires_grad_()

    def __iter__(self):

        dataset = []
        for b in range(self.batches):
            X = self.data.detach().clone()

            # if shuffle is enabled shuffle Dataset starting from 1th axis
            if self.shuffle:
                for i in range(1,self.dim):
                    rand_shuffle = torch.randint(0, self.N, (self.N,))
                    X[:,i] = X[rand_shuffle,i] 
            
                # Select random batch
                rand_batch = torch.randint(0, self.N, (self.B,))
                X = X[rand_batch, :]

            dataset.append(X.requires_grad_())

            # Only give a single batch from the dataset 
            if self.single_batch:
                break
        return iter(dataset)
        
    def __len__(self):
        return self.batches

class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, X_coords, N, B, datapoint_type='residual',  sample_type='latin', dtype=np.float32):
        self.X_coords = X_coords              # Dataset coordinates
        self.N        = int(N)                # Number of samples
        self.dim      = X_coords.shape[1]     # dimension
        self.B        = int(B)                # Batch size
        self.batches  = self.N // self.B

        self.sample_type    = sample_type
        self.datapoint_type = datapoint_type
        if dtype == torch.float32:
            dtype = np.float32
        self.dtype          = dtype


    def __iter__(self):

        dataset = []

        for b in range(self.batches):           

            if self.sample_type == 'latin':
                if self.datapoint_type == 'residual':
                    X  = np.hstack([latin_hypercube(self.X_coords[0,0], self.X_coords[1,0], self.B), latin_hypercube(self.X_coords[0,1], self.X_coords[1,1], self.B)])
                elif self.datapoint_type == 'bc':
                    X_bc1   = np.vstack([self.X_coords[0,0]*np.ones((self.B // 2,1), dtype=self.dtype), self.X_coords[1,0]*np.ones((self.B // 2,1), dtype=self.dtype)] )
                    X_bc2   = np.vstack([latin_hypercube(self.X_coords[0,1], self.X_coords[1,1], self.B // 2), latin_hypercube(self.X_coords[0,1], self.X_coords[1,1], self.B//2)])
                    X       = np.hstack([X_bc1, X_bc2])
                elif self.datapoint_type == 'ic':
                    X = np.hstack([latin_hypercube(self.X_coords[0,0], self.X_coords[1,0], self.B), np.zeros((self.B,1), dtype=self.dtype)])
            elif self.sample_type == 'random':
                # Sample once and randomly fetch
                if self.datapoint_type =='residual':
                    ## interior points
                    X = self.X_coords[0,:] + (self.X_coords[1,:] - self.X_coords[0,:])*np.random.rand(self.N, self.dim).astype(np.float32)
                elif self.datapoint_type == 'bc':
                    ## boundary condition
                    X_bc1 = self.X_coords[0,0]*np.ones((self.N // 2, self.dim)).astype(np.float32)
                    X_bc2 = self.X_coords[1,0]*np.ones((self.N // 2, self.dim)).astype(np.float32)
                    # randomize along each axis
                    X_bc1[:,-1] = self.X_coords[0,-1] + (self.X_coords[1,-1] - self.X_coords[0,-1])*np.random.rand(self.N // 2).astype(np.float32)
                    X_bc2[:,-1] = self.X_coords[0,-1] + (self.X_coords[1,-1] - self.X_coords[0,-1])*np.random.rand(self.N // 2).astype(np.float32)
                    X = np.vstack([X_bc1, X_bc2])
                elif self.datapoint_type == 'ic':
                    ## initial condition
                    X = self.X_coords[0,:] + (self.X_coords[1,:] - self.X_coords[0,:])*np.random.rand(self.N, self.dim).astype(np.float32)
                    X[:,1] = 0.
            
            X = torch.from_numpy(X).requires_grad_()
            dataset.append(X)

        return iter(dataset)
        
    def __len__(self) -> int:
        return self.batches
