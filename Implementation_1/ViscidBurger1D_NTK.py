"""
CS4240: Deep Learning
Reproducbility project
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torchinfo

### Own modules
from src.PDE import ViscidBurger1D
from src.PINN import PINN
from src.plotFunctions import plot_results2D, plot_NTK, plot_param_ntk_diff, plot_NTK_change, plot_convergence_rate

### Set dtype and device to be used
dtype = torch.float32

train_model     = True
save_model      = True
model_name      = 'ViscidBurger1D'

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### Define Viscid1D Exact, forcing function and boundary condition
def f_u_exact(x, t):
    """ 
    Exact solution
    """
    u_exact = -torch.sin(torch.pi*x)*torch.cos(t)

    return u_exact

def f_x(x):
    """
    Source/Forcing function
    """
    fx = torch.zeros(x.size(), dtype=dtype)
       
    return fx.view(-1,1)

def g_x(x, t, xb, t_ic):
    """
    Boundary condition
    """
    
    ub = torch.zeros(x.size(), dtype=dtype, device=device).view(-1,1)

    # check for boundary condition
    xb1_idx = torch.where(x == xb[0])[0]
    xb2_idx = torch.where(x == xb[1])[0]

    # assert boundary condition
    ub[xb1_idx] = 0
    ub[xb2_idx] = 0

    # check for initial condition
    t_idx = torch.where(t == 0)[0]

    ub[t_idx] = h_x(x[t_idx], t, t_ic)

    return ub.view(-1,1)

def h_x(x, t, t_ic):
    """
    Initial condition
    """
    u_ic = -torch.sin(torch.pi*x)

    return u_ic.view(-1,1)


### Setup PDE Equation
nu  = 0.01/torch.pi
PDE = ViscidBurger1D(nu)

# Define PDE domain
X_0, X_N = -1.,1.
T_0, T_N = 0, 1
t_ic     = T_0

X_bc     = [X_0, X_N]

# Number of points for interior, boundary and inital condition
NX      = int(5000)
Nb = Ni = int(500)

from scipy.stats import qmc
def latin_hypercube(X_0, X_N, N):
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=N).reshape(-1,1)
    sample = qmc.scale(sample, X_0, X_N)

    if dtype == torch.float32:
        sample = sample.astype(np.float32)

    return torch.from_numpy(sample).requires_grad_(True).to(device)

Xr = latin_hypercube(X_0, X_N, NX)
T  = latin_hypercube(T_0, T_N, NX)

Xb = torch.cat( [-1*torch.ones((Nb//2, 1), dtype=dtype), torch.ones((Nb//2, 1), dtype=dtype)] ).to(device).requires_grad_(True)
Tb = latin_hypercube(T_0, T_N, Nb)

Xi = latin_hypercube(X_0, X_N, Ni)
Ti = latin_hypercube(T_0, T_N, Ni)

print(Xb.shape, Tb.shape, Xi.shape, Ti.shape)
X_r     = torch.hstack([Xr, T])
X_bc_ic = torch.hstack([Xb, Tb, Xi, Ti] )

# Dataset preparation

# BATCH SIZES HAVE TO BE SIMILAIR! 
Br = 500
Bb = Bi = 500
rand_sampler1 = RandomSampler(X_r, replacement=True)
XTrain        = DataLoader(X_r, Br ,sampler=rand_sampler1)

rand_sampler2 = RandomSampler(X_bc_ic, replacement=True)
XTrain_bc_ic  = DataLoader(X_bc_ic, Bb, sampler=rand_sampler2)

training_batches    = len(XTrain)
training_batches_xb = len(XTrain_bc_ic)

### NTK computation settings
compute_NTK          = True
compute_NTK_interval = 100

# Logging parameters
log_NTK            = True
log_parameters     = True

# Create network
input_size  = 2
output_size = 1
neurons     = [20, 20, 20, 20, 20, 20, 20, 20]
# neurons     = [100]
net         = PINN(input_size, output_size, neurons, PDE, dtype, device, log_parameters, log_NTK)
net.to(device)

torchinfo.summary(net, input_size=(Br, 2), device=device)

# Training parameters
learning_rate = 1e-3
epochs        = int(10e3)
# optimizer = optim.SGD
optimizer = optim.Adam
# optimizer = optim.LBFGS(net.parameters(), learning_rate)

##################### Train network

### Adapation algorithm
use_adaptation_algorithm = False

# Auto Mixed Precision settings
use_amp = True
if device == torch.device('cpu'):
    use_amp = False
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

### Model save settings
if use_adaptation_algorithm:
    model_adaption = '_adapted'
else:
    model_adaption = ''

if optimizer == optim.SGD:
    opt = 'SGD'
elif optimizer == optim.Adam:
    opt = 'Adam'

file_name = f'{model_name}_Epoch={epochs}_Optimizer={opt}{model_adaption}'
path      = './output/models/'
pathfile  = path+file_name

#### TRAIN MODEL
train_losses = []
optimizer = optimizer(net.parameters(), learning_rate)

if train_model:
    ## Observe initial estimation of NTK Matrix
    if compute_NTK:
        net.eval()

        epoch = 0
        x       = next(iter(XTrain))
        x_prime = next(iter(XTrain))

        x_bc_ic         = next(iter(XTrain_bc_ic))
        x_bc_ic_prime   = next(iter(XTrain_bc_ic))

        xb          = x_bc_ic[:,[0,1]].view(-1,2);          xi          = x_bc_ic[:,[2,3]].view(-1,2)
        xb_prime    = x_bc_ic_prime[:,[0,1]].view(-1,2);    xi_prime    = x_bc_ic_prime[:,[2,3]].view(-1,2)
        
        x.to(device)
        xb.to(device);          xi.to(device)
        xb_prime.to(device);    x_prime.to(device)

        x       = torch.stack([x, xb, xi], dim=0)
        x_prime = torch.stack([x_prime, xb_prime, xi_prime], dim=0)

        net.NTK(x, x_prime)

        if log_NTK:
            net.log_NTK(epoch)

        # reset lambda
        # net.lambda_adaptation = [1., 1., 1.]

    # Training loop
    for epoch in range(epochs+1):

        # log parameters and set in training mode
        if log_parameters:
            net.log_parameters(epoch)

        net.train()

        epoch_loss   = 0.0
        j = 0
        for i, xr in enumerate(XTrain):

            # reset gradients
            optimizer.zero_grad()

            # asymmetrical training sets
            if j == training_batches_xb:           
                j = 0
            if j == 0:
                boundary_ic_dataset = enumerate(XTrain_bc_ic)        
            j, x_bc_ic = next(boundary_ic_dataset)

            # set up training sets
            xr = xr.to(device)
            xb = x_bc_ic[:,[0,1]].view(-1,2).to(device)
            xi = x_bc_ic[:,[2,3]].view(-1,2).to(device)

            x = torch.stack([xr, xb, xi], dim=0)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                ### INTERIOR Domain
                u_hat_x     = net(xr)
                # determine gradients w.r.t interior points
                U_x         =  net.compute_pde_gradient(u_hat_x, xr)

                ### BOUNDARY Domain
                u_hat_xb    = net(xb)
                # determine gradients w.r.t boundary points
                U_xb        =  net.compute_pde_gradient(u_hat_xb, xb)
                
                ### INITIAL condition
                u_hat_xi    = net(xi)
                # determine gradients w.r.t initial condition points
                U_xi        =  net.compute_pde_gradient(u_hat_xi, xi)    
            
                # Compute forcing/source function
                fx = f_x(xr[:,0]).T.to(device)

                # compute boundary condition
                gx = g_x(xb[:,0], xb[:,1], X_bc, t_ic).to(device)

                # compute initial condition
                hx = h_x(xi[:,0], xi[:,1], t_ic).T.to(device)

                # Stack
                U = torch.stack([U_x, U_xb, U_xi], dim=0)

                ## Backward step
                net.backward(x, U, fx, gx, hx, use_adaption=use_adaptation_algorithm)
                epoch_loss += net.loss.item()

                if i == len(XTrain) - 1:
                    x_prime  = x

            # Do optimisation step
            if use_amp:
                scaler.scale(net.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                net.loss.backward()
                optimizer.step()
        ### END Batch loop

        # Compute NTK
        if epoch > 0:
            if (epoch % compute_NTK_interval == 0 or epoch == epochs - 1) and compute_NTK:
                net.eval()
                net.NTK(x, x_prime)

                if log_NTK:
                    net.log_NTK(epoch)
                    net.save_log(pathfile)

        train_losses.append(epoch_loss / len(XTrain))
        
        if epoch % 100 == 0 or epoch == epochs: 
            print(f"Epoch: {epoch:4d}     loss: {train_losses[-1]:5f}")
    ### End training loop

    #### save model
    if save_model:
        net.save_model(pathfile)
        net.save_log(pathfile)
        with open(f'{pathfile}.npy', 'wb') as f:
            np.save(f, np.array(train_losses))

######### Plot results

read_model = True

if read_model:
    net.read_model(pathfile)
    net.read_log(pathfile)
    with open(f'{pathfile}.npy', 'rb') as f:
        train_losses = np.load(f)

path      = './output/figures/'
pathfile  = path+file_name
Path(path).mkdir(parents=True, exist_ok=True)

net.eval()
N = int(1e3) + 1

xplot = torch.linspace(X_0, X_N, N, dtype=dtype, device=device).view(-1,1)
tplot = torch.linspace(T_0, T_N, N, dtype=dtype, device=device).view(-1,1)

u_exact = []
u_pred  = []

# compute prediction solution
for t_i in tplot:
    t_i = torch.ones(xplot.shape, dtype=dtype, device=device)*t_i
    u_pred.append( net(torch.hstack((xplot, t_i))) )
u_pred  = torch.hstack(u_pred)

xplot   = xplot.cpu().detach().numpy()
tplot   = tplot.cpu().detach().numpy()
u_pred  = u_pred.cpu().detach().numpy()

## Plot 1 - Prediction and training loss
xplot = np.hstack([xplot, tplot])
T_idxs = [0.25, 0.5, 0.75]
plot_results2D(xplot, u_pred, T_idxs, train_losses )
plt.savefig(pathfile+'_plot_2D')

if log_NTK and log_parameters:
    # Plot 2 - Parameter and ntk difference
    fig2, axs2 = plt.subplots(1,2, figsize=(18,6))
    plot_param_ntk_diff(net, fig2, axs2)
    plt.savefig(pathfile+'_plot_param_ntk_diff')

    # Plot 3 - Plot all NTK matrices
    fig3, axs3 = plt.subplots(1,4, figsize=(18,6))
    plot_NTK(net, fig3, axs3)
    plt.savefig(pathfile+'_plot_NTK')

    # Plot 4 - NTK matrix K change
    fig4, axs4 = plt.subplots(1,1)
    plot_NTK_change(net, fig4, axs4)
    plt.savefig(pathfile+'_plot_NTK_change')

    # Plot 5 - Convergence rate for all matrices
    fig5, axs5 = plt.subplots(1,1)
    plot_convergence_rate(net, fig5, axs5)
    plt.savefig(pathfile+'_plot_convergence_rate')


plt.show()
