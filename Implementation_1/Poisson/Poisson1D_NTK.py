"""
CS4240: Deep Learning
Reproducbility project
"""

import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torchinfo

### Own modules
### Own modules
import sys
sys.path.insert(0, '../src/')

from PDE import Poisson1D
from PINN import PINN
from plotFunctions import plot_results1D, plot_NTK, plot_param_ntk_diff, plot_NTK_change, plot_convergence_rate

### Set dtype and device to be used
dtype = torch.float32

save_model      = True
train_model     = True
model_name      = 'Poisson1D'

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### Define Poisson1D Exact, forcing function and boundary condition
def f_u_exact(a,x):
    """ 
    Exact solution
    """
    u_exact = torch.sin(a*torch.pi*x)

    return u_exact

def f_x(a, x):
    """
    Source/Forcing function
    """
    fx = -(a**2)*(torch.pi**2)*torch.sin(a*torch.pi*x)
       
    return fx

def g_x(x, xb):
    """
    Boundary condition
    """
    
    ub = torch.zeros(x.size(), dtype=dtype)

    xb1_idx = torch.where(x == xb[0])[0]
    xb2_idx = torch.where(x == xb[1])[0]

    ub[xb1_idx] = 0
    ub[xb2_idx] = 0

    return ub

### Setup PDE Equation
a   = 4
PDE = Poisson1D(a)

# Define PDE domain
X_0,X_N = 0.,1.
X_bc  = [X_0, X_N]

# Number of points
NX  = 100
dx = (X_N - X_0) / NX

# Create points for interior and boundary
Xr = torch.linspace(X_0, X_N, NX, dtype=dtype, device=device, requires_grad=True).view(-1,1)
Xb = torch.randint(0, 2, (NX,1),  dtype=dtype, device=device, requires_grad=True)
X  = torch.hstack((Xr, Xb))

### Setup PINN Network

# Batch size
Br      = 100 
Bb      = 100
rand_sampler = RandomSampler(X, replacement=True)
XTrain       = DataLoader(X, Br ,sampler=rand_sampler)

# Logging parameters
log_NTK            = True
log_parameters     = True

# net parameters
input_size  = 1
output_size = 1
neurons     = [100]
net         = PINN(input_size, output_size, neurons, PDE, 'normal', dtype, device, log_parameters, log_NTK)
net.to(device)

torchinfo.summary(net, input_size=(Br, 1))

# Training parameters
size          = len(XTrain.dataset)
learning_rate = 1e-5
epochs        = int(40e3)

optimizer = optim.SGD(net.parameters(), learning_rate)
# optimizer = optim.Adam(net.parameters(), learning_rate)

##################### Train network

### NTK computation settings
compute_NTK          = True
compute_NTK_interval = 10

### Adapation algorithm
use_adaptation_algorithm = True

### Model save settings
if use_adaptation_algorithm:
    model_adaption = '_adapted'
else:
    model_adaption = ''

file_name = f'{model_name}_{epochs}{model_adaption}'
path      = './output/models/'
pathfile  = path+file_name


#### Train loop
train_losses = []

# Auto Mixed Precision settings
use_amp = False
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)


if train_model:

    for epoch in range(epochs+1):

        if epoch == 0 and compute_NTK:
            ## Observe initial estimation of NTK Matrix
            net.eval()
            x       = next(iter(XTrain)).view(-1, Br, 1)
            x_prime = next(iter(XTrain)).view(-1, Br, 1)

            net.NTK(x, x_prime)
            if log_NTK:
                net.log_NTK(0)
            # reset lambda
            # net.lambda_adaptation = torch.tensor([1., 1.], dtype=dtype, device=device)

        # log parameters and set in training mode
        net.log_parameters(epoch)
        net.train()

        epoch_loss   = 0.0

        for i, x in enumerate(XTrain):
            # reset gradients
            optimizer.zero_grad()

            xr = x[:,0].view(-1,1).to(device); xb = x[:,1].view(-1,1).to(device)

            x = torch.stack([xr, xb], dim=0)

            ### INTERIOR DOMAIN
            # make prediction w.r.t. interior points

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

                ### Predict interior points
                u_hat_x   = net(xr)
            
                # determine gradients w.r.t interior points
                U_x       =  net.compute_pde_gradient(u_hat_x, xr)

                ### BOUNDARY DOMAIN
                u_hat_xb    = net(xb)

                # determine gradients w.r.t boundary points
                U_xb       =  net.compute_pde_gradient(u_hat_xb, xb)
                
                # Compute forcing/source function
                fx = f_x(a, xr).T.to(device)

                # compute boundary condition
                gx = g_x(xb, X_bc).T.to(device)

                # Stack
                U = torch.stack((U_x, U_xb), dim=0)

                ## Backward step
                net.backward(x, U, fx, gx, use_adaption=use_adaptation_algorithm)
                epoch_loss += net.loss.item()
                if i == len(XTrain) - 1:
                    x_prime  = x

            # Do optimisation step
            scaler.scale(net.loss).backward()
            scaler.step(optimizer)
            scaler.update()

        ### END Batch loop

        # Compute NTK
        if epoch > 0:
            if (epoch % compute_NTK_interval == 0 or epoch == epochs - 1) and compute_NTK:

                net.eval()
                net.NTK(x, x_prime)

                if log_NTK:
                    net.log_NTK(epoch)

        train_losses.append(epoch_loss / len(XTrain))
        
        if epoch % 100 == 0 or epoch == epochs: 
            print(f"Epoch: {epoch:4d}     Loss: {train_losses[-1]:4f}   Lr: {optimizer.param_groups[0]['lr']:.2E}")

            if use_adaptation_algorithm:
                lambda_weights = ""
                for lambda_i in net.lambda_adaptation:
                    lambda_weights += f"{lambda_i.item():5f} "
                print(f"                Lambda Adaption: " + lambda_weights)
    ### END training loop

    #### save model
    if save_model:
        net.save_model(pathfile)
        net.save_log(pathfile)

#%% 
### Plot Results

net.eval()
net.read_model(pathfile)
net.read_log(pathfile)

NX = 100

xplot = torch.linspace(X_0, X_N, NX, dtype=dtype).view(-1,1).to(device)

# compute exact solution
u_exact = f_u_exact(a, xplot)
u_pred  = net(xplot)

xplot   = xplot.cpu().detach().numpy()
u_exact = u_exact.cpu().detach().numpy()
u_pred  = u_pred.cpu().detach().numpy()

## Plot 1 - Prediction and training loss
fig1, axs1 = plot_results1D(xplot, u_pred, u_exact, train_losses)
fig1.suptitle(f'Poisson 1D - a = {a} Width = {neurons}')

# Plot 2 - Parameter and ntk difference
fig2, axs2 = plt.subplots(1,2, figsize=(18,6))
plot_param_ntk_diff(net, fig2, axs2)

# Plot 3 - Plot all NTK matrices
fig3, axs3 = plt.subplots(1,3, figsize=(18,6))
plot_NTK(net, fig3, axs3)

# Plot 4 - NTK matrix K change
fig4, axs4 = plt.subplots(1,1)
plot_NTK_change(net, fig4, axs4)

# Plot 5 - Convergence rate for all matrices
fig5, axs5 = plt.subplots(1,1)
plot_convergence_rate(net, fig5, axs5)

plt.show()

