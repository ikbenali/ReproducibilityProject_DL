"""
CS4240: Deep Learning
Reproducbility project
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

### Own modules
from src.PDE import Poisson1D
from src.PINN import PINN
from src.NTK import NTK
from src.plotFunctions import plot_NTK

### Set dtype and device to be used
dtype = torch.float32

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
a   = 2
PDE = Poisson1D(a)

# Define PDE domain
X_0,X_N = 0.,1.
X_bc  = [X_0, X_N]

# Number of points
NX  = 500
dx = (X_N - X_0) / NX

# Create points for interior and boundary
Xr = torch.linspace(X_0, X_N, NX, dtype=dtype, device=device, requires_grad=True).view(-1,1)
Xb = torch.randint(0, 2, (NX,1),  dtype=dtype, device=device, requires_grad=True)
X  = torch.hstack((Xr, Xb))

### Setup PINN Network
Nr      = 100
Nb      = 100
rand_sampler = RandomSampler(X, replacement=True)
XTrain       = DataLoader(X, Nr ,sampler=rand_sampler)

# Logging parameters
log_NTK            = True
log_parameters     = True

# net parameters
input_size  = 1
output_size = 1
neurons     = 10
net         = PINN(input_size, output_size, neurons, PDE, dtype, device, log_parameters, log_NTK);
net.to(device)

# Training parameters
size          = len(XTrain.dataset)
learning_rate = 1e-3
epochs        = int(10000)

loss_fn   = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), learning_rate)
# optimizer = optim.Adam(net.parameters(), learning_rate)

##################### Train network

### NTK computation settings
compute_NTK          = True
compute_NTK_interval = 10

## Observe initial estimation of NTK Matrix
net.eval()
x       = next(iter(XTrain))
x_prime = next(iter(XTrain))
net.NTK(x, x_prime)
net.log_NTK(0)

# Plot initial 
plot_NTK(net)
plt.show()

#### Train loop
train_losses = []

# Auto Mixed Precision settings
use_amp = False
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs+1):

    net.log_parameters(epoch)
    net.train()

    epoch_loss   = 0.0

    for i, x in enumerate(XTrain):
        # reset gradients
        optimizer.zero_grad()

        xr = x[:,0].view(-1,1).to(device); xb = x[:,1].view(-1,1).to(device)

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
            net.backward(x, U, fx, gx)
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
    
    if epoch % 100 == 0 or epoch == epochs - 1: 
        print(f"Epoch: {epoch:4d}     loss: {train_losses[-1]:5f}")
### END training loop


#%% 
### Plot Results

net.eval()

xplot = torch.linspace(X_0, X_N, NX, dtype=dtype).view(-1,1).to(device)

# compute exact solution
u_exact = f_u_exact(a, xplot)
u_pred  = net(xplot)

xplot   = xplot.cpu().detach().numpy()
u_exact = u_exact.cpu().detach().numpy()
u_pred  = u_pred.cpu().detach().numpy()

pointWise_err = u_exact - u_pred

print(pointWise_err.shape)

### PLOT Prediction and training loss

fig, axs = plt.subplots(1,2, figsize=(24,6))

# predict and error plot
axs[0].plot(xplot, u_exact, label='$u_{exact}$')
axs[0].plot(xplot, u_pred, label='$u_{pred}$')
axs[0].legend()
axs[0].set_ylabel(r'$u$')
axs[0].set_xlabel(r'$x$')

axs[1].plot(xplot, pointWise_err)
axs[1].set_ylabel('Point-wise error')
axs[1].set_xlabel(r'$x$')

# train losses plot
fig,axs = plt.subplots(1,1, figsize=(12,6))
axs.semilogy(train_losses)
axs.set_ylabel(r'loss per epoch')
axs.set_xlabel(r'$Epoch$')

plt.show()

# %%
