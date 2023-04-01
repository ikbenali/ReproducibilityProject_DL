
#%%

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt

import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler


### Own modules
from src.PDE import Poisson1D
from src.PINN import PINN
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


###### Setup PINN 

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

# net parameters
input_size  = 1
output_size = 1
neurons     = 100

# Training parameters
size          = len(XTrain.dataset)
learning_rate = 1e-5
epochs        = int(10e3)

###### RESULT 1

### Setup PDE Equation
a   = [1,2,4]
PDE = [Poisson1D(a_i) for a_i in a]
neural_nets =  [PINN(input_size, output_size, neurons, PDE_i, dtype, device) for PDE_i in PDE]; 

for net_i in neural_nets:
    net_i.to(device)
    net_i.eval()

### Observe initial estimation of NTK Matrix

x       = next(iter(XTrain))
x_prime = next(iter(XTrain))

### PLOT Eigenvalue of NTK matrices
fig, axs = plt.subplots(1,3, figsize=(23,6))

ylabels = [r'$\lambda_{K}$', r'$\lambda_{uu}$', r'$\lambda_{rr}$']

for i in range(len(a)):
    neural_nets[i].NTK(x, x_prime)

    eig_K_plot    = np.sort(np.real(neural_nets[i].lambda_K.detach().cpu().numpy()))[::-1]
    eig_K_uu_plot = np.sort(np.real(neural_nets[i].lambda_Kuu.detach().cpu().numpy()))[::-1]
    eig_K_rr_plot = np.sort(np.real(neural_nets[i].lambda_Krr.detach().cpu().numpy()))[::-1]

    axs[0].semilogx(eig_K_plot,      label=f'a={a[i]}');    axs[0].set_title('Eigenvalue of K')
    axs[1].semilogx(eig_K_uu_plot,   label=f'a={a[i]}');    axs[1].set_title('Eigenvalue of {}'.format(r"$K_{uu}$"))
    axs[2].semilogx(eig_K_rr_plot,   label=f'a={a[i]}');    axs[2].set_title('Eigenvalue of {}'.format(r"$K_{rr}$"))

    for ax in axs:
        ax.legend()
        # ax.ticklabel_format(axis='y', scilimits=(0,0))
        ax.set_yscale('log')
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel(r'$Index$')

plt.show()

###### RESULT 2
#%%
neurons = [10, 100, 500]
a       = 4
PDE     = Poisson1D(a)

neural_nets  = [PINN(input_size, output_size, neurons_i, PDE, dtype, device, log_parameters=True, log_NTK=True) for neurons_i in neurons];

x       = next(iter(XTrain))
x_prime = next(iter(XTrain))

for net in neural_nets:
    net.to(device)
    net.NTK(x, x_prime)
    net.log_NTK(0)

##################### Train network

optimizers = [optim.SGD(net_i.parameters(), learning_rate) for net_i in neural_nets]

### NTK computation settings
compute_NTK          = True
compute_NTK_interval = 100
log_NTK              = True
eigenvalues_NTK_log  = {}

#### Train loop
train_losses = []
log_parameters = True

# Auto Mixed Precision settings
use_amp = True
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs+1):
    epoch_loss   = 0.0
    for i, x in enumerate(XTrain):

        xr = x[:,0].view(-1,1).to(device); xb = x[:,1].view(-1,1).to(device)

        ### INTERIOR DOMAIN
        # make prediction w.r.t. interior points

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

            for i,net in enumerate(neural_nets):
                net.log_parameters(epoch)
                net.train()

                # reset gradients  
                optimizers[i].zero_grad()

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
                scaler.step(optimizers[i])
                scaler.update()

    ### END Batch loop

    # Compute NTK
    if epoch > 0:
        if (epoch % compute_NTK_interval == 0 or epoch == epochs - 1) and compute_NTK:

            for net in neural_nets:
                net.eval()
                net.NTK(x, x_prime)
                if log_NTK:
                    net.log_NTK(epoch)
    
    train_losses.append(epoch_loss / len(XTrain))
    
    if epoch % 100 == 0 or epoch == epochs - 1: 
        print(f"Epoch: {epoch:4d}     loss: {train_losses[-1]:5f}")
        
### END training loop

##### Plot results
#%%

def compute_norm(matrix):
    norm = torch.sqrt(torch.sum(matrix**2))

    return norm

def compute_parameter_diff(net): 

    parameter_epochs = list(net.network_parameters_log.keys())
    parameters_diff = [0.0]

    initial_params = net.network_parameters_log[0]

    for epoch in parameter_epochs[1:]:
        diff = torch.tensor([0.0], dtype=dtype, device=device)
        params = net.network_parameters_log[epoch]
        for layer_weight, init_layer_weight in zip(params['weight'], initial_params['weight']):
            diff += compute_norm(layer_weight - init_layer_weight) / compute_norm(init_layer_weight)
        for layer_bias, init_layer_bias in zip(params['bias'], initial_params['bias']):
            diff += compute_norm(layer_bias - init_layer_bias) / compute_norm(init_layer_bias)

        parameters_diff.append(diff.item())

    return parameter_epochs, parameters_diff

def compute_NTK_diff(net): 

    NTK_epochs = list(net.NTK_log.keys())
    NTK_diff = [0.0]

    K0 = net.NTK_log[0]['NTK_matrix'][0]

    for epoch in NTK_epochs[1:]:
        K   = net.NTK_log[epoch]['NTK_matrix'][0]
        diff = torch.linalg.matrix_norm(K - K0, ord=2) / torch.linalg.matrix_norm(K0, ord=2)
        NTK_diff.append(diff.item())

    return NTK_epochs, NTK_diff


# Plot 1
fig, axs = plt.subplots(1,2, figsize=(18,6))
 
for i in range(len(neurons)):

    parameter_epoch, parameter_diff = compute_parameter_diff(neural_nets[i])
    NTK_epoch, NTK_diff             = compute_NTK_diff(neural_nets[i])

    axs[0].plot(parameter_epoch,parameter_diff, label=f'width={neurons[i]}');    
    axs[1].plot(NTK_epoch, NTK_diff,            label=f'width={neurons[i]}');   

    for ax in axs:
        # ax.ticklabel_format(axis='y', scilimits=(0,0))
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel(r'$Epoch$')
axs[0].set_ylabel(r'$\frac{||\theta - \theta(0)||^{2}}{||\theta(0)||^{2}}$')
axs[1].set_ylabel(r'$\frac{||K(n) - K(0)||^{2}}{||K(0)||^{2}}$')


fig, axs = plt.subplots(1,1)

net = neural_nets[-1]
NTK_epochs = list(net.NTK_log.keys())

for epoch in NTK_epochs:
    if epoch == 0:
        eig_K = net.NTK_log[epoch]['NTK_eigenvalues'][0]
        eig_K_plot    = np.sort(np.real(eig_K.detach().cpu().numpy()))[::-1]
        axs.semilogy(eig_K_plot,   label=f'epoch={epoch}'); 
    elif epoch == NTK_epochs[-1]:
        eig_K = net.NTK_log[epoch]['NTK_eigenvalues'][0]
        eig_K_plot    = np.sort(np.real(eig_K.detach().cpu().numpy()))[::-1]
        axs.semilogy(eig_K_plot,   label=f'epoch={epoch}'); 

axs.set_xscale('log')
axs.set_xlabel(r'$Epoch$')
axs.legend()

plt.show()

# %%
