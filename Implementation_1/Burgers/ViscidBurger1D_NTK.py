"""
CS4240: Deep Learning
Reproducbility project
"""

import time
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
import sys
sys.path.insert(0, '../src/')

from PDE import ViscidBurger1D
from PINN import PINN
from plotFunctions import plot_results2D, plot_NTK, plot_param_ntk_diff, plot_NTK_change, plot_convergence_rate

### Set dtype and device to be used
dtype = torch.float32

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# device = torch.device('cpu')


### Define Viscid1D Exact, forcing function and boundary condition
def u_exact():
    """ 
    Exact solution
    """
    data = scipy.io.loadmat('../Data/burgers_shock.mat')

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    u_exact = np.real(data['usol'])

    return x, t, u_exact

def f_x(x,t):
    """
    Source/Forcing function
    """

    fx = torch.zeros( (1, x.shape[0]), dtype=dtype, device=device)
       
    return fx

def g_x(x, t, xb, t_ic):
    """
    Boundary condition
    """
    
    ub = torch.zeros((1, x.shape[0]), dtype=dtype, device=device)

    # check for boundary condition
    xb1_idx = torch.where(x == xb[0])[0]
    xb2_idx = torch.where(x == xb[1])[0]

    # assert boundary condition
    ub[:,xb1_idx] = 0
    ub[:,xb2_idx] = 0

    return ub

def h_x(x, t, t_ic):
    """
    Initial condition
    """
    u_ic = -torch.sin(torch.pi*x)

    return u_ic

from scipy.stats import qmc
def latin_hypercube(X_0, X_N, N):
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=N).reshape(-1,1)
    sample = qmc.scale(sample, X_0, X_N)

    if dtype == torch.float32:
        sample = sample.astype(np.float32)

    sample = torch.from_numpy(sample).requires_grad_(True).to(device)

    return sample

def create_bc_points(N, lb, up):
    Xb = torch.cat( [lb*torch.ones((N//2, 1), dtype=dtype), up*torch.ones((N//2, 1), dtype=dtype)] ).to(device).requires_grad_(True)
    return Xb

### Setup PDE Equation
nu  = 0.01/torch.pi
PDE = ViscidBurger1D(nu)

# Define PDE domain
X_0, X_N = -1., 1.
T_0, T_N =  0,  1.
t_ic     = T_0

X_bc     = [X_0, X_N]

# Number of points for interior, boundary and inital condition
NX = int(2048)
Nb = int(512)
Ni = int(512)

Xr = latin_hypercube(X_0, X_N, NX)
T  = latin_hypercube(T_0, T_N, NX)

Xb = create_bc_points(Nb, X_0, X_N)
Tb = latin_hypercube(T_0, T_N, Nb)

Xi = latin_hypercube(X_0, X_N, Ni)
Ti = torch.zeros((Ni,1), dtype=dtype, requires_grad=True, device=device)

X_r     = torch.hstack([Xr, T])
X_bc_ic = torch.hstack([Xb, Tb, Xi, Ti] )


#### PINN

### Settings

# save model
train_model     = True
save_model      = True
model_name      = 'ViscidBurger1D'

# NTK computation settings
compute_NTK          = True
compute_NTK_interval = 100

# Logging parameters
log_parameters     = True
log_NTK            = True

# Adapation algorithm
use_adaptation_algorithm = False

# correct for coupled parameters
if not compute_NTK:
    use_adaptation_algorithm = False
if not compute_NTK:
    log_NTK = False
if not compute_NTK:
    use_adaptation_algorithm = False    


### Setup PINN

# Dataset preparation

# Batch size
Br = 512
Bb = 512
Bi = 512
rand_sampler1 = RandomSampler(X_r, replacement=True)
XTrain        = DataLoader(X_r, batch_size=Br ,sampler=rand_sampler1)

rand_sampler2 = RandomSampler(X_bc_ic, replacement=True)
XTrain_bc_ic  = DataLoader(X_bc_ic, batch_size=Bb, sampler=rand_sampler2)

training_batches    = len(XTrain)
training_batches_xb = len(XTrain_bc_ic)

print(f"Training batch XTrain: {training_batches} \nTraining batch XTrain_bc: {training_batches_xb}")

input_size  = 2
output_size = 1
neurons     = [24, 24, 24, 24, 24, 24, 24]
# neurons     = [1000]

init_type = 'xavier' # initialisation type for the weights

net         = PINN(input_size, output_size, neurons, PDE, init_type,  dtype, device, log_parameters, log_NTK)
net.to(device)

if dtype == torch.float64:
    net.double()

torchinfo.summary(net, input_size=(Br, 2), dtypes=[dtype], device=device)

# Training parameter
learning_rate = 1e-5
epochs        = int(10e3)
optimizer = optim.SGD
# optimizer = optim.Adam
# optimizer = optim.LBFGS

##################### Train network
if optimizer in [optim.SGD, optim.Adam]:
    optimizer = optimizer(net.parameters(), learning_rate)
elif optimizer == optim.LBFGS:
    optimizer = optimizer(net.parameters(), learning_rate, 
                          max_iter=5, max_eval=5,
                          history_size= 20, 
                          line_search_fn='strong_wolfe')

# Auto Mixed Precision settings
use_amp = True
if optimizer == optim.LBFGS:
    print("Automixed precision not working with LBFGS\n")
    use_amp = False
if device == torch.device('cpu'):
    print("Using CPU")
    use_amp = False

# Use scaler
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.9)

### Model save settings
if use_adaptation_algorithm and compute_NTK:
    model_adaption = '_adapted'
else:
    model_adaption = ''

if isinstance(optimizer, optim.SGD):
    opt = 'SGD'
elif isinstance(optimizer , optim.Adam):
    opt = 'Adam'
elif isinstance(optimizer , optim.LBFGS):
    opt = 'LBFGS'

file_name = f'{model_name}_Epoch={epochs}_Optimizer={opt}{model_adaption}'
path      = './output/models/'
pathfile  = path+file_name

#### Train loop
train_losses = []

if train_model:
    ## Observe initial estimation of NTK Matrix
    if compute_NTK:

        net.eval()
        print("Compute initial NTK estimation\n")

        x       = next(iter(XTrain))
        x_prime = next(iter(XTrain))

        x_bc_ic         = next(iter(XTrain_bc_ic))
        x_bc_ic_prime   = next(iter(XTrain_bc_ic))

        xb          = x_bc_ic[:,[0,1]].view(-1,2);          xi          = x_bc_ic[:,[2,3]].view(-1,2)
        xb_prime    = x_bc_ic_prime[:,[0,1]].view(-1,2);    xi_prime    = x_bc_ic_prime[:,[2,3]].view(-1,2)
        
        x       = [x, xb, xi]
        x_prime = [x_prime, xb_prime, xi_prime]

        net.NTK(x, x_prime)

        if log_NTK:
            net.log_NTK(0)
            plot_NTK(net)
            plt.show()

        max_lr = 2/torch.max(torch.real(net.lambda_K))

        if(learning_rate > max_lr):
            print(f"Learning step greater than max_NTK_lr: 2 / lambda_max, unstable training. Lower learning rate. lr= {learning_rate} max_lr: {max_lr.item()}")
        
        # reset lambda
        # net.lambda_adaptation = torch.Tensor([1., 1., 1.]).to(device)

    ## Training loop
    print("Start training\n")
    start_epoch = time.time()
    for epoch in range(epochs+1):
        net.train()

        # log parameters and set in training mode
        if log_parameters:
            net.log_parameters(epoch)


        global epoch_loss
        epoch_loss = 0.0

        for i, xr in enumerate(XTrain):
            # asymmetrical training sets
            x_bc_ic = next(iter(XTrain_bc_ic))

            # set up training sets
            xb = x_bc_ic[:,[0,1]].view(-1,2)
            xi = x_bc_ic[:,[2,3]].view(-1,2)

            x = [xr, xb, xi]

            if i == len(XTrain) - 1 and epoch % compute_NTK_interval != 0:
                x_prime  = x

            def closure():

                optimizer.zero_grad()
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
                    fx = f_x(xr[:,0], xr[:,1])

                    # compute boundary condition
                    gx = g_x(xb[:,0], xb[:,1], X_bc, t_ic)

                    # compute initial condition
                    hx = h_x(xi[:,0], xi[:,1], t_ic)

                    # Stack
                    U = [U_x, U_xb, U_xi]

                    ## Backward step
                    net.backward(x, U, fx, gx, hx, use_adaption=use_adaptation_algorithm)

                if isinstance(optimizer, optim.LBFGS):
                    net.loss.backward(retain_graph=True)

                return net.loss

            # Do optimisation step
            if isinstance(optimizer, optim.LBFGS):
                net.loss = optimizer.step(closure)
            else:
                closure()
                if use_amp:
                    scaler.scale(net.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    net.loss.backward()
                    optimizer.step()
            
            # compute loss over batch
            epoch_loss += net.loss.item()
        ### END Batch loop
        # scheduler.step()

        # Compute NTK
        if epoch > 0 and epoch < epochs - 1:
            if (epoch % compute_NTK_interval == 0) and compute_NTK:
            
                net.eval()
                net.NTK(x, x_prime)

                max_lr = 2/torch.max(torch.real(net.lambda_K))

                if log_NTK:
                    net.log_NTK(epoch)                    
                    net.save_log(pathfile)

        train_losses.append(epoch_loss / len(XTrain))
        if epoch % 100 == 0 or epoch == epochs: 
            end_epoch = time.time()
            print(f"Epoch: {epoch:4d}     Loss: {train_losses[-1]:5f}   Lr: {optimizer.param_groups[0]['lr']:.2E}       t: {(end_epoch - start_epoch):2f} [s]")

            if use_adaptation_algorithm:
                lambda_weights = ""
                for lambda_i in net.lambda_adaptation:
                    lambda_weights += f"{lambda_i.item():5f} "
                print(f"                Lambda Adaption: " + lambda_weights)
            start_epoch = time.time()


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
Nx = int(256)
Nt = 100

xplot = torch.linspace(X_0, X_N, Nx, dtype=dtype, device=device).view(-1,1)
tplot = torch.linspace(T_0, T_N, Nt, dtype=dtype, device=device).view(-1,1)

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
T_idxs = [0, 0.25, 0.5, 0.75]
plot_results2D(xplot, u_pred, u_exact, T_idxs, train_losses )
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
