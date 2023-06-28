"""
CS4240: Deep Learning
Reproducbility project
"""

import time
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchinfo

### Import own modules
import sys
from pathlib import Path
sys.path.insert(0, '../src/')

from PDE            import ViscidBurger1D
from PINN           import PINN
from plotFunctions  import plot_results2D, plot_results2D_animate, plot_NTK, plot_param_ntk_diff, plot_NTK_change, plot_convergence_rate
from DataGenerator  import Loader, RandomSampler, grid_spaced_points, plot_dataset
from NTK_helper     import compute_adaptionWeights, NTK_scheduler

### Set dtype and device to be used
dtype = torch.float32

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# save model
train_model     = True
save_model      = True
model_name      = 'ViscidBurger1D'

###############################################################################
# 1. Define Viscid1D Exact, forcing function and boundary condition
###############################################################################

def u_exact():
    """ 
    Exact solution
    """
    data = scipy.io.loadmat('../Data/burgers_shock.mat')

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    u_exact = np.real(data['usol'])

    return x, t, u_exact

def f_x(x,t, nu):
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


###############################################################################
# 2. Setup Data
###############################################################################

# Define PDE domain
X_0, X_N = -1., 1.
T_0, T_N =  0,  1.
t_ic     = T_0

# set up domain for the residual, boundary condition and initial condition
X_coords    = np.array([[X_0, T_0], [X_N, T_N]], dtype=np.float32)

### Number of points for interior, boundary and inital condition
Nr = int(1200) # Grid: Nr x Nr Points , Latin: Nr Points
Nb = int(80) 
Ni = int(120)

### Batch size
Br = 400
Bb = 80
Bi = 120

sample_type = 'random'

# data generation settings
shuffleData    = True
normalizeInput = True
doSingleBatch  = False

# Create a new random batch every epoch
randomBatch = False

if sample_type == 'random':
    if randomBatch:
        # Loader that creates a random batch each time
        XTrain      = RandomSampler(X_coords, Nr, Br, sample_type=sample_type, datapoint_type='residual', dtype=dtype)
        XTrain_bc   = RandomSampler(X_coords, Nb, Bb, sample_type=sample_type, datapoint_type='bc', dtype=dtype)
        XTrain_ic   = RandomSampler(X_coords, Ni, Bi, sample_type=sample_type, datapoint_type='ic', dtype=dtype)
    else:
        # Loader that generates all samples at once at randomly selects each batch from the set of samples
        XTrain      = Loader(X_coords, Nr, 2, Br, 'residual',  sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)
        XTrain_bc   = Loader(X_coords, Nb, 2, Bb, 'bc',        sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)
        XTrain_ic   = Loader(X_coords, Ni, 2, Bi, 'ic',        sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)
    
    Lu_points = Nr;             training_batches    = Lu_points / Br 
    bc_points = Nb;             training_batches_bc = bc_points / Bb
    ic_points = Ni;             training_batches_ic = ic_points / Bi

elif sample_type == 'latin':
    if randomBatch:
        # Loader that creates a random batch each time
        XTrain      = RandomSampler(X_coords, Nr, Br, sample_type=sample_type, datapoint_type='residual', dtype=dtype)
        XTrain_bc   = RandomSampler(X_coords, Nb, Bb, sample_type=sample_type, datapoint_type='bc', dtype=dtype)
        XTrain_ic   = RandomSampler(X_coords, Ni, Bi, sample_type=sample_type, datapoint_type='ic', dtype=dtype)
    else:
        # Loader that generates all samples at once at randomly selects each batch from the set of samples
        XTrain      = Loader(X_coords, Nr, 2, Br, 'residual',  sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)
        XTrain_bc   = Loader(X_coords, Nb, 2, Bb, 'bc',        sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)
        XTrain_ic   = Loader(X_coords, Ni, 2, Bi, 'ic',        sample_type=sample_type, shuffle=shuffleData, single_batch=doSingleBatch, dtype=dtype)

    Lu_points = Nr;             training_batches    = Lu_points / Br 
    bc_points = Nb;             training_batches_bc = bc_points / Bb
    ic_points = Ni;             training_batches_ic = ic_points / Bi

elif sample_type == 'grid':
    _, XTrain, XTrain_bc, XTrain_ic = grid_spaced_points(X_0, X_N, T_0, T_N, Nr, Nb, Ni, 'uniform', dtype=dtype)

    XTrain    = torch.from_numpy(XTrain).requires_grad_()
    XTrain_bc = torch.from_numpy(XTrain_bc).requires_grad_()
    XTrain_ic = torch.from_numpy(XTrain_ic).requires_grad_()

    Lu_points = XTrain.shape[0];                training_batches    = Lu_points / Br 
    bc_points = XTrain_bc.shape[0];             training_batches_bc = bc_points / Bb
    ic_points = XTrain_ic.shape[0];             training_batches_ic = ic_points / Bi

    # Create dataloader
    XTrain      = DataLoader(XTrain,    Br, shuffle=shuffleData)
    XTrain_bc   = DataLoader(XTrain_bc, Bb, shuffle=shuffleData)
    XTrain_ic   = DataLoader(XTrain_ic, Bi, shuffle=shuffleData)

print(f"Lu points: {Lu_points}    bc points: {bc_points}     ic points: {ic_points}")
print(f"Training batch XTrain: {training_batches} \nTraining batch XTrain_bc: {training_batches_bc}\nTraining batch XTrain_ic: {training_batches_ic}")

# Plot datapoints
plot_dataset(XTrain, XTrain_bc, XTrain_ic)
plt.show()

###############################################################################
# 3. Setup PINN
###############################################################################

# PINN and train Settings

log_param_interval = 100

# NTK computation settings
compute_NTK          = True
compute_NTK_interval = 1000

# Logging parameters
log_parameters     = True
log_NTK            = True

# Adapation algorithm
use_adaptation_algorithm = True
adapt_lr                 = False

# correct for coupled parameters
if not compute_NTK:
    log_NTK = False
    use_adaptation_algorithm = False

# Setup PDE Equation
nu  = 0.01/torch.pi
PDE = ViscidBurger1D(nu)

# Configure network
input_size  = 2
output_size = 1
# neurons = [400, 400, 400, 400, 400]
neurons = [320, 320, 320, 320, 320]

init_type = 'normal' # initialisation type for the weights
net       = PINN(input_size, output_size, neurons, PDE, init_type,  dtype, device, log_parameters, log_NTK)
net.to(device)

if normalizeInput:
    net._init_normalization(X_coords)

torchinfo.summary(net, input_size=(Br, 2), dtypes=[dtype], device=device)

###############################################################################
# 4. Setup PINN Training
###############################################################################

# Training configuration
learning_rate = 1e-5
epochs        = int(45e3)

optimizer = optim.SGD
optimizer = optim.Adam

optimizer = optimizer(net.parameters(), learning_rate)

# Auto Mixed Precision settings
use_amp = True
if device == torch.device('cpu'):
    print("Using CPU")
    use_amp = False

# Use scaler
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

# Model save settings
if use_adaptation_algorithm and compute_NTK:
    model_adaption = '_adapted'
else:
    model_adaption = ''

if isinstance(optimizer, optim.SGD):
    opt = 'SGD'
elif isinstance(optimizer , optim.Adam):
    opt = 'Adam'

uneven_batches = ''
if Br != Bb or Br != Bi:
    uneven_batches = '_unevenBatches'

file_name   = f'{model_name+model_adaption+uneven_batches}'
path      = f'./output/models/{model_name}/{epochs}/{opt}/'
pathfile  = path+file_name
Path(path).mkdir(parents=True, exist_ok=True)

###############################################################################
# 5. PINN Training
###############################################################################

# Train loop
train_losses = []
x            = [] 
x_prime      = []
adaption_weights = []

if train_model:

    ## Observe initial estimation of NTK Matrix
    if compute_NTK:

        net.eval()
        print("Compute initial NTK estimation\n")

        x       = next(iter(XTrain)).to(device)
        x_prime = next(iter(XTrain)).to(device)

        x_bc         = next(iter(XTrain_bc)).to(device)
        x_bc_prime   = next(iter(XTrain_bc)).to(device)

        x_ic         = next(iter(XTrain_ic)).to(device)
        x_ic_prime   = next(iter(XTrain_ic)).to(device)

        x       = [x, x_bc, x_ic]
        x_prime = [x_prime, x_bc_prime, x_ic_prime]

        net.NTK(x, x_prime)

        if log_NTK:
            net.log_NTK(0)
            # plot_NTK(net)
            # plt.show()
        
        if use_adaptation_algorithm and adapt_lr:
            NTK_scheduler(net, optimizer)
            adaption_weights.append(net.lambda_adaptation.cpu().numpy())

    ## Training loop
    print("Start training\n")
    start_epoch = time.time()
    for epoch in range(epochs+1):

        # log parameters and set in training mode
        if log_parameters and (epoch % log_param_interval) == 0:
            net.log_parameters(epoch)
        net.train()

        # Reset gradients
        optimizer.zero_grad()

        # Epoch loss
        epoch_loss = []
        train_loss_xr = 0.0

        # Step 1. Interior points
        for i, xr in enumerate(XTrain):
            xr = xr.to(device);
            
            if len(x_prime)  == 0 and epoch % compute_NTK_interval != 0:
                x_prime.append(xr)
            elif len(x) == 0:
                x.append(xr)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

                ### INTERIOR Domain
                u_hat_x     = net(xr)
                # determine gradients w.r.t interior points
                U_x         =  net.compute_pde_gradient(u_hat_x, xr)

                # Compute forcing/source function
                fx = f_x(xr[:,0], xr[:,1], nu)
                
                # backward pass
                net.backward(xr, U_x, f=fx, use_adaption=use_adaptation_algorithm)

                # accumulating loss
                if doSingleBatch:
                    train_loss_xr += net.train_loss 
                else:
                    train_loss_xr += net.train_loss / training_batches

        # add loss over interior points
        epoch_loss.append(train_loss_xr)

        # Step 2. Boundary condition
        train_loss_bc = 0.0
        for i, x_bc in enumerate(XTrain_bc):
            # set up training sets
            x_bc = x_bc.to(device)

            if len(x_prime) == 1 and epoch % compute_NTK_interval != 0:
                x_prime.append(x_bc)
            elif len(x) == 1:
                x.append(x_bc)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

                ### BOUNDARY Domain
                u_hat_xb    = net(x_bc)
                # determine gradients w.r.t boundary points
                U_xb        =  net.compute_pde_gradient(u_hat_xb, x_bc)

                # compute boundary condition
                gx = g_x(x_bc[:,0], x_bc[:,1], X_coords[:,0], t_ic)

                # Compute backward
                net.backward(x_bc, U_xb, g=gx, use_adaption=use_adaptation_algorithm)

                # accumulating loss
                if doSingleBatch:
                    train_loss_bc  += net.train_loss
                else:
                    train_loss_bc  += net.train_loss / training_batches_bc

        # add loss over boundary points
        epoch_loss.append(train_loss_bc)

        # Step 3. Compute initial condition
        train_loss_ic = 0.0
        for i, x_ic in enumerate(XTrain_ic):
            # set up training sets
            x_ic = x_ic.to(device)

            if len(x_prime) == 2 and epoch % compute_NTK_interval != 0:
                x_prime.append(x_ic)
            elif len(x) == 2:
                x.append(x_ic)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

                ### Initial condition Domain
                u_hat_xi    = net(x_ic)
                # determine gradients w.r.t initial condition points
                U_xi        =  net.compute_pde_gradient(u_hat_xi, x_ic)

                # compute initial condition
                hx = h_x(x_ic[:,0], x_ic[:,1], t_ic)

                # Compute backward
                net.backward(x_ic, U_xi, h=hx, use_adaption=use_adaptation_algorithm)

                # accumulating loss
                if doSingleBatch:
                    train_loss_ic  += net.train_loss
                else:
                    train_loss_ic  += net.train_loss / training_batches_ic

        # add loss over initial condition points
        epoch_loss.append(train_loss_ic)

        # Step 4. Backward gradient and Optimization step

        # Convert to tensor
        epoch_loss = torch.hstack(epoch_loss)
        loss       = epoch_loss.sum(0)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Stack respective losses in this Epoch
        train_losses.append(epoch_loss.detach().cpu().numpy())

        # Step 5. Compute NTK
        if epoch > 0:
            if (epoch % compute_NTK_interval == 0) and compute_NTK:

                net.eval()

                net.NTK(x, x_prime)

                # reset datapoints
                x          = []
                x_prime    = []

                # log
                if log_NTK:
                    net.log_NTK(epoch)
                    if save_model:                   
                        net.save_log(pathfile)
                if use_adaptation_algorithm:
                    adaption_weights.append(net.lambda_adaptation.cpu().numpy())
                # adapt lr
                if adapt_lr:
                    NTK_scheduler(net, optimizer)
        # Print epoch
        if epoch % 100 == 0 or epoch == epochs: 
            end_epoch = time.time()
            loss = [f'{loss_i:.3E}' for loss_i in train_losses[-1]] 
            print(f"Epoch: {epoch:4d}     t: {(end_epoch - start_epoch):2f} [s]")
            print(f"                Loss_du: {train_losses[-1][0]:.3E}        Loss_bc: {train_losses[-1][1]:.3E}     Loss_ic: {train_losses[-1][2]:.3E}\n                Total loss: {train_losses[-1].sum().item():.5E}   Lr: {optimizer.param_groups[0]['lr']:.3E} ") 
            if use_adaptation_algorithm:
                lambda_weights = ""
                for lambda_i in net.lambda_adaptation:
                    lambda_weights += f"{lambda_i.item():5f} "
                print(f"                Lambda Adaption: " + lambda_weights)
            start_epoch = time.time()
    # End training loop

    # save model
    if save_model:
        net.save_model(pathfile)
        net.save_log(pathfile)
        with open(f'{pathfile}.npy', 'wb') as f:
            np.save(f, np.array(train_losses))

        if use_adaptation_algorithm:
            adaption_weights = np.vstack(adaption_weights)

###############################################################################
# 6. Plot results Training
###############################################################################

# Read Data
read_model  = True
file_name   = f'{model_name+model_adaption+uneven_batches}'
path      = f'./output/models/{model_name}/{epochs}/{opt}/'
pathfile    = path+file_name

if read_model:
    net.read_model(pathfile)
    if compute_NTK or log_parameters:
        net.read_log(pathfile)
    with open(f'{pathfile}.npy', 'rb') as f:
        train_losses = np.load(f)

# Save figure
file_name   = f'{model_name+model_adaption+uneven_batches}'
path        = f'./output/figures/{model_name}/{epochs}/{opt}/'
pathfile    = path+file_name
Path(path).mkdir(parents=True, exist_ok=True)

# Set network in evaluation mode
net.eval()

# Get exact solution
xplot, tplot, U_exact = u_exact()
xplot = [xplot, tplot]

## Plot 1 - Prediction and training loss
T_idxs = [0, 0.25, 0.5, 1.0]
# Plot 1 - Prediction and training loss
fig1, axs1 = plot_results2D(net, xplot, U_exact, T_idxs, train_losses, [r'$\mathcal{L}_{rr}$', r'$\mathcal{L}_{uu}$', r'$\mathcal{L}_{ii}$'])
fig1.tight_layout()
plt.savefig(pathfile+'_plot_2D')

if log_NTK and log_parameters:
    # Plot 2 - Parameter and ntk difference
    # fig2, axs2 = plt.subplots(1,2, figsize=(18,6))
    # plot_param_ntk_diff(net, fig2, axs2)
    # fig2.tight_layout()
    # plt.savefig(pathfile+'_plot_param_ntk_diff')

    # Plot 3 - Plot all NTK matrices
    fig3, axs3 = plt.subplots(1,4, figsize=(18,6))
    plot_NTK(net, fig3, axs3)
    fig3.tight_layout()
    plt.savefig(pathfile+'_plot_NTK')

    # Plot 4 - NTK matrix K change
    fig4, axs4 = plt.subplots(1,1)
    plot_NTK_change(net, fig4, axs4, plot_intervals=True)
    fig4.tight_layout()
    plt.savefig(pathfile+'_plot_NTK_change')

    # Plot 5 - Convergence rate for all matrices
    fig5, axs5 = plt.subplots(1,1)
    plot_convergence_rate(net, fig5, axs5)
    fig5.tight_layout()
    plt.savefig(pathfile+'_plot_convergence_rate')

    # Plot 6 - Adaption weights
    if len(adaption_weights)  == 0:
        adaption_weights = compute_adaptionWeights(net)
    fig6, axs6 = plt.subplots(1,1)

    plot_epochs = np.linspace(0, epochs,  adaption_weights.shape[0])

    axs6.semilogy(plot_epochs, adaption_weights)
    axs6.set_xlabel(r'$\lambda$')
    axs6.set_xlabel(r'$Epoch$')
    axs6.legend([r'$\lambda_{rr}$', r'$\lambda_{uu}$', r'$\lambda_{ii}$'])
    axs6.set_title(r'Adaption weights $\lambda$' + '\n' + r'$\overline{\lambda_{rr}}$: '+ f'{np.mean(adaption_weights[:,0]):.3f}' + r' $\overline{\lambda_{uu}}$: '+ f'{np.mean(adaption_weights[:,1]):.3f}' + r' $ \overline{\lambda_{ii}}$: '+ f'{np.mean(adaption_weights[:,2]):.3f}'  )
    fig6.tight_layout()
    if not Path(pathfile+'_plot_adaption_weights.png').is_file():
        plt.savefig(pathfile+'_plot_adaption_weights')

plt.tight_layout()
plt.show()

# save animation
fig, ani = plot_results2D_animate(net, xplot, U_exact, T_idxs, train_losses, [r'$\mathcal{L}_{rr}$', r'$\mathcal{L}_{uu}$', r'$\mathcal{L}_{ii}$'], ani_interval=10)
fig.tight_layout()
writer = matplotlib.animation.FFMpegWriter(fps=15, codec='h264')
ani.save(pathfile+'_ani_2D.mp4', writer=writer)