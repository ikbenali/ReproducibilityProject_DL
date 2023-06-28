"""
CS4240: Deep Learning
Reproducibility project
Author: Ali Ul Haq
"""
import torch
import numpy as np
import torch.nn as nn

import os
import h5py

class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, PDE, init_type='normal', dtype=torch.float32, device='cpu', log_parameters=True, log_NTK=False):
        super(PINN, self).__init__()

        self.dtype  = dtype
        self.device = device

        self.enable_parameter_log = log_parameters
        self.enable_NTK_log       = log_NTK

        # initialize values for nn
        self.xin        = input_size
        self.xout       = output_size
        self.neurons    = torch.tensor(neurons)
        
        # single activation function for whole network
        self.activation = nn.Tanh()      

        # Define layers of network
        self.input_layer      = nn.Linear(input_size, self.neurons[0], dtype=dtype,  device=device)
        layers = [self.input_layer]
        layers.append(self.activation)

        if len(neurons) > 1:
            for i, neuron in enumerate(self.neurons[1:]):
                layer = nn.Linear(self.neurons[i-1], neuron, dtype=dtype, device=device)
                layers.append(layer)
                layers.append(self.activation)

        self.output_layer     = nn.Linear(self.neurons[-1], output_size, dtype=dtype, device=device)
        layers.append(self.output_layer)

        self.layers = nn.Sequential(*layers)
        self.n_layers = len(self.layers)

        # Initialize weights of the network
        self.init_type = init_type
        self.apply(self._init_weights)

        # initialize normalization
        self.normalized = False

        # import and initialize PDE
        lambda_adaptation = []
        if hasattr(PDE,'pde_residual'):
            self.pde_residual = PDE.pde_residual
            lambda_adaptation.append(1.)
        if hasattr(PDE, 'bc_residual'):
            self.bc_residual = PDE.bc_residual
            lambda_adaptation.append(1.)
        if hasattr(PDE, 'ic_residual'):
            self.ic_residual = PDE.ic_residual
            lambda_adaptation.append(1.)

        # copy gradient computation
        self.compute_pde_gradient = PDE.compute_gradient

        # logging parameters
        if log_parameters:
            self.network_parameters_log = {}
        if log_NTK:
            self.NTK_log = {}

        # Adaptation algorithm
        self.lambda_adaptation = torch.tensor(lambda_adaptation, dtype=dtype, device=device)

    def _init_weights(self, module):
        # Glorot Weight initalisation
        if isinstance(module, nn.Linear):
            n = module.in_features
            std = 1 / np.sqrt(n)
            if module.weight is not None: 
                if self.init_type == 'normal':
                    nn.init.normal_(module.weight, 0., std)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight)                
            if module.bias is not None:
                if self.init_type == 'normal':
                    nn.init.normal_(module.bias, 0., std)
                elif self.init_type == 'xavier':
                    nn.init.zeros_(module.bias)
    
    def _init_normalization(self, X_coords):
        self.normalized = True
        NX = int(1e5)
        N = X_coords.shape[0]
        dim = X_coords.shape[1]
        X = []
        for i in range(dim):
            X_i  = torch.linspace(X_coords[0,i].item(), X_coords[1,i].item(), NX, dtype=self.dtype, device=self.device).view(-1,1)
            X.append(X_i)
        X = torch.hstack(X)

        self.mu_X    = X.mean(0)
        self.sigma_X = X.std(0)
                                
    def log_parameters(self, epoch):
        params = {k: v.detach().clone() for k, v in self.named_parameters()}

        weights = []
        bias    = []

        for layer_param in params.keys():
            if 'weight' in layer_param:
                weights.append(params[layer_param].detach().cpu().numpy())
            elif 'bias' in layer_param:
                bias.append(params[layer_param].detach().cpu().numpy())

        self.network_parameters_log[epoch] = {'weight': weights, 'bias':bias}

    def forward(self, x):
        if self.normalized:
            x = (x - self.mu_X) / self.sigma_X
        x = self.layers(x)
        return x
    
    def backward(self, X, U, f=None, g=None, h=None, use_adaption=False):

        if isinstance(X, list) and isinstance(U, list):
            N = len(X);       M = len(U)
        elif isinstance(X, torch.Tensor) and isinstance(U, torch.Tensor):
            N = X.shape[0];   M = U.shape[0]

        if N == 3:
            xr = X[0].T
            xb = X[1].T
            xi = X[2].T
        elif N == 2:
            xr = X[0].T
            xb = X[1].T
        else:
            xr = xb = xi = X.T
        if M == 3:
            U_x = U[0]
            U_xb = U[1]
            U_xi = U[2]
        elif M == 2:
            U_x  = U[0]
            U_xb = U[1]
        else:
            U_x = U_xb = U_xi = U

        loss = []

        if hasattr(self, 'pde_residual') and f != None:
            lambda_1 = 1.0
            if use_adaption:
                lambda_1 = self.lambda_adaptation[0]
                
            residual        = self.pde_residual(xr, U_x, f)
            self.pde_loss   = lambda_1 * torch.mean(torch.square(residual))
        
            loss.append(self.pde_loss)

        if hasattr(self, 'bc_residual') and g !=None:
            lambda_2 = 1.0

            if use_adaption:
                lambda_2 = self.lambda_adaptation[1]

            residual        = self.bc_residual(xb, U_xb, g)
            self.bc_loss    = lambda_2 * torch.mean(torch.square(residual))
        
            loss.append(self.bc_loss)   

        if hasattr(self, 'ic_residual') and h != None:
            lambda_3 = 1.0
            if use_adaption:
                lambda_3 = self.lambda_adaptation[2]

            residual        = self.ic_residual(xi, U_xi, h)
            self.ic_loss    = lambda_3 * torch.mean(torch.square(residual))
            loss.append(self.ic_loss)

        self.train_loss = torch.stack(loss, dim=0)
        self.loss       = self.train_loss.sum()

    @staticmethod
    def compute_vjp(lhs, params, v):
        row = torch.autograd.grad(lhs, params, grad_outputs=v, retain_graph=True)
        return row
    
    def compute_jacobian(self, x1, x2, grad_fn, residual_fn, compute_grad=True, offload=True):

        device = self.device
        dtype  = self.dtype

        #### forward pass points with current parameters and compute gradients w.r.t points

        # Compute for batch set 1
        y1      = self.forward(x1)
        y1_grad = grad_fn(y1, x1)

        # Compute for batch set 2
        y2   = self.forward(x2)
        y2_grad = grad_fn(y2, x2)

        #### Compute LHS of PDE equation and set the rhs to zero
        rhs = torch.zeros((x1.shape[0],1), dtype=dtype, device=device).T

        # batch set 1
        lhs1 = residual_fn(x1.T, y1_grad, rhs).view(-1)

        # batch set 2
        lhs2 = residual_fn(x2.T, y2_grad, rhs).view(-1)

        # Dimension should be the same for batch set 1 and 2
        N   = lhs1.shape[0]
        I_N = torch.eye(N, dtype=dtype, device=device)

        jacobian = torch.vmap(self.compute_vjp, (None, None, 0), chunk_size=400)

        #### Compute the Jacobian
        if N <= 400:
            J_y1 = list(torch.autograd.grad(lhs1, self.parameters(), I_N, is_grads_batched=True))
            J_y2 = list(torch.autograd.grad(lhs2, self.parameters(), I_N, is_grads_batched=True))
        else:
            J_y1 = []
            J_y2 = []
            for param in self.parameters():
                J_y1_i = jacobian(lhs1, param, I_N)[0]
                J_y2_i = jacobian(lhs2, param, I_N)[0]

                J_y1.append(J_y1_i)
                J_y2.append(J_y2_i)

        ### Compute the NTK matrix
        K  = []
        for j_i, [j1,j2] in enumerate(zip(J_y1, J_y2)):
            j1 = j1.view(N,-1)
            j2 = j2.view(N,-1)
            K_i = torch.mm(j1, j2.T)
            K.append(K_i)

            # move to RAM to offload VRAM
            if offload:
                J_y1[j_i] = j1.detach().cpu()
                J_y2[j_i] = j2.detach().cpu()
            else:
                J_y1[j_i] = j1.detach()
                J_y2[j_i] = j2.detach()

        K = torch.stack(K).sum(0)

        if offload:
            K = K.cpu()

        return K, J_y1, J_y2
    
    def NTK(self, X1, X2):

        PDE_K = False; BC_K = False;    IC_K = False

        if isinstance(X1, list) and isinstance(X2, list):
            N1 = len(X1);    N2 = len(X2)
        elif isinstance(X1, torch.Tensor) and isinstance(X2, torch.Tensor):
            N1 = X1.shape[0];   N2 = X2.shape[0]
        
        if N1 == 3 and N2 == 3:
            xr1 = X1[0];   xb1 = X1[1];   xi1 = X1[2]
            xr2 = X2[0];   xb2 = X2[1];   xi2 = X2[2]
        elif N1 == 2 and N2 == 2:
            xr1 = X1[0];   xb1 = X1[1]
            xr2 = X2[0];   xb2 = X2[1]
        else:
            xr1 = X1 
            xr2 = X2 

        if hasattr(self, 'pde_residual'):

            PDE_K = True

            self.Krr, J_r1, J_r2 = self.compute_jacobian(xr1, xr2, self.compute_pde_gradient, self.pde_residual)
            # compute NTK eigenvalues for PDE residual
            self.lambda_Krr = torch.linalg.eigvals(self.Krr)

        if hasattr(self, 'bc_residual'):
            BC_K = True   

            self.Kuu, J_u1, J_u2 = self.compute_jacobian(xb1, xb2, self.compute_pde_gradient, self.bc_residual)
            # Compute NTK eigenvalues for PDE boundary condition
            self.lambda_Kuu = torch.linalg.eigvals(self.Kuu)

        if hasattr(self, 'ic_residual'):
            IC_K = True

            self.Kii, J_i1, J_i2 = self.compute_jacobian(xi1, xi2, self.compute_pde_gradient, self.ic_residual)
            # Compute NTK eigenvalues for PDE boundary condition
            self.lambda_Kii = torch.linalg.eigvals(self.Kii)

        if PDE_K and BC_K and IC_K:
            N = xr1.shape[0] + xb1.shape[0] + xi1.shape[0]
            M = xr1.shape[0]
            K = torch.zeros((N,N), dtype=self.dtype, device=self.device)

            for i in range(len(J_u1)):
                K1      = torch.vstack((J_u1[i],   J_r1[i],     J_i1[i])).to(self.device)
                K2      = torch.hstack((J_u2[i].T, J_r2[i].T,   J_i2[i].T)).to(self.device)
                K      += torch.matmul(K1, K2)
                
            self.K          = K          
            self.lambda_K   = torch.linalg.eigvals(K)
        elif PDE_K and BC_K:
            N = xr1.shape[0] + xb1.shape[0]
            M = xr1.shape[0]
            K = torch.zeros((N,N), dtype=self.dtype, device=self.device)

            for i in range(len(J_u1)):
                K1      = torch.vstack((J_u1[i],   J_r1[i])).to(self.device)
                K2      = torch.hstack((J_u2[i].T, J_r2[i].T)).to(self.device)
                K      += torch.matmul(K1, K2)

            self.K        = K
            self.lambda_K = torch.linalg.eigvals(K)

        # update adaption terms
        old_weights = self.lambda_adaptation.detach().clone()
        K_trace = torch.trace(self.K)
        if PDE_K:
            Krr_trace = torch.trace(self.Krr)
            # print(K_trace, Krr_trace)
            self.lambda_adaptation[0] = K_trace / Krr_trace
        if BC_K:
            Kuu_trace = torch.trace(self.Kuu)
            # print(K_trace, Kuu_trace)
            self.lambda_adaptation[1] = K_trace / Kuu_trace
        if IC_K:
            Kii_trace = torch.trace(self.Kii)
            self.lambda_adaptation[2] = K_trace / Kii_trace 

        # Clip adaption terms to not be negative and also not diminish the NTK and loss contribution
        for i,weight in enumerate(self.lambda_adaptation):
            if weight < 1.:
                self.lambda_adaptation[i] = old_weights[i]

    def log_NTK(self, epoch):
        NTK_matrix = []
        NTK_eigenvalues = []
        if hasattr(self, 'K'):
            NTK_matrix.append(self.K.detach().cpu().numpy())
            NTK_eigenvalues.append(self.lambda_K.detach().cpu().numpy())
        if hasattr(self, 'Krr'):
            NTK_matrix.append(self.Krr.detach().cpu().numpy())
            NTK_eigenvalues.append(self.lambda_Krr.detach().cpu().numpy())
        if hasattr(self, 'Kuu'):
            NTK_matrix.append(self.Kuu.detach().cpu().numpy())
            NTK_eigenvalues.append(self.lambda_Kuu.detach().cpu().numpy())
        if hasattr(self, 'Kii'):
            NTK_matrix.append(self.Kii.detach().cpu().numpy())
            NTK_eigenvalues.append(self.lambda_Kii.detach().cpu().numpy())

        self.NTK_log[epoch] = {'NTK_matrix': NTK_matrix, 
                               'NTK_eigenvalues': NTK_eigenvalues}
        
    def save_model(self, pathfile):
        torch.save(self.state_dict(), f'{pathfile}.pt')

    def read_model(self, pathfile):
        self.load_state_dict(torch.load(f'{pathfile}.pt'))

    def save_log(self, pathfile, reset=True):

        filename  = f'{pathfile}.hdf5'

        if os.path.isfile(filename):
            f = h5py.File(filename, mode='a')
        else:
            f = h5py.File(filename, mode='w')

        logs = [] 

        if hasattr(self, 'network_parameters_log'):
            logs.append(self.network_parameters_log)
        if hasattr(self, 'NTK_log'):
            logs.append(self.NTK_log)

        for log in logs:
            for epoch in log:
                for group in log[epoch]:
                    grp_name = f'{str(epoch)}/{group}'
                    if grp_name not in f:
                        grp = f.create_group(grp_name)
                    else:
                        grp = f[grp_name]
                    for i, item in enumerate(log[epoch][group]):
                        if isinstance(item, torch.Tensor):
                            new_item = item.detach().cpu().numpy()
                        else:
                            new_item = item
                        if str(i) not in grp:
                            grp.create_dataset(str(i), data=new_item)
                        else:
                            grp[str(i)][:] = new_item
        if reset:
            if hasattr(self, 'network_parameters_log'):
                self.network_parameters_log = {}
            if hasattr(self, 'NTK_log'):
                self.NTK_log = {}
        f.close()

    def read_log(self, pathfile):
        # read file
        f = h5py.File(f'{pathfile}.hdf5', mode='r')

        epochs  = []
        NTK_log = {}
        network_parameters_log = {}

        for epoch in f:
            for group in f[epoch]:
                row = []
                for array in f[epoch][group]:
                    value = f[epoch][group][array][:]
                    row.append(value)
                if group == "weight":
                    weights = row
                elif group == "bias":
                    bias = row 
                elif group == "NTK_matrix":
                    NTK_matrix = row
                elif group == "NTK_eigenvalues":
                    NTK_eigenvalues = row            
            epoch = int(epoch)
            epochs.append(epoch)
            network_parameters_log[epoch] = {'weight': weights, 'bias':bias}

            if self.enable_NTK_log:
                NTK_log[epoch] = {'NTK_matrix': NTK_matrix, 
                                        'NTK_eigenvalues': NTK_eigenvalues}
        epochs.sort()
        self.network_parameters_log = {epoch_i: network_parameters_log[epoch_i] for epoch_i in epochs}
        if self.enable_NTK_log:
            self.NTK_log                = {epoch_i: NTK_log[epoch_i] for epoch_i in epochs}

        f.close()
