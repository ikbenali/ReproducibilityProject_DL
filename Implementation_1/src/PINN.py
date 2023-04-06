
import torch
import torch.nn as nn

import orjson
import numpy as np


class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, PDE, dtype=torch.float32, device='cpu', log_parameters=True, log_NTK=False):
        super(PINN, self).__init__()

        self.dtype  = dtype
        self.device = device

        # initialize values for nn
        self.xin        = input_size
        self.xout       = output_size
        self.neurons    = torch.tensor(neurons)

        # Define layers of network
        self.input_layer      = nn.Linear(input_size, self.neurons[0], dtype=dtype,  device=device)
        self.layers = [self.input_layer]
        if len(neurons) > 1:
            for i, neuron in enumerate(self.neurons[1:]):
                layer = nn.Linear(self.neurons[i-1], neuron, dtype=dtype, device=device)
                self.layers.append(layer)

        self.output_layer     = nn.Linear(self.neurons[-1], output_size, dtype=dtype, device=device)
        self.layers.append(self.output_layer)
        
        # single activation function for whole network
        self.activation = nn.Tanh()      

        # Initialize weights of the network
        self.apply(self._init_weights)

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
            nn.init.xavier_normal_(module.weight.data)    
            if module.bias is not None:
                nn.init.normal_(module.bias.data)            

    def log_parameters(self, epoch):
        params = {k: v.detach().clone() for k, v in self.named_parameters()}

        weights = []
        bias    = []

        for layer_param in params.keys():
            if 'weight' in layer_param:
                weights.append(params[layer_param])
            elif 'bias' in layer_param:
                bias.append(params[layer_param])

        self.network_parameters_log[epoch] = {'weight': weights, 'bias':bias}

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x
    
    def backward(self, X, U, f=None, g=None, h=None, use_adaption=False):

        if X.shape[0] == 3:
            xr = X[0].T
            xb = X[1].T
            xi = X[2].T
            
        elif X.shape[0] == 2:
            xr = X[0].T
            xb = X[1].T
        else:
            xr = xb = X

        if U.shape[0] == 3:
            U_x = U[0]
            U_xb = U[1]
            U_xi = U[2]
        elif U.shape[0] == 2:
            U_x  = U[0]
            U_xb = U[1]
        else:
            U_x = U
            U_b = U

        loss = []

        if hasattr(self, 'pde_residual') and f != None:
            lambda_1 = 1.0
            if use_adaption:
                lambda_1 = self.lambda_adaptation[0]

            residual        = self.pde_residual(xr, U_x, f).T
            self.pde_loss   = lambda_1 * torch.mean(residual**2)
        
            loss.append(self.pde_loss)

        if hasattr(self, 'bc_residual') and g !=None:
            lambda_2 = 1.0

            if use_adaption:
                lambda_2 = self.lambda_adaptation[1]

            residual        = self.bc_residual(xb, U_xb, g).T
            self.bc_loss    = lambda_2*torch.mean(residual**2)
        
            loss.append(self.bc_loss)   

        if hasattr(self, 'ic_residual') and h != None:
            lambda_3 = 1.0
            if use_adaption:
                lambda_3 = self.lambda_adaptation[2]
            residual        = self.ic_residual(xi, U_xi, h).T
            self.ic_loss    = lambda_3*torch.mean(residual**2)
            loss.append(self.ic_loss)

        loss = torch.stack(loss, dim=0).sum()
        
        self.loss = loss

    def compute_jacobian(self, x, grad_fn, residual_fn):

        # create function for use to flatten array
        flatten_grad = lambda row: torch.cat([elem.reshape(-1) for elem in row]).view(1,-1)

        device = self.device
        dtype  = self.dtype

        #### forward pass points with current parameters and compute gradients w.r.t points
        y   = self(x)
        y_x = grad_fn(y, x)
        #### Compute LHS of PDE equation or condition and set the rhs to zero
        rhs = torch.zeros((x.shape[0],1), dtype=dtype, device=device).T
        lhs = residual_fn(x.T, y_x, rhs)

        #### Compute the Jacobian
        J_y = []
        for i in range(len(x)):
            row = flatten_grad(torch.autograd.grad(lhs[:,i], self.parameters(), grad_outputs=torch.ones_like(lhs[:,i]), retain_graph=True))
            J_y.append(row)
        # End jacobian computation over parameters

        return torch.vstack(J_y)

    def NTK(self, X1, X2):

        PDE_K = False; BC_K = False;    IC_K = False

        if X1.shape[0] == 3 and X2.shape[0] == 3:
            xr1 = X1[0, :];   xb1 = X1[1, :];   xi1 = X1[2, :]
            xr2 = X2[0, :];   xb2 = X2[1, :];   xi2 = X2[2, :]
        elif X1.shape[0] == 2 and X2.shape[0] == 2:
            xr1 = X1[0, :];   xb1 = X1[1, :]
            xr2 = X2[0, :];   xb2 = X2[1, :]
        else:
            xr1 = xb1 = X1
            xr2 = xb2 = X2

        if hasattr(self, 'pde_residual'):

            PDE_K = True

            J_r1 = self.compute_jacobian(xr1, self.compute_pde_gradient, self.pde_residual)
            J_r2 = self.compute_jacobian(xr2, self.compute_pde_gradient, self.pde_residual)

            # compute NTK matrix for PDE residual
            self.Krr            = torch.matmul(J_r1, J_r2.T)
            self.lambda_Krr, _  = torch.linalg.eig(self.Krr)
        # endif

        if hasattr(self, 'bc_residual'):

            BC_K = True   

            J_u1 = self.compute_jacobian(xb1, self.compute_pde_gradient, self.bc_residual)
            J_u2 = self.compute_jacobian(xb2, self.compute_pde_gradient, self.bc_residual)  

            # Compute NTK matrix for PDE boundary condition
            self.Kuu            = torch.matmul(J_u1, J_u2.T)
            self.lambda_Kuu, _  = torch.linalg.eig(self.Kuu)
        
        if hasattr(self, 'ic_residual'):
            IC_K = True

            J_i1 = self.compute_jacobian(xi1, self.compute_pde_gradient, self.ic_residual)
            J_i2 = self.compute_jacobian(xi2, self.compute_pde_gradient, self.ic_residual)

            # Compute NTK matrix for PDE boundary condition
            self.Kii            = torch.matmul(J_i1, J_i2.T)
            self.lambda_Kii, _  = torch.linalg.eig(self.Kii)

        if PDE_K and BC_K and IC_K:
            K1 = torch.vstack((J_u1,   J_r1, J_i1))
            K2 = torch.hstack((J_u2.T, J_r2.T, J_i2.T))
            self.K = torch.matmul(K1, K2)
            self.lambda_K,_ = torch.linalg.eig(self.K)

        elif PDE_K and BC_K:
            K1 = torch.vstack((J_u1,   J_r1))
            K2 = torch.hstack((J_u2.T, J_r2.T))

            self.K = torch.matmul(K1, K2)

            self.lambda_K,_ = torch.linalg.eig(self.K)

        # update adaption terms
        K_trace = torch.trace(self.K)

        if PDE_K:
            Krr_trace = torch.trace(self.Krr)
            self.lambda_adaptation[0] = K_trace / Krr_trace
        if BC_K:
            Kuu_trace = torch.trace(self.Kuu)
            self.lambda_adaptation[1] = K_trace / Kuu_trace
        if IC_K:
            Kii_trace = torch.trace(self.Kii)
            self.lambda_adaptation[2] = K_trace / Kii_trace

    def log_NTK(self, epoch):
        NTK_matrix = []
        NTK_eigenvalues = []
        if hasattr(self, 'K'):
            NTK_matrix.append(self.K)
            NTK_eigenvalues.append(self.lambda_K)
        if hasattr(self, 'Krr'):
            NTK_matrix.append(self.Krr)
            NTK_eigenvalues.append(self.lambda_Krr)
        if hasattr(self, 'Kuu'):
            NTK_matrix.append(self.Kuu)
            NTK_eigenvalues.append(self.lambda_Kuu)
        if hasattr(self, 'Kii'):
            NTK_matrix.append(self.Kii)
            NTK_eigenvalues.append(self.lambda_Kii)

        self.NTK_log[epoch] = {'NTK_matrix': NTK_matrix, 
                               'NTK_eigenvalues': NTK_eigenvalues}
        
    
    def save_model(self, pathfile):
        torch.save(self.state_dict(), f'{pathfile}.pt')
    
    def save_log(self, pathfile):
        def default(obj):  
            if obj.dtype == np.complex64:
                return {'complex': np.vstack([obj.real, obj.imag])}

        with open(pathfile+'.json', 'wb') as f:
            logs = {"network_parameter_log": log2json(self.network_parameters_log), "NTK_log": log2json(self.NTK_log)}
            f.write(orjson.dumps(logs, default=default, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS))

    def read_model(self, pathfile):
        self.load_state_dict(torch.load(f'{pathfile}.pt'))

    def read_log(self, pathfile):
        with open(pathfile+'.json', 'rb') as f:
            logs = orjson.loads(f.read())

        self.network_parameters_log = json2log(logs['network_parameter_log'], self.device)
        self.NTK_log                = json2log(logs['NTK_log'], self.device)

def log2json(log):
    new_log = {}
    for key in log.keys():
        value = log[key]
        if isinstance(value, dict):
            value = log2json(value)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    new_item = item.detach().cpu().numpy()
                    if new_item.dtype == np.complex64:
                        new_item = {'complex': np.vstack([new_item.real, new_item.imag])}
                    new_list.append(new_item)
            value = new_list
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        new_log[key] = value
    return new_log

def json2log(json, device='cpu'):
    log = {}
    for key in json.keys():
        value = json[key]
        if isinstance(value, dict):
            value = json2log(value)
        elif isinstance(value, list):
            new_list = []
            i = 0
            for item in value:
                if isinstance(item, dict):
                    if 'complex' in item.keys():
                        real = torch.Tensor(item['complex'][0]).view(-1,1)
                        imag = torch.Tensor(item['complex'][1]).view(-1,1)
                        new_item = torch.view_as_complex(torch.hstack([real, imag]))
                elif isinstance(item, list):
                    # check for nested list:
                    if isinstance(item[0], list):
                        new_item = torch.stack([torch.Tensor(item_i) for item_i in item], dim=0)
                    else:
                        new_item = torch.Tensor(item)
                elif isinstance(item, np.ndarray):
                    new_item = torch.Tensor(item)
                new_list.append(new_item)
                i+= 1
            value = new_list
        elif isinstance(value, np.ndarray):
            value = torch.Tensor(value, device=self.device)
        if key.isdigit():
            key = int(key)
        log[key] = value
    return log
