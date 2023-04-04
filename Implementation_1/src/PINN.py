
import torch
import torch.nn as nn



class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, PDE, dtype=torch.float32, device='cpu', log_parameters=True, log_NTK=False):
        super(PINN, self).__init__()

        self.dtype  = dtype
        self.device = device

        # initialize values for nn
        self.xin        = input_size
        self.xout       = output_size
        self.neurons    = neurons

        # Define layers of network
        self.layer1     = nn.Linear(input_size, neurons, dtype=dtype)
        self.layer2     = nn.Linear(neurons, output_size, dtype=dtype)

        self.layers = [self.layer1, self.layer2]

        self.activation = nn.Tanh()      

        # Initialize weights of the network
        self.apply(self._init_weights)

        # import and initialize PDE
        if hasattr(PDE,'pde_residual'):
            self.pde_residual = PDE.pde_residual
        if hasattr(PDE, 'bc_residual'):
            self.bc_residual = PDE.bc_residual
        if hasattr(PDE, 'ic_residual'):
            self.ic_residual = PDE.ic_residual

        # copy gradient computation
        self.compute_pde_gradient = PDE.compute_gradient

        # logging parameters
        if log_parameters:
            self.network_parameters_log = {}
        if log_NTK:
            self.NTK_log = {}


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
    
    def backward(self, X, U, f=None, g=None, h=None):

        if X.shape[1] == 2:
            xr = X[:, 0].view(-1,1)
            xb = X[:, 1].view(-1,1)
        else:
            xr = xb = X

        if len(U.shape) == 3:
            U_x = U[0]
            U_b = U[1]
        else:
            U_x = U
            U_b = U

        loss = []

        if hasattr(self, 'pde_residual') and f != None:
            residual        = self.pde_residual(xr, U_x, f).T
            self.pde_loss   = torch.mean(residual**2)
        
            loss.append(self.pde_loss)

        if hasattr(self, 'bc_residual') and g !=None:
            residual        = self.bc_residual(xb, U_b, g).T
            self.bc_loss    = torch.mean(residual**2)
        
            loss.append(self.bc_loss)   

        if hasattr(self, 'ic_residual') and h != None:
            residual        = self.ic_residual(X, U, h).T
            self.ic_loss    = torch.mean(residual**2)
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
        rhs = torch.zeros(x.shape, dtype=dtype, device=device)
        lhs = residual_fn(x, y_x, rhs)

        #### Compute the Jacobian
        J_y = []
        for i in range(len(x)):
            row = flatten_grad(torch.autograd.grad(lhs[:,i], self.parameters(), grad_outputs=torch.ones_like(lhs[:,i]), retain_graph=True))
            J_y.append(row)
        # End jacobian computation over parameters

        return torch.vstack(J_y)

    def NTK(self, X1, X2):

        PDE_K = False; BC_K = False

        if X1.shape[1] == 2 and X2.shape[1] == 2:
            xr1 = X1[:, 0].view(-1,1);   xb1 = X1[:, 1].view(-1,1)
            xr2 = X2[:, 0].view(-1,1);   xb2 = X2[:, 1].view(-1,1)
        else:
            xr1 = xb1 = X1
            xr2 = xb2 = X2

        if hasattr(self, 'pde_residual'):

            PDE_K = True

            J_r1 = self.compute_jacobian(xr1, self.compute_pde_gradient, self.pde_residual)
            J_r2 = self.compute_jacobian(xr2, self.compute_pde_gradient, self.pde_residual)

            # compute NTK matrix for PDE residual
            self.Krr       = J_r1 @ J_r2.T
            self.lambda_Krr  = torch.linalg.eigvals(self.Krr)
        # endif

        if hasattr(self, 'bc_residual'):

            BC_K = True   

            J_u1 = self.compute_jacobian(xb1, self.compute_pde_gradient, self.bc_residual)
            J_u2 = self.compute_jacobian(xb2, self.compute_pde_gradient, self.bc_residual)  

            # Compute NTK matrix for PDE boundary condition
            self.Kuu       = J_u1 @ J_u2.T
            self.lambda_Kuu  = torch.linalg.eigvals(self.Kuu)
            
        if PDE_K and BC_K:
            K1 = torch.vstack((J_u1,   J_r1))
            K2 = torch.hstack((J_u2.T, J_r2.T))

            self.K = K1 @ K2

            self.lambda_K = torch.linalg.eigvals(self.K)

    def log_NTK(self, epoch):
        self.NTK_log[epoch] = {'NTK_matrix': [self.K, self.Krr, self.Kuu], 
                               'NTK_eigenvalues': [self.lambda_K, self.lambda_Krr, self.lambda_Kuu]}
        