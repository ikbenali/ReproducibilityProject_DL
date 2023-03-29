
import torch
import torch.nn as nn



class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, PDE, dtype=torch.float32, device='cpu'):
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

        # import and initialize PDE
        if hasattr(PDE,'pde_residual'):
            self.pde_residual = PDE.pde_residual
        if hasattr(PDE, 'bc_residual'):
            self.bc_residual = PDE.bc_residual
        if hasattr(PDE, 'ic_residual'):
            self.ic_residual = PDE.ic_residual

        self.apply(self._init_weights)

        # copy gradient computation
        self.compute_pde_gradient = PDE.compute_gradient

    def _init_weights(self, module):
        # Glorot Weight initalisation
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)            
            if module.bias is not None:
                nn.init.normal_(module.bias.data)            


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

