import torch
import sympy as sm
import numpy as np


sympyTorchmodules = {'tan': torch.tan,'sin': torch.sin, 'cos': torch.cos, 'exp':torch.exp}

class Poisson1D:

    def __init__(self, a=None):

        self.setup_equations()
        self.setup_residuals()

        if a != None:
            self.a = 1

    def setup_equations(self, f_eqn=None, g_eqn=None):
        ### Setup

        # Variables/Coefficients
        a   = sm.symbols('a'); 

        # PDE States
        x   = sm.symbols('x')        # domain
        # xbc = sm.symbols('x1:3')   # partial domain for boundary condition

        u   = sm.symbols('u', cls=sm.Function)(x)
        ux  = u.diff(x)
        uxx = ux.diff(x)

        # Forcing/External/Boundary/Initial condition functions
        f   = sm.symbols('f', cls=sm.Function)(x)
        g   = sm.symbols('g', cls=sm.Function)(x)

        # Set up PDE_eqn
        self.PDE_eqn = sm.Eq(uxx, f)

        # Set up boundary condition
        # bc_eq1 = sm.Piecewise((u, sm.Eq(x, xbc[0])),  (u, sm.Eq(x, xbc[1])), (0, True))
        # bc_eq2 = sm.Piecewise((g, sm.Eq(x, xbc[0])),  (g, sm.Eq(x, xbc[1])), (0, True))
        self.BC_eqn  = sm.Eq(u, g)

        # For reuse in class
        self.x   = x 
        # self.xbc = xbc
        self.U = [u, ux, uxx]
        self.f = f 
        self.g = g 

    def setup_residuals(self):
        pde_residual = self.PDE_eqn.lhs - self.PDE_eqn.rhs
        bc_residual  = self.BC_eqn.lhs  - self.BC_eqn.rhs

        self.pde_residual = sm.lambdify([self.x, self.U, self.f], pde_residual, modules=sympyTorchmodules)
        self.bc_residual  = sm.lambdify([self.x, self.U, self.g], bc_residual,  modules=sympyTorchmodules)      

    def compute_gradient(self, u, x):
            
        ux   = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        uxx  = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux),  create_graph=True)[0]

        return torch.hstack([u, ux, uxx]).T

class ViscidBurger1D:

    def __init__(self, nu=None):

        self.nu = nu

        self.setup_equations()
        self.setup_residuals()
        
    def setup_equations(self, f_eqn=None, g_eqn=None):
        ### Setup

        # Variables/Coefficients
        # a   = sm.symbols('a'); 

        if self.nu == None:
            nu  = sm.symbols('nu');
        else:
            nu = self.nu

        # PDE States
        x   = sm.symbols('x')      # space domain
        t   = sm.symbols('t')      # time domain
        # xbc = sm.symbols('x1:3')   # partial domain for boundary condition

        u   = sm.symbols('u', cls=sm.Function)(x,t) 
        ux  = u.diff(x)
        uxx = ux.diff(x)
        ut  = u.diff(t)

        # Forcing/External/Boundary/Initial condition functions
        f   = sm.symbols('f', cls=sm.Function)(x,t)
        g   = sm.symbols('g', cls=sm.Function)(x,t)
        h   = sm.symbols('h', cls=sm.Function)(x,t)

        # Set up PDE_eqn
        self.PDE_eqn = sm.Eq(ut + u*ux - nu*uxx, f)

        # Set up boundary condition
        self.BC_eqn  = sm.Eq(u, g)

        # Set up initial condition
        self.IC_eqn = sm.Eq(u, h)

        # For reuse in class
        self.x   = x 
        self.t   = t
        # self.xbc = xbc
        self.U = [u, ux, ut, uxx]
        self.f = f 
        self.g = g 
        self.h = h 

    def setup_residuals(self):
        pde_residual = self.PDE_eqn.lhs - self.PDE_eqn.rhs
        bc_residual  = self.BC_eqn.lhs  - self.BC_eqn.rhs
        ic_residual  = self.IC_eqn.lhs  - self.IC_eqn.rhs

        self.pde_residual = sm.lambdify([[self.x, self.t], self.U, self.f], pde_residual, modules=sympyTorchmodules) 
        self.bc_residual  = sm.lambdify([[self.x, self.t], self.U, self.g], bc_residual,  modules=sympyTorchmodules)
        self.ic_residual  = sm.lambdify([[self.x, self.t], self.U, self.h], ic_residual,  modules=sympyTorchmodules)      

    def compute_gradient(self, u, x):
        ux_ut   = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(u),   create_graph=True)[0]    
        ux   = ux_ut[:,0].view(-1,1)
        ut   = ux_ut[:,1].view(-1,1)
        uxx_uxt  = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        uxx = uxx_uxt[:,0].view(-1,1)

        return torch.hstack([u, ux, ut, uxx]).T

