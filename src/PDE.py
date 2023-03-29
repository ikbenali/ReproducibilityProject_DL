import torch
import sympy as sm
import numpy as np


sympyTorchmodules = {'sin': torch.sin, 'cos': torch.cos}

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
        self.PDE_eqn = sm.Eq(uxx,f)

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

    def compute_gradient(self, u, x, t=None):
            
        ux   = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(u),  retain_graph=True, create_graph=True)[0]
        uxx  = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

        return torch.hstack([u, ux, uxx]).T

