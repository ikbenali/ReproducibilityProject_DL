import torch



def NTK(net, X1, X2):

    device = net.device
    dtype  = net.dtype

    # Get params of network
    params = {k: v for k, v in net.named_parameters()}

    PDE_K = False; BC_K = False

    if X1.shape[1] == 2 and X2.shape[1] == 2:
        xr1 = X1[:, 0].view(-1,1);   xb1 = X1[:, 1].view(-1,1)
        xr2 = X2[:, 0].view(-1,1);   xb2 = X2[:, 1].view(-1,1)

    else:
        xr1 = xb1 = X1
        xr2 = xb2 = X2

    if hasattr(net, 'pde_residual'):

        PDE_K = True;

        #### forward pass points with current parameters and compute gradients w.r.t interior points
        # For X
        u_hat_x1    = net(xr1)
        U_x1        = net.compute_pde_gradient(u_hat_x1, xr1)
        # For X'
        u_hat_x2    = net.forward(xr2)
        U_x2        = net.compute_pde_gradient(u_hat_x2, xr2)

        # Compute LHS of PDE 

        # Only compute lhs, so rhs function is set to 0
        f = torch.zeros(xr1.size(), device=device).T

        L_u1   = net.pde_residual(xr1, U_x1, f).T
        L_u2   = net.pde_residual(xr2, U_x2, f).T

        # L_u1.retain_grad(); L_u2.retain_grad()

        J_r1 = [];     J_r2 = []

        for i, layer_param in enumerate(params.keys()):

            theta    = params[layer_param]

            if 'weight' in layer_param:
                L_u1_grad = torch.autograd.grad(L_u1, theta, grad_outputs=torch.ones_like(L_u1), retain_graph=True)[0].flatten()
                L_u2_grad = torch.autograd.grad(L_u2, theta, grad_outputs=torch.ones_like(L_u2), retain_graph=True)[0].flatten()

                J_r1.append(L_u1_grad);    J_r2.append(L_u2_grad)
        ### End gradient computation over parameters

        J_r1  = torch.stack(J_r1, dim=0).T;    J_r2 = torch.stack(J_r2, dim=0).T
        
        # compute NTK matrix for PDE residual
        net.K_rr       = J_r1 @ J_r2.T
        net.lambda_rr  = torch.linalg.eigvals(net.K_rr)
    # endif

    if hasattr(net, 'bc_residual'):

        BC_K = True 

        #### forward pass points with current parameters and compute gradients w.r.t boundary points
        # For Xb
        u_hat_xb1    = net.forward(xb1)
        U_xb1        = net.compute_pde_gradient(u_hat_xb1, xb1)
        # For Xb'
        u_hat_xb2    = net.forward(xb2)
        U_xb2        = net.compute_pde_gradient(u_hat_xb2, xb2)

        # Compute LHS of B.C. 
        # Only compute lhs, so rhs function is set to 0
        g = torch.zeros(xb1.size(), device=device).T

        u1   = net.bc_residual(xb1, U_xb1, g).T.flatten()
        u2   = net.bc_residual(xb2, U_xb2, g).T.flatten()

        J_u1 = [];     J_u2 = []

        for i, layer_param in enumerate(params.keys()):
            theta    = params[layer_param]

            if 'weight' in layer_param:
                u1_grad = torch.autograd.grad(u1, theta, grad_outputs=torch.ones_like(u1), retain_graph=True)[0].flatten()
                u2_grad = torch.autograd.grad(u2, theta, grad_outputs=torch.ones_like(u2), retain_graph=True)[0].flatten()

                J_u1.append(u1_grad);    J_u2.append(u2_grad)

        ### End backward computation over parameters
        
        J_u1  = torch.stack(J_u1, dim=0).T;    J_u2 = torch.stack(J_u2, dim=0).T

        net.K_uu       = J_u1 @ J_u2.T
        net.lambda_uu  = torch.linalg.eigvals(net.K_uu)
        
    if PDE_K and BC_K:

        K1 = torch.vstack((J_u1,   J_r1))
        K2 = torch.hstack((J_u2.T, J_r2.T))

        net.K = K1 @ K2

        net.lambda_K = torch.linalg.eigvals(net.K)

