import torch


def compute_jacobian(x, net, grad_fn, residual_fn):

    # create function for use to flatten array
    flatten_grad = lambda row: torch.cat([elem.reshape(-1) for elem in row]).view(1,-1)

    device = net.device
    dtype  = net.dtype

    #### forward pass points with current parameters and compute gradients w.r.t points
    y   = net(x)
    y_x = grad_fn(y, x)
    #### Compute LHS of PDE equation or condition and set the rhs to zero
    rhs = torch.zeros(x.shape, dtype=dtype, device=device)
    lhs = residual_fn(x, y_x, rhs)

    #### Compute the Jacobian
    J_y = []
    for i in range(len(x)):
        row = flatten_grad(torch.autograd.grad(lhs[:,i], net.parameters(), grad_outputs=torch.ones_like(lhs[:,i]), retain_graph=True))
        J_y.append(row)
    # End jacobian computation over parameters

    return torch.vstack(J_y)

def NTK(net, X1, X2):

    device = net.device
    dtype  = net.dtype

    PDE_K = False; BC_K = False

    if X1.shape[1] == 2 and X2.shape[1] == 2:
        xr1 = X1[:, 0].view(-1,1);   xb1 = X1[:, 1].view(-1,1)
        xr2 = X2[:, 0].view(-1,1);   xb2 = X2[:, 1].view(-1,1)
    else:
        xr1 = xb1 = X1
        xr2 = xb2 = X2

    if hasattr(net, 'pde_residual'):

        PDE_K = True;

        J_r1 = compute_jacobian(xr1, net, net.compute_pde_gradient, net.pde_residual)
        J_r2 = compute_jacobian(xr2, net, net.compute_pde_gradient, net.pde_residual)

        # compute NTK matrix for PDE residual
        net.K_rr       = J_r1 @ J_r2.T
        net.lambda_rr  = torch.linalg.eigvals(net.K_rr)
    # endif

    if hasattr(net, 'bc_residual'):

        BC_K = True   

        J_u1 = compute_jacobian(xb1, net, net.compute_pde_gradient, net.bc_residual)
        J_u2 = compute_jacobian(xb2, net, net.compute_pde_gradient, net.bc_residual)  

        # Compute NTK matrix for PDE boundary condition
        net.K_uu       = J_u1 @ J_u2.T
        net.lambda_uu  = torch.linalg.eigvals(net.K_uu)
        
    if PDE_K and BC_K:
        K1 = torch.vstack((J_u1,   J_r1))
        K2 = torch.hstack((J_u2.T, J_r2.T))

        net.K = K1 @ K2

        net.lambda_K = torch.linalg.eigvals(net.K)

