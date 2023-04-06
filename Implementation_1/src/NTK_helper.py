import torch

def compute_norm(matrix):
    # norm = torch.sqrt(torch.sum(matrix**2))
    if len(matrix.shape) > 1:
        norm = torch.linalg.matrix_norm(matrix, ord=2)
    else:
        norm = torch.linalg.norm(matrix, ord=2)
    return norm

def compute_parameter_diff(net): 

    dtype  = net.dtype
    device = net.device

    parameter_epochs    = list(net.network_parameters_log.keys())
    n                   = len(parameter_epochs)
    parameters_diff     = torch.zeros((n,1), dtype=dtype, device=device)
    initial_params      = net.network_parameters_log[0]

    for i, epoch in enumerate(parameter_epochs[1:]):
        diff = torch.tensor([0.0], dtype=dtype, device=device)
        params = net.network_parameters_log[epoch]
        for layer_weight, init_layer_weight in zip(params['weight'], initial_params['weight']):
            diff += compute_norm(layer_weight - init_layer_weight) / compute_norm(init_layer_weight)
        for layer_bias, init_layer_bias in zip(params['bias'], initial_params['bias']):
            diff += compute_norm(layer_bias - init_layer_bias) / compute_norm(init_layer_bias)

        parameters_diff[i+1] = diff

    return parameter_epochs, parameters_diff

def compute_NTK_diff(net): 

    dtype  = net.dtype
    device = net.device

    NTK_epochs = list(net.NTK_log.keys())
    n          = len(NTK_epochs)
    NTK_diff   = torch.zeros((n,1), dtype=dtype, device=device)

    K0 = net.NTK_log[0]['NTK_matrix'][0]

    for i,epoch in enumerate(NTK_epochs[1:]):
        K   = net.NTK_log[epoch]['NTK_matrix'][0]
        diff = torch.linalg.matrix_norm(K - K0, ord=2) / torch.linalg.matrix_norm(K0, ord=2)
        NTK_diff[i+1] = diff

    return NTK_epochs, NTK_diff

def compute_convergence_rate(net):

    NTK_epochs = list(net.NTK_log.keys())
    NTK_convergenceRate = []

    for epoch in NTK_epochs:
        NTK_convergenceRate_row = []
        for NTK_eigenvalues in net.NTK_log[epoch]['NTK_eigenvalues']:
            n = NTK_eigenvalues.shape[0]
            convergence_rate = torch.sum(NTK_eigenvalues) / n
            NTK_convergenceRate_row.append(convergence_rate)

        NTK_convergenceRate.append(torch.hstack(NTK_convergenceRate_row))

    NTK_convergenceRate = torch.vstack(NTK_convergenceRate)

    return NTK_epochs, NTK_convergenceRate