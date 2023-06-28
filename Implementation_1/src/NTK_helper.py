import torch
import numpy as np

def compute_norm(matrix):

    if torch.cuda.is_available():
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.to(torch.device('cuda'))
        elif isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix).to(torch.device('cuda'))

        if len(matrix.shape) > 1:
            norm = torch.linalg.matrix_norm(matrix, 2)
        else:
            norm = torch.linalg.norm(matrix, 2)
    elif isinstance(matrix, torch.Tensor):
        if len(matrix.shape) > 1:
            norm = torch.linalg.matrix_norm(matrix, 2)
        else:
            norm = torch.linalg.norm(matrix, 2)
    elif isinstance(matrix, np.ndarray):
        norm = np.linalg.norm(matrix, 2)

    return norm

def NTK_scheduler(net, optimizer):
    """
    Adaptive learning rate adjustment
    """

    # Estimate maximum learning rate according to forward euler stability
    max_eigenvalue = torch.max(torch.abs(torch.real(net.lambda_K)))
    max_lr = 2/max_eigenvalue
    old_lr = optimizer.param_groups[0]['lr']
    
    for g in optimizer.param_groups:
        if g['lr'] > max_lr:
            new_lr = f'{max_lr:.4E}'
            smallest_lr = float(new_lr[0:3])
            power_      = int(new_lr.split('E')[-1])
            g['lr'] = smallest_lr * 10**power_

            print(f"Learning step greater than max_NTK_lr: 2 / lambda_max, adjusting learning rate\nold_lr = {old_lr} new_lr = {optimizer.param_groups[0]['lr']} max_lr: {max_lr.item()}") 


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

        for wi, [layer_weight, init_layer_weight] in enumerate(zip(params['weight'], initial_params['weight'])):

            init_norm = compute_norm(init_layer_weight)

            if init_norm.item() == 0.0 and i == 0:
                init_norm = torch.tensor([1.0], dtype=dtype, device=device)

            if init_norm.item() != 0.0:
                diff += compute_norm(layer_weight - init_layer_weight) / init_norm
            else:
                diff += compute_norm(layer_weight - prev_params['weight'][wi]) / compute_norm(prev_params['weight'][wi])

        for bi, [layer_bias, init_layer_bias] in enumerate(zip(params['bias'], initial_params['bias'])):

            init_norm = compute_norm(init_layer_bias)

            if init_norm.item() == 0.0 and i == 0:
                init_norm = torch.tensor([1.0], dtype=dtype, device=device)
            if init_norm.item() != 0.0:
                diff += compute_norm(layer_bias - init_layer_bias) / init_norm
            else:
                diff += compute_norm(layer_bias - prev_params['bias'][bi]) / compute_norm(prev_params['bias'][bi])

        parameters_diff[i+1] = diff

        prev_epoch  = epoch
        prev_params = net.network_parameters_log[prev_epoch]

    return parameter_epochs, parameters_diff

def compute_NTK_diff(net): 

    NTK_epochs = list(net.NTK_log.keys())
    n          = len(NTK_epochs)
    NTK_diff   = np.zeros((n,1))

    K0 = net.NTK_log[0]['NTK_matrix'][0]
    K0_norm = compute_norm(K0)

    for i,epoch in enumerate(NTK_epochs[1:]):
        K   = net.NTK_log[epoch]['NTK_matrix'][0]
        K_diff = K - K0 
        diff = compute_norm(K_diff) / K0_norm
        NTK_diff[i+1] = diff.detach().cpu().numpy()

    return NTK_epochs, NTK_diff

def compute_convergence_rate(net):

    NTK_epochs = list(net.NTK_log.keys())
    NTK_convergenceRate = []

    for epoch in NTK_epochs:
        NTK_convergenceRate_row = []
        for NTK_eigenvalues in net.NTK_log[epoch]['NTK_eigenvalues']:
            n = NTK_eigenvalues.shape[0]
            convergence_rate = np.sum(NTK_eigenvalues) / n
            NTK_convergenceRate_row.append(convergence_rate)

        NTK_convergenceRate.append(np.hstack(NTK_convergenceRate_row))

    NTK_convergenceRate = np.vstack(NTK_convergenceRate)

    return NTK_epochs, NTK_convergenceRate


def compute_adaptionWeights(net):
        
    epochs = list(net.NTK_log.keys())

    adaption_weights = []

    for epoch in epochs:
        lambda_adaptation = []
        NTK_ = net.NTK_log[epoch]['NTK_matrix']

        # update adaption terms
        K_trace = np.trace(NTK_[0])
        if len(NTK_) > 0:
            Krr_trace = np.trace(NTK_[1])
            # print(K_trace, Krr_trace)
            lambda_adaptation.append(K_trace / Krr_trace)
        if len(NTK_) > 1:
            Kuu_trace = np.trace(NTK_[2])
            # print(K_trace, Kuu_trace)
            lambda_adaptation.append(K_trace / Kuu_trace)
        if len(NTK_) > 2:
            Kii_trace = np.trace(NTK_[3])
            lambda_adaptation.append(K_trace / Kii_trace)

        # Clip adaption terms to not be negative
        lambda_adaptation = np.array(lambda_adaptation)
        if epoch > 0:
            for i,weight in enumerate(lambda_adaptation):
                if weight < 1.:
                    lambda_adaptation[i] = old_weights[i]
                    print("Negative weight, re-adjusting to previous weight")
        
        adaption_weights.append(lambda_adaptation)
        old_weights = lambda_adaptation

    adaption_weights = np.vstack(adaption_weights)
    return adaption_weights
