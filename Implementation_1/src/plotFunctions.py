
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from NTK_helper import compute_parameter_diff, compute_NTK_diff, compute_convergence_rate


def plot_results1D(xplot, u_pred, u_exact, train_losses):
    fig = plt.figure(1, (23,8), layout='tight')
    gs  = fig.add_gridspec(2,2)

    axs = []
    # Plot 1 - predict 
    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(xplot, u_exact, label=r'$u_{exact}$')
    ax0.plot(xplot, u_pred,  label=r'$u_{pred}$')
    ax0.set_title('Exact vs. neural network prediction')
    ax0.set_ylabel(r'$u$')
    ax0.set_xlabel(r'$x$')
    ax0.legend()
    axs.append(ax0)

    # Plot 2 - error plot
    pointWise_err = u_exact - u_pred

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(xplot, pointWise_err)
    ax1.set_title('Error')
    ax1.set_ylabel('Point-wise difference')
    ax1.set_xlabel(r'$x$')
    axs.append(ax1)

    # Plot 3 - train losses plot
    ax2 = fig.add_subplot(gs[1,:])
    ax2.semilogy(train_losses)
    ax2.set_title('Training loss over time')
    ax2.set_ylabel(r'Loss per epoch')
    ax2.set_xlabel(r'$Epoch$')
    axs.append(ax2)

    return fig, axs

def plot_results2D(xplot, u_pred, Intervals, train_losses): 


    fig = plt.figure(1, (23,8), layout='tight')
    gs = fig.add_gridspec(2,3)

    ### Plot 1 - solution profile
    ax0 = fig.add_subplot(gs[0,:2])
    ax0.set_xlabel(r't [s]')
    ax0.set_ylabel(r'$x$')
    
    # plot result
    h = ax0.imshow(u_pred, interpolation='nearest', cmap='rainbow',
    extent=[xplot[:,1].min(), xplot[:,1].max(), xplot[:,0].min(), xplot[:,0].max()],
    origin='lower', aspect='auto')

    # add colorbar
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ### Plot 2 - training loss
    ax1 = fig.add_subplot(gs[0,2])
    ax1.semilogy(train_losses)
    ax1.set_ylabel(r'loss per epoch')
    ax1.set_xlabel(r'$Epoch$')

    ### Plot 3 - solution slice at different domain intervals
    Intervals = [0.25, 0.5, 0.75]

    for i, t_idx in enumerate(Intervals):
        idx = (np.abs(xplot[:,1] - t_idx)).argmin()
        ax2 = fig.add_subplot(gs[1,i])
        ax2.plot(xplot[:,0], u_pred[:,idx])
        ax2.set_title(f't = {xplot[idx, 1]:1g} ')

def plot_NTK(net, fig=None, axs=None):

    n_lambda = len(net.NTK_log[0]['NTK_eigenvalues']) 
    epochs = list(net.NTK_log.keys())
    last_epoch = epochs[-1]

    # create figure
    if fig == None and axs == None:
        if n_lambda == 4:
            fig, axs = plt.subplots(1,4, figsize=(23,6))
        elif n_lambda == 3:
            fig, axs = plt.subplots(1,3, figsize=(23,6))

    # set array
    if n_lambda >= 3:
        lambda_K   = net.NTK_log[last_epoch]['NTK_eigenvalues'][0].cpu().numpy()
        lambda_Krr = net.NTK_log[last_epoch]['NTK_eigenvalues'][1].cpu().numpy()
        lambda_Kuu = net.NTK_log[last_epoch]['NTK_eigenvalues'][2].cpu().numpy()
    if n_lambda >= 4:
        lambda_Kii = net.NTK_log[last_epoch]['NTK_eigenvalues'][3].cpu().numpy()


    if hasattr(net, 'lambda_K'):
        eig_K_plot    = np.sort(np.real(lambda_K))[::-1]
        axs[0].semilogx(eig_K_plot,      label=r'$\lambda_{K}$');     axs[0].set_title('Eigenvalue of K')
    if hasattr(net, 'lambda_Kuu'):
        eig_K_uu_plot    = np.sort(np.real(lambda_Kuu))[::-1]
        axs[1].semilogx(eig_K_uu_plot,   label=r'$\lambda_{uu}$');    axs[1].set_title('Eigenvalue of {}'.format(r"$K_{uu}$"))
    if hasattr(net, 'lambda_Krr'):
        eig_K_rr_plot    = np.sort(np.real(lambda_Krr))[::-1]
        axs[2].semilogx(eig_K_rr_plot,   label=r'$\lambda_{rr}$');    axs[2].set_title('Eigenvalue of {}'.format(r"$K_{rr}$"))
    if hasattr(net, 'lambda_Kii'):
        eig_K_ii_plot    = np.sort(np.real(lambda_Kii))[::-1]
        axs[3].semilogx(eig_K_ii_plot,   label=r'$\lambda_{ii}$');    axs[3].set_title('Eigenvalue of {}'.format(r"$K_{ii}$"))

    for ax in axs:
        ax.legend()
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_yscale('log')
        ax.set_ylabel(r'$\lambda$')
        ax.set_xlabel(r'$Index$')

    return fig, axs

def plot_param_ntk_diff(net, fig = None, axs=None):

    if fig == None and axs == None:
        fig, axs = plt.subplots(1,2, figsize=(18,6))

    fig.suptitle('Network parameter and NTK eigenvalues change')

    parameter_epoch, parameter_diff = compute_parameter_diff(net)
    NTK_epoch, NTK_diff             = compute_NTK_diff(net)

    # convert to numpy for plotting
    parameter_diff  = parameter_diff.detach().cpu().numpy()
    NTK_diff        = NTK_diff.detach().cpu().numpy()

    axs[0].plot(parameter_epoch,parameter_diff, label=f'width={net.neurons}');    
    axs[1].plot(NTK_epoch, NTK_diff,            label=f'width={net.neurons}');   

    for ax in axs:
        # ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel(r'$Epoch$')
    axs[0].set_ylabel(r'$\frac{||\theta - \theta(0)||^{2}}{||\theta(0)||^{2}}$')
    axs[1].set_ylabel(r'$\frac{||K(n) - K(0)||^{2}}{||K(0)||^{2}}$')

    return fig, axs

def plot_NTK_change(net, fig=None, axs= None):

    if fig == None and axs == None:
        fig, axs = plt.subplots(1,2, figsize=(18,6))

    fig.suptitle('NTK kernel matrix K change over time')

    NTK_epochs = list(net.NTK_log.keys())

    for epoch in NTK_epochs:
        if epoch == 0:
            eig_K       = net.NTK_log[epoch]['NTK_eigenvalues'][0]
            eig_K_plot  = np.sort(np.real(eig_K.detach().cpu().numpy()))[::-1]
            axs.semilogy(eig_K_plot,   label=f'epoch={epoch}'); 
        elif epoch == NTK_epochs[-1]:
            eig_K       = net.NTK_log[epoch]['NTK_eigenvalues'][0]
            eig_K_plot  = np.sort(np.real(eig_K.detach().cpu().numpy()))[::-1]
            axs.semilogy(eig_K_plot,   label=f'epoch={epoch}'); 

    axs.set_xscale('log')
    axs.set_xlabel(r'$Epoch$')
    axs.legend()

    return fig, axs


# Plot convergence rate for all matrices

def plot_convergence_rate(net, fig=None, axs=None):

    if fig == None and axs == None:
        fig, axs = plt.subplots(1,2, figsize=(18,6))

    fig.suptitle('Convergence rate: ' + r'$c_{K}= $' + r'$\frac{Tr(K_{i})}{n}$')

    NTK_epochs, NTK_convergenceRate = compute_convergence_rate(net)
    
    # convert to numpy for plotting
    NTK_convergenceRate  = np.real(NTK_convergenceRate.detach().cpu().numpy())
    n       = NTK_convergenceRate.shape[1]
    ylabels = [r'$c_{K}$', r'$c_{K_{uu}}$', r'$c_{K_{rr}}$', r'$c_{K_{ii}}$']

    for i in range(n):
        axs.semilogy(NTK_epochs, NTK_convergenceRate[:,i],   label=ylabels[i]); 

    # axs.set_xscale('log')
    axs.set_xlabel(r'$Epoch$')
    axs.legend()

    return fig, axs
