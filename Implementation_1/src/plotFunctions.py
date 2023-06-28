import torch 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


from NTK_helper import compute_parameter_diff, compute_NTK_diff, compute_convergence_rate


def plot_results1D(xplot, u_pred, u_exact, train_losses, loss_labels=None):
    fig = plt.figure(1, (23,8), layout='tight')
    gs  = fig.add_gridspec(2,2)

    axs = []
    # Plot 1 - predict 
    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(xplot, u_exact, label=r'$u_{exact}$')
    ax0.plot(xplot, u_pred, '-', label=r'$u_{pred}$')
    ax0.set_title('Exact vs. neural network prediction')
    ax0.set_ylabel(r'$u$')
    ax0.set_xlabel(r'$x$')
    ax0.legend()
    axs.append(ax0)

    # Plot 2 - error plot
    pointWise_err = np.abs(u_exact - u_pred)
    error_u_l2    = np.linalg.norm(u_exact - u_pred,2) / np.linalg.norm(u_exact,2)

    ax1 = fig.add_subplot(gs[0,1])
    ax1.semilogy(xplot, pointWise_err)
    ax1.set_title('Absolute error \n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}')
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
    
    if loss_labels != None:
        ax2.legend(loss_labels)

    return fig, axs

#### 2D 
def predict2D(net, x_i):
    return net(x_i)

def copy_weights(net, epoch):
    i = 0
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.from_numpy(net.network_parameters_log[epoch]['weight'][i]).to(net.device).requires_grad_()
            module.bias.data   = torch.from_numpy(net.network_parameters_log[epoch]['bias'][i]).to(net.device).requires_grad_()
            i += 1

def plot_results2D(net, xplot, u_exact, Intervals, train_losses, loss_labels=None): 

    epochs = list(net.network_parameters_log.keys())
    copy_weights(net, epochs[-1])

    X = np.zeros((xplot[1].shape[0], xplot[0].shape[0], len(xplot)), dtype=np.float32)

    for i, t_i in enumerate(xplot[1]):
        t_i = np.ones(xplot[0].shape)*t_i
        X[i] = np.hstack([xplot[0], t_i])

    X = torch.from_numpy(X).to(net.device)

    make_prediction = torch.vmap(predict2D, in_dims=(None, 0))

    u_pred = make_prediction(net, X) 
    u_pred = np.hstack([u_pred_i.detach().cpu().numpy() for u_pred_i in u_pred])
    
    fig = plt.figure(1, (23,8), layout='tight')
    gs = fig.add_gridspec(3,4)

    ### Plot 1 - solution profile
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.set_xlabel(r't [s]')
    ax1.set_ylabel(r'$x$')
    ax1.set_title('Predicted u')
    
    # plot result
    h = ax1.imshow(u_pred, interpolation='nearest', cmap='rainbow',
        extent=[xplot[1].min(), xplot[1].max(), xplot[0].min(), xplot[0].max()],
        origin='lower', aspect='auto')

    # add colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ### Plot 2 - point wise difference
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_xlabel(r't [s]')
    ax2.set_ylabel(r'$x$')

    error_u     = np.abs((u_exact - u_pred))
    error_u_l2  = np.linalg.norm(u_exact - u_pred,2) / np.linalg.norm(u_exact,2)
    ax2.set_title('Absolute error \n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}')

    # plot result
    h = ax2.imshow(error_u, interpolation='nearest', cmap='rainbow',norm=matplotlib.colors.LogNorm(),
        extent=[xplot[1].min(), xplot[1].max(), xplot[0].min(), xplot[0].max()],
        origin='lower', aspect='auto')

    # add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ### Plot 3 - solution slice at different domain intervals
    ax3s = []
    for i, t_idx in enumerate(Intervals):
        closest_neighbor = np.abs(xplot[1] - t_idx)
        if closest_neighbor.min() > 0.1:
            print("warning gap to closest neighbor is larger than 0.1")
        
        idx = (closest_neighbor).argmin()
        ax3 = fig.add_subplot(gs[1,i])
        ax3.plot(xplot[0], u_pred[:,idx],  label=r'$u_{pred}$' )
        ax3.plot(xplot[0], u_exact[:,idx], label=r'$u_{exact}$')
        error_u_l2  = np.linalg.norm(u_exact[:,idx] - u_pred[:,idx],2) / np.linalg.norm(u_exact[:,idx],2)
        ax3.set_title(f't = {t_idx}\n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}') 
        ax3.legend()

        # append
        ax3s.append(ax3)

    ### Plot 4 - training loss
    ax4 = fig.add_subplot(gs[2,:])
    ax4.semilogy(train_losses)
    ax4.set_ylabel(r'loss per epoch')
    ax4.set_xlabel(r'$Epoch$')
    ax4.set_title('Training loss')

    if loss_labels != None:
        ax4.legend(loss_labels)
    
    return fig, [ax1, ax2, ax3s, ax4]

### ANIMATE FUNCTION

def plot_results2D_animate(net, xplot, u_exact, Intervals, train_losses, loss_labels=None, ani_interval=100): 

    epochs = list(net.network_parameters_log.keys())

    # make dataset for predictions
    X = np.zeros((xplot[1].shape[0], xplot[0].shape[0], len(xplot)), dtype=np.float32)

    for i, t_i in enumerate(xplot[1]):
        t_i = np.ones(xplot[0].shape)*t_i
        X[i] = np.hstack([xplot[0], t_i])
    
    X = torch.from_numpy(X.astype(np.float32)).to(net.device)

    make_prediction = torch.vmap(predict2D, in_dims=(None, 0))

    N = len(epochs)

    if N // ani_interval in [0,1]:
        ani_interval = 1
    
    U_PRED      = []
    plot_epochs = []
    # Predict
    for i in range(0, N, ani_interval):
        copy_weights(net, epochs[i])
        u_pred = make_prediction(net, X) 
        u_pred = np.hstack([u_pred_i.detach().cpu().numpy() for u_pred_i in u_pred])
        U_PRED.append(u_pred)
        plot_epochs.append(epochs[i])

    U_PRED = np.stack(U_PRED)

    ## MAKE INITIAL PLOTS
    fig = plt.figure(1, (23,8), layout='tight')
    gs = fig.add_gridspec(3,4)

    # plot 1
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.set_xlabel(r't [s]')
    ax1.set_ylabel(r'$x$')
    ax1.set_title('Predicted u')

    h1 = ax1.imshow(U_PRED[0], interpolation='nearest', cmap='rainbow',
        extent=[xplot[1].min(), xplot[1].max(), xplot[0].min(), xplot[0].max()],
        origin='lower', aspect='auto', animated=True)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = fig.colorbar(h1, cax=cax1)

    # plot 2
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_xlabel(r't [s]')
    ax2.set_ylabel(r'$x$')

    # plot initial
    error_u     = np.abs((u_exact - U_PRED[0]))
    error_u_l2  = np.linalg.norm(u_exact - U_PRED[0],2) / np.linalg.norm(u_exact,2)
    ax2.set_title('Absolute error \n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}')

    h2 = ax2.imshow(error_u, interpolation='nearest', cmap='rainbow', norm=matplotlib.colors.LogNorm(),
        extent=[xplot[1].min(), xplot[1].max(), xplot[0].min(), xplot[0].max()],
        origin='lower', aspect='auto', animated=True)

    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    # add colorbar
    cb2 = fig.colorbar(h2, cax=cax2)

    # plot 3
    axs3 = [fig.add_subplot(gs[1,i]) for i in range(len(Intervals))]
    closest_neighbors = []
    updated_lines = []
    for i, [t_idx, ax] in enumerate(zip(Intervals, axs3)):
        closest_neighbor = np.abs(xplot[1] - t_idx)
        if closest_neighbor.min() > 0.1:
            print("warning gap to closest neighbor is larger than 0.1")
        
        ax.set_title(f't = {t_idx}')
        idx = (closest_neighbor).argmin()
        # plot initial
        line = ax.plot(xplot[0], np.stack([u_exact[:,idx], U_PRED[0][:,idx]]).T, animated=True)
        ax.legend([r'$u_{pred}$', r'$u_{exact}$'])

        updated_lines.append(line[1])
        closest_neighbors.append(idx)

    # plot 4
    ax4 = fig.add_subplot(gs[2,:])

    # plot 4
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][:train_losses[0].shape[0]]
    ax4.set_prop_cycle(plt.cycler(color=colors))
    ax4.semilogy(train_losses, alpha=0.4)
    ax4.set_ylabel(r'loss per epoch')
    ax4.set_xlabel(r'$Epoch$')
    ax4.set_title('Training loss')

    if loss_labels != None:
        ax4.legend(loss_labels)

    def update(frame):
        # for each frame, update the data stored on each artist.
        epoch = plot_epochs[frame]
        fig.suptitle(f'Epoch: {epoch}')

        print(epoch, frame)

        ### Plot 1 - solution profile     
        # add colorbar
        if epoch != 0:
            h1.set_data(U_PRED[frame])  
            h1.norm.autoscale(h1._A)
            h1.colorbar.update_normal(h1.colorbar.mappable)

            ### Plot 2 - point wise difference
            error_u     = np.abs((u_exact - U_PRED[frame]))
            error_u_l2  = np.linalg.norm(u_exact - U_PRED[frame],2) / np.linalg.norm(u_exact,2)
            ax2.set_title('Absolute error \n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}')

            h2.set_data(error_u)  
            h2.norm.autoscale(h2._A)
            h2.colorbar.update_normal(h2.colorbar.mappable)

            ### Plot 3 - solution slice at different domain intervals
            i = 0
            for idx, line in zip(closest_neighbors,updated_lines):
                error_u_l2  = np.linalg.norm(u_exact[:,idx] - U_PRED[frame][:,idx],2) / np.linalg.norm(u_exact[:,idx],2)
                line.set_ydata(U_PRED[frame][:,idx])
                axs3[i].set_title(f't = {Intervals[i]}\n' +  r'$L^{2}$ error = '  + f'{error_u_l2:.3E}') 
                i += 1
            ### Plot 4 - training loss
            ax4.semilogy(train_losses[:epoch], linewidth=3)

            fig.canvas.draw()
            fig.canvas.flush_events()
       
        return (h1, h2, *updated_lines)
    
    ani = animation.FuncAnimation(fig, func=update, frames=len(plot_epochs), blit=True, interval=1, repeat=False)

    return fig, ani

### NEURAL NET PLOTS

def plot_NTK(net, fig=None, axs=None):

    n_lambda = len(net.NTK_log[0]['NTK_eigenvalues']) 
    epochs = list(net.NTK_log.keys())

    # create figure
    if fig == None and axs == None:
        if n_lambda == 4:
            fig, axs = plt.subplots(1,4, figsize=(23,6))
        elif n_lambda == 3:
            fig, axs = plt.subplots(1,3, figsize=(23,6))

    # set array

    for epoch in epochs:
        if epoch == 0 or (epoch % 10000 == 0) or epoch == epochs[-1]:
            for i, entry in enumerate(net.NTK_log[epoch]['NTK_eigenvalues']):
                if isinstance(entry, torch.Tensor):
                    net.NTK_log[epoch]['NTK_eigenvalues'][i] = entry.cpu().numpy()
            
            if n_lambda >= 3:
                lambda_K   = net.NTK_log[epoch]['NTK_eigenvalues'][0]
                lambda_Krr = net.NTK_log[epoch]['NTK_eigenvalues'][1]
                lambda_Kuu = net.NTK_log[epoch]['NTK_eigenvalues'][2]
            if n_lambda >= 4:
                lambda_Kii = net.NTK_log[epoch]['NTK_eigenvalues'][3]

            # adjust linewidth to highlight final and begin NTK
            if epoch == 0 or epoch == epochs[-1]:
                linewidth  = 2
            else:
                linewidth = 0.5
                
            if n_lambda >= 3:
                eig_K_plot    = np.sort(np.real(lambda_K))[::-1]
                axs[0].semilogx(eig_K_plot,      linewidth = linewidth, label=r'$\lambda_{K}$' + f'| Epoch: {epoch}');     
                axs[0].set_title('Eigenvalue of K')

                eig_K_uu_plot    = np.sort(np.real(lambda_Kuu))[::-1]
                axs[1].semilogx(eig_K_uu_plot,   linewidth = linewidth, label=r'$\lambda_{uu}$'+ f'| Epoch: {epoch}');    
                axs[1].set_title('Eigenvalue of {}'.format(r"$K_{uu}$"))

                eig_K_rr_plot    = np.sort(np.real(lambda_Krr))[::-1]
                axs[2].semilogx(eig_K_rr_plot,   linewidth = linewidth, label=r'$\lambda_{rr}$'+ f'| Epoch: {epoch}');    
                axs[2].set_title('Eigenvalue of {}'.format(r"$K_{rr}$"))

            if n_lambda >= 4:
                eig_K_ii_plot    = np.sort(np.real(lambda_Kii))[::-1]
                axs[3].semilogx(eig_K_ii_plot,   linewidth = linewidth, label=r'$\lambda_{ii}$'+ f'| Epoch: {epoch}');    
                axs[3].set_title('Eigenvalue of {}'.format(r"$K_{ii}$"))

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
    NTK_diff        = NTK_diff

    axs[0].plot(parameter_epoch,parameter_diff, label=f'width={net.neurons}');    
    axs[1].plot(NTK_epoch, NTK_diff,            label=f'width={net.neurons}');   

    for ax in axs:
        # ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel(r'$Epoch$')
    axs[0].set_ylabel(r'$\frac{||\theta - \theta(0)||^{2}}{||\theta(0)||^{2}}$', size=14)
    axs[1].set_ylabel(r'$\frac{||K(n) - K(0)||^{2}}{||K(0)||^{2}}$', size=14)

    return fig, axs

def plot_NTK_change(net, fig=None, axs= None, c='b', plot_intervals=False):

    if fig == None and axs == None:
        fig, axs = plt.subplots(1,2, figsize=(18,6))

    fig.suptitle('NTK kernel matrix K change over time')
    NTK_epochs = list(net.NTK_log.keys())

    initial_NTK = NTK_epochs[0]

    # initial NTK
    eig_K       = net.NTK_log[initial_NTK]['NTK_eigenvalues'][0]
    if torch.is_tensor(eig_K):
        eig_K = eig_K.detach().cpu().numpy()
    eig_K_plot  = np.sort(np.real(eig_K))[::-1]
    axs.semilogy(eig_K_plot, color=c, linewidth=4, label=f'epoch={initial_NTK}'); 

    # Final NTK
    final_NTK   = NTK_epochs[-1]
    eig_K       = net.NTK_log[final_NTK]['NTK_eigenvalues'][0]
    if torch.is_tensor(eig_K):
        eig_K = eig_K.detach().cpu().numpy()
    eig_K_plot  = np.sort(np.real(eig_K))[::-1]
    axs.semilogy(eig_K_plot, color='r', linewidth=4, label=f'epoch={final_NTK}'); 

    if plot_intervals:
        for epoch in NTK_epochs[1:-1]:
            if epoch % 10000 == 0:
                eig_K       = net.NTK_log[epoch]['NTK_eigenvalues'][0]
                if torch.is_tensor(eig_K):
                    eig_K = eig_K.detach().cpu().numpy()
                eig_K_plot  = np.sort(np.real(eig_K))[::-1]
                axs.semilogy(eig_K_plot, '--', alpha=0.8, label=f'epoch={epoch}'); 

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
    NTK_convergenceRate  = np.real(NTK_convergenceRate)
    n       = NTK_convergenceRate.shape[1]
    ylabels = [r'$c_{K}$', r'$c_{K_{uu}}$', r'$c_{K_{rr}}$', r'$c_{K_{ii}}$']

    for i in range(n):
        axs.semilogy(NTK_epochs, NTK_convergenceRate[:,i],   label=ylabels[i]); 
        # axs.plot(NTK_epochs, NTK_convergenceRate[:,i],   label=ylabels[i]); 

    # axs.set_xscale('log')
    axs.set_ylabel(r'$c_{K}$')
    axs.set_xlabel(r'$Epoch$')
    axs.legend()

    return fig, axs
