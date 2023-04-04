
import numpy as np

import matplotlib.pyplot as plt

def plot_NTK(net):

    eig_K_plot    = np.sort(np.real(net.lambda_K.detach().cpu().numpy()))[::-1]
    eig_K_uu_plot = np.sort(np.real(net.lambda_Kuu.detach().cpu().numpy()))[::-1]
    eig_K_rr_plot = np.sort(np.real(net.lambda_Krr.detach().cpu().numpy()))[::-1]

    ### PLOT Eigenvalue of NTK matrices
    fig, axs = plt.subplots(1,3, figsize=(23,6))

    axs[0].semilogx(eig_K_plot,      label=r'$\lambda_{K}$');     axs[0].set_title('Eigenvalue of K')
    axs[1].semilogx(eig_K_uu_plot,   label=r'$\lambda_{uu}$');    axs[1].set_title('Eigenvalue of {}'.format(r"$K_{uu}$"))
    axs[2].semilogx(eig_K_rr_plot,   label=r'$\lambda_{rr}$');    axs[2].set_title('Eigenvalue of {}'.format(r"$K_{rr}$"))

    for ax in axs:
        ax.legend()
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_yscale('log')
        ax.set_ylabel(r'$\lambda$')
        ax.set_xlabel(r'$Index$')