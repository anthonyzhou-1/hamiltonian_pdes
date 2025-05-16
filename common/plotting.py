import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

def plot_result(traj, traj_pred, path):
    # result is a trajectory in shape (b, nt, nx, 1)
    idx = 0
    vmin = torch.amin(traj[idx, ..., -1]).item()
    vmax = torch.amax(traj[idx, ..., -1]).item()
    u = traj[idx, ..., -1].detach().cpu() # shape (nt, nx)
    u_pred = traj_pred[idx, ..., -1].detach().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))

    axs[0].imshow(u, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
    axs[0].set_title('Ground Truth')
    
    axs[1].imshow(u_pred, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
    axs[1].set_title('Prediction')

    im2= axs[2].imshow(torch.abs(u - u_pred), cmap='hot', origin='lower')
    axs[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axs[2])

    plt.savefig(path)
    plt.close()
    return 

def plot_result_2d(u_true, u_pred=None, n_t=5, path=None, channel=0):
    u_true = u_true[0].detach().cpu()
    u_pred = u_pred[0].detach().cpu() if u_pred is not None else None
    
    if len(u_true.shape) > 3:
        u_true = u_true[..., channel] # get the first channel
        u_pred = u_pred[..., channel] if u_pred is not None else None
        
    # u in shape nt nx ny 
        
    vmin = torch.min(u_true)
    vmax = torch.max(u_true)

    n_skip = u_true.shape[0] // n_t 
    u_downs = u_true[::n_skip]

    if u_pred is not None:
        u_pred_downs = u_pred[::n_skip]
        fig, ax = plt.subplots(n_t, 2, figsize=(8, 4*n_t))
        if n_t == 1:
            ax = [ax]
        for j in range(2):
            for i in range(n_t):
                ax[i][j].set_axis_off()
                if j == 0:
                    velocity = u_downs[i] 
                else:
                    velocity = u_pred_downs[i]

                im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
                ax[i][j].title.set_text(f'Timestep {i*n_skip}')
            ax[0][j].title.set_text(f'Ground Truth' if j == 0 else f'Prediction')
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))
        if n_t == 1:
            ax = [ax]
        for i in range(n_t):
            ax[i].set_axis_off()
            velocity = u_downs[i] 

            im = ax[i].imshow(velocity, vmin=vmin, vmax=vmax, cmap='inferno')
            ax[i].title.set_text(f'Timestep {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)