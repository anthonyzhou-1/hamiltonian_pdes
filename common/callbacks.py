import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from common.plotting import plot_result, plot_result_2d
from common.utils import calculate_Hamiltonian

class PlottingCallback(Callback):
    def on_validation_end(self, trainer, pl_module) -> None:     
        valid_loader = trainer.val_dataloaders
        mode = pl_module.mode
        pde = pl_module.pde
        nbatch = 2

        with torch.no_grad():
            batch = next(iter(valid_loader))
            batch = {k: v.to(pl_module.device) for k, v in batch.items()}

            accumulated_loss, correlation_time, result, result_pred = pl_module.validation_step(batch, 0, eval=True)

            plt.plot(accumulated_loss, label='Avg Validation Loss')
            plt.axvline(correlation_time, color='r', linestyle='--', label='Steps to 0.8 correlation')
            plt.xlabel('Time step')
            plt.ylabel('Validation Loss')
            plt.legend()
            plt.title(f"Mode: {mode}, Correlation time: {str(correlation_time)}")
            epoch = trainer.current_epoch
            path = trainer.default_root_dir + "/error_epoch-" + str(epoch) + ".png"
            
            plt.savefig(path)
            plt.close()

            if pde == "swe":
                H_result = result[:nbatch]
                H_pred = result_pred[:nbatch]
                dx = batch['dx'][:nbatch]
            else: # squeeze channel dim for 1D PDEs
                H_result = result[:nbatch, ..., -1]
                H_pred = result_pred[:nbatch, ..., -1]
                dx = batch['dx'][:nbatch]

            true_H = calculate_Hamiltonian(H_result, dx, pde=pde) 
            true_H = true_H.detach().cpu().numpy()
            pred_H = calculate_Hamiltonian(H_pred, dx, pde=pde) 
            pred_H = pred_H.detach().cpu().numpy()

            plt.plot(pred_H[0], label="Estimated_0", color='b', linestyle='--')
            plt.plot(true_H[0], label="True_0", color='k')
            plt.plot(pred_H[1], label="Estimated_1", color='red', linestyle='--')
            plt.plot(true_H[1], label="True_1", color='purple')

            plt.xlabel('Time step')
            plt.ylabel('Hamiltonian')
            plt.title('Hamiltonian over time')
            plt.legend()
            path = trainer.default_root_dir + "/hamiltonian_epoch-" + str(epoch) + ".png"
            plt.savefig(path)
            plt.close()

            if result is not None:
                path = trainer.default_root_dir + "/traj_epoch-" + str(epoch) + ".png"
                if len(result.shape) == 4: # b nt nx 1
                    plot_result(result, result_pred, path)
                else:
                    plot_result_2d(result, result_pred, path=path)