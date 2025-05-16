import lightning as L
import torch
from einops import rearrange

# custom imports
from modules.models.FNO1D import FNO1d_cond
from modules.models.FNO2D import FNO2d_cond
from modules.models.Unet import Unet_cond
from modules.models.Hamiltonian import Hamiltonian_Wrapper
from modules.models.NeuralFunctional import Neural_Functional
from common.utils import richardson_extrapolation, calculate_dH_du
from common.loss import ScaledLpLoss, PearsonCorrelationScore

class TrainModule(L.LightningModule):
    def __init__(self,
                 modelconfig: dict,
                 normalizer = None):
        '''
        args:
            modelconfig: dict, model configuration
        '''

        super().__init__()
        self.model_name = modelconfig["model_name"]
        self.lr = modelconfig["lr"]
        self.correlation = modelconfig["correlation"]
        self.mode = modelconfig['mode'] # normal or derivative or hamiltonian
        self.pde = modelconfig.get('pde', None)
        self.weight_decay = modelconfig.get('weight_decay', 0.0)
        self.derivative_mode = modelconfig.get('derivative_mode', 'central')
        self.derivative_order = modelconfig.get('derivative_order', 2)
        self.optimize_grad = modelconfig.get('optimize_grad', False)
        self.ablate_grad = modelconfig.get('ablate_grad', False)
        self.ablate_H = modelconfig.get('ablate_H', False)
        self.filter = modelconfig.get('filter', None)
        self.normalizer = normalizer

        self.criterion = ScaledLpLoss()
        self.correlation_criterion = PearsonCorrelationScore(reduce_batch=True)

        if self.model_name == "fno":
            fnoconfig = modelconfig["fno"]
            self.model = FNO1d_cond(**fnoconfig)
        elif self.model_name == "fno2d":
            fnoconfig = modelconfig["fno2d"]
            self.model = FNO2d_cond(**fnoconfig)
        elif self.model_name == "unet":
            unetconfig = modelconfig["unet"]
            self.model = Unet_cond(**unetconfig)
        elif self.model_name == "nf":
            nfconfig = modelconfig["nf"]
            self.model = Neural_Functional(nfconfig)
        else:
            raise ValueError("Model not found")

        if self.mode == "hamiltonian":
            self.model = Hamiltonian_Wrapper(self.model, 
                                             pde=self.pde, 
                                             derivative_mode=self.derivative_mode,
                                             derivative_order=self.derivative_order,
                                             optimize_grad=self.optimize_grad,
                                             ablate_H=self.ablate_H,
                                             ablate_grad=self.ablate_grad,
                                             filter=self.filter)

        print(f"Training: {self.model_name}, with mode: {self.mode}, and optimize_grad: {self.optimize_grad}")
        self.save_hyperparameters()

    def forward(self, u, x, cond=None, return_H=False, return_grad=False):
        if self.mode == 'hamiltonian':
            return self.model(u, x, cond, return_H, return_grad)
        return self.model(u, cond)

    def get_data_labels(self, u, idx, dt, mode='normal', dx=None):
        '''
        Get data and labels for training
        args:
            u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
            idx: shape (b,)
            dt: float (assumed constant)
            mode: str, training mode
            dx: shape (b, 1), grid spacing
        returns:
            data: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
            label: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c)
        '''
        b = u.shape[0]
        batch_range = torch.arange(b)
        data = u[batch_range, idx] # get u(t), shape (b, nx, 1) or (b, nx, ny, c)

        if mode == 'normal':
            label = u[batch_range, idx+1] # gets u(t+1), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        elif mode == "hamiltonian" or "derivative":
            if self.optimize_grad:
                label = calculate_dH_du(data, dx.unsqueeze(-1), pde=self.pde)
            else:
                label = richardson_extrapolation(u, idx, dt) # gets u'(t), shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
        else:
            raise ValueError("Mode not found")

        return data, label
    
    def inference_step(self, u_start, pred, dt, mode="normal", pred_cache=None):
        '''
        Steps u(t) to u(t+1), using u_start and model prediction
        args:
            u_start: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), u(t)
            pred: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), model prediction F(u(t), cond) = u'(t)
            dt: float, step size 
            mode: str, inference mode
            pred_cache: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), cached previous prediction u'(t-1), used in multistep methods

        returns:
            u_pred: shape (b, nx, 1) or (b, nx, ny, 1) or (b, nx, ny, c), prediction at t+1
        '''

        if mode == "normal":
            return pred # pred is already u(t+1)
        
        elif mode == "hamiltonian" or "derivative":
            # use adams bashforth to integrate u'(t), can use higher-order integrators
            if pred_cache is None: # need u'(t-1) for adams_bashforth
                return u_start + dt * pred # default to forward euler
            else:
                return u_start + dt * (3/2 * pred - 1/2 * pred_cache) # u(t+1) = u(t) + dt * (3/2 * u'(t) - 1/2 * u'(t-1))
        
        else:
            raise ValueError("Mode not found")
        
    def rollout(self, batch):
        u = batch['u']
        dt = batch['dt'][0] # assume constant dt across samples in batch. Shape (1,)
        x = batch['x'] # shape (b, nx, 1) or (b, nx, ny, 2)
        nt = u.shape[1]
        cond = batch.get('cond', None) # shape (b, cond_dim)

        u_input = u[:, 0]

        u_pred = torch.zeros_like(u) # shape (b, nt, nx, 1) or (b, nt, nx, ny, c)
        u_pred[:, 0] = u[:, 0] # set initial condition

        accumulated_loss = []
        at_correlation = False
        pred_cache = None 
        last_step = nt-1 # rollout to nt-1, since t = 0 is given.
  
        for i in range(0, last_step):
            if self.normalizer is not None:
                u_input = self.normalizer.normalize(u_input)

            pred = self.forward(u_input, x, cond)

            if self.optimize_grad:
                pred = self.model.poisson_bracket(pred, x, u_input) # get du_dt from dH_du

            u_input = self.inference_step(u_input,
                                            pred,
                                            dt,
                                            mode=self.mode,
                                            pred_cache=pred_cache,) # shape (b, nx, 1)

            if self.mode == "hamiltonian":
                u_input = u_input.detach() # detach next timestep to avoid backprop more than once through network, because of autoregressive rollout 

            if self.normalizer is not None:
                u_input = self.normalizer.denormalize(u_input)

            u_true = u[:, i+1]
            loss = self.criterion(u_input, u_true) # calculate loss
            u_pred[:, i+1] = u_input # save prediction
            accumulated_loss.append(loss.item())
            correlation = self.correlation_criterion(u_input, u_true) # calculate correlation
            
            if correlation < self.correlation and not at_correlation:
                correlation_time = float(i+1) # get time step at correlation
                at_correlation = True 
            
            if torch.isnan(u_input).any() or torch.isnan(loss).any() and not at_correlation:
                break

            pred_cache = pred # cache prediction for adams_bashforth
        
        if not at_correlation:
            correlation_time = nt-1 # didn't go below correlation threshold, therefore the time is the last step

        return accumulated_loss, correlation_time, u, u_pred
    
    def training_step(self, batch, batch_idx, eval=False):
        u = batch['u']
        dt = batch['dt'][0] # assume constant dt across samples in batch, shape (1,)
        x = batch['x'] # shape (b, nx, 1) or (b, nx, ny, 2)
        dx = batch.get('dx', None) # shape (b, 1)
        b = u.shape[0]
        nt = u.shape[1]

        if self.normalizer is not None:
            u = self.normalizer.normalize(u)

        t_idx = torch.randint(0, nt-1, (b,)) # shape (b,) get random start indexes.
        cond = batch.get('cond', None) # shape (b,) or (b, n_cond) depending on dimensionality of condition

        data, labels = self.get_data_labels(u, t_idx, dt, mode=self.mode, dx=dx) # slice data and labels with t_idx

        target = self.forward(data, x, cond) # shape (b, nx, 1) or (b, nx, ny, 1)
        loss = self.criterion(target, labels)

        if eval:
            return loss, data, labels, target

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, eval=False):
        if self.mode == "hamiltonian":
            torch.set_grad_enabled(True) # need to enable grad to backprop through network during inference
        
        if eval: 
            accumulated_loss, correlation_time, u, u_pred = self.rollout(batch)
            return accumulated_loss, correlation_time, u, u_pred
        else:
            accumulated_loss, correlation_time, _, _ = self.rollout(batch)
            avg_loss = sum(accumulated_loss) / len(accumulated_loss)
            self.log("rollout_loss", avg_loss, on_step=False, on_epoch=True)
            self.log("correlation_time", correlation_time, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        return [optimizer], [scheduler]