import torch 
import torch.nn as nn
from einops import rearrange
from modules.models.Kernels import Siren_Kernel, MLP_Kernel

class Integral_Kernel_Functional(nn.Module):
    def __init__(self, 
                 kernel, 
                 kernel_type = "linear", 
                 quadrature="trapezoidal",
                 nx = 256,
                 proj_dim = -1,
                 func_dim = 1,
                 coord_dim = 1):
        super(Integral_Kernel_Functional, self).__init__()
        # linear kernel, k(y) (batch, pos_dim) -> (batch, f_dim) 
        # nonlinear kernel k(y, u(y)) (batch, pos_dim + f_dim) -> (batch, f_dim)
        # kernel is used to approximate F[u] = \int k(y, u(y))u(y)dy

        self.kernel = kernel
        self.kernel_type = kernel_type
        self.quadrature = quadrature # riemann or trapezoidal
        self.nx = nx

        # use nx to cache quadrature weights since assume inputs are all of same discretization
        # alternatively can calculate on the fly for arbitrarily discretized inputs
        if self.quadrature == "trapezoidal":
            weights = torch.ones((nx, nx)) # shape (nx, ny)
            weights[0, 0] = 1/4
            weights[0, -1] = 1/4
            weights[-1, 0] = 1/4
            weights[-1, -1] = 1/4
            weights[0, 1:-1] = 1/2
            weights[-1, 1:-1] = 1/2
            weights[1:-1, 0] = 1/2
            weights[1:-1, -1] = 1/2
            self.register_buffer("quad_weights", weights)
        
        if proj_dim != -1:
            self.func_proj = nn.Linear(func_dim, proj_dim)
            self.coord_proj = nn.Linear(coord_dim, proj_dim//2) # use fourier features
            self.coord_scale = 2 * torch.pi
        
        self.project = False if proj_dim == -1 else True
        
    def forward(self, u, x):
        '''
        args:
            u: shape (b, nx, c) or (b, nx, ny, c), nodal values
            x: shape (b, nx, 1) or (b, nx, ny, 2), coordinates
        returns:
            F[u] = \int k(y, u(y))u(y)dy, shape (b, 1)
            Optionally, lift y -> P(y) and u(y) -> P(u(y)) by projection
        '''

        if len(u.shape) == 3: # 1D kernel integration. Assume nodal values have a channel dimension of 1
            dx = x[:, 1] - x[:, 0] # b 1

            if self.project:
                # lift u and x to proj_dim
                u = self.func_proj(u) # b nx 1 -> b nx proj_dim
                x = self.coord_scale * self.coord_proj(x) # b nx 1 -> b nx proj_dim/2
                x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # b nx proj_dim

            kernel_input = x  # b nx 1
            if self.kernel_type == "nonlinear": 
                kernel_input = torch.cat([x, u], dim = -1)  # b nx 1+1
            kernel_input = rearrange(kernel_input, 'b nx c -> (b nx) c') # (b nx) 1 or (b nx) 1+1

            if self.kernel_type == "global":
                kernel_output = self.kernel(kernel_input, u) # (b nx) 1
            else:
                kernel_output = self.kernel(kernel_input) # (b nx) 1

            kernel_output = rearrange(kernel_output, '(b nx) c -> b nx c', b = u.shape[0])
            # in theory, take dot product between kernel_output and u, but we eventually sum over last dims so doesn't matter
            summand = kernel_output * u # b nx c

            if self.quadrature == "riemann":
                summand = summand[:, :-1] # b nx-1 c, left riemann sum
                functional = dx * torch.sum(summand, dim = 1) # b 1
            elif self.quadrature == "trapezoidal":
                summand_trap = 2*summand # b nx c
                summand_trap[:, 0] = summand_trap[:, 0] / 2
                summand_trap[:, -1] = summand_trap[:, -1] / 2
                functional = dx/2 * torch.sum(summand_trap, dim = 1) # b 1
            else:
                raise ValueError("Quadrature not found")
            
        else:  # 2D kernel integration.
            dx = x[:, 1, 0, 0] - x[:, 0, 0, 0] # b 
            dx = dx.unsqueeze(-1) # b 1

            if self.project:
                u = self.func_proj(u) # b nx ny c -> b nx ny proj_dim
                x = self.coord_scale * self.coord_proj(x) # b nx ny 2 -> b nx ny proj_dim/2
                x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # b nx ny proj_dim

            kernel_input = x  # b nx ny 2, coordinates
            if self.kernel_type == "nonlinear": 
                kernel_input = torch.cat([x, u], dim = -1) # b nx ny 2+c
            kernel_input = rearrange(kernel_input, 'b nx ny c -> (b nx ny) c') # (b nx ny) 2 or (b nx ny) 2+c

            if self.kernel_type == "global":
                kernel_output = self.kernel(kernel_input, u) # (b nx ny) c
            else:
                kernel_output = self.kernel(kernel_input) # (b nx ny) c
            
            kernel_output = rearrange(kernel_output, '(b nx ny) c -> b nx ny c', b = u.shape[0], nx=self.nx)
            summand = kernel_output * u # b nx ny c

            if self.quadrature == "trapezoidal":
                quad_weights = self.quad_weights
                functional = torch.einsum('bijc,ij->bc', summand, quad_weights) # shape (b, c)
                functional = dx**2 * functional
                functional = torch.sum(functional, dim = 1).unsqueeze(-1) # shape (b, 1)
            else:
                raise ValueError("Quadrature not found")

        return functional

class Neural_Functional(nn.Module):
    def __init__(self, 
                 nfconfig): 
        super(Neural_Functional, self).__init__()

        self.kernel_name = nfconfig["kernel_name"]
        self.quadrature = nfconfig["quadrature"]
        self.kernel_type = nfconfig["kernel_type"]
        self.func_dim = nfconfig["func_dim"]
        self.coord_dim = nfconfig["coord_dim"]
        self.spatial_dim = nfconfig.get("spatial_dim", 1)
        self.proj_dim = nfconfig.get("proj_dim", -1)

        if self.proj_dim != -1: # lift coordinate and function dim to proj_dim
            self.coord_dim = self.proj_dim
            self.func_dim = self.proj_dim
        
        if self.kernel_name == "mlp":
            kernelconfig = nfconfig["mlp"]
            kernel_model = MLP_Kernel
        elif self.kernel_name == "siren":
            kernelconfig = nfconfig["siren"]
            kernel_model = Siren_Kernel
        else:
            raise ValueError("Kernel not found")
        
        if self.kernel_type == "nonlinear" and kernelconfig["film_type"] == "disabled": 
            # use concatenation of coordinates and function values
            self.kernel = kernel_model(in_features=self.func_dim + self.coord_dim, 
                                out_features=self.func_dim,
                                **kernelconfig)
        else:
            self.kernel = kernel_model(in_features=self.coord_dim, 
                                out_features=self.func_dim,
                                **kernelconfig)
                
        self.IKF = Integral_Kernel_Functional(self.kernel, 
                                              self.kernel_type, 
                                              self.quadrature, 
                                              self.spatial_dim, 
                                              self.proj_dim,
                                              nfconfig["func_dim"],
                                              nfconfig["coord_dim"])

        print(f"Neural Functional: {self.kernel_name}, with quadrature: {self.quadrature}, and kernel type: {self.kernel_type}, and proj_dim: {self.proj_dim}")
    
    def forward(self, u, x, cond=None):
        '''
        args:
            u: shape (b, nx, c) or (b, nx, ny, c), nodal values
            x: shape (b, nx, 1) or (b, nx, ny, 2), coordinates
        returns:
            F[u(y)] = \int k(y, u(y))u(y)dy, shape (b, 1)
        '''
        return self.IKF(u, x) # shape (b, 1)
    
    def get_derivative(self, pred, u, x):
        '''
        args:
            pred: shape (b, 1), prediction F[u]
            u: shape (b, nx, c) or (b, nx, ny, c), nodal values
        returns:
            gradient: shape (b, nx, c) or (b, nx, ny, c), dH_pred/du
        '''

        # use autograd to get dF/du
        gradient = torch.autograd.grad(pred, u, grad_outputs=torch.ones_like(pred), create_graph=True)[0] # b nx c or b nx ny c
        
        # re-scale gradients from integral approximation
        if self.quadrature == "trapezoidal": # scale gradients back since trapezoidal rule scales the summands
            if len(u.shape) == 3: # 1D integration
                dx = x[:, 1] - x[:, 0] # b 1
                gradient = gradient / dx.unsqueeze(-1)
                gradient[:, 0] = gradient[:, 0] * 2
                gradient[:, -1] = gradient[:, -1] * 2
            else: # 2D integration
                dx = x[:, 1, 0, 0] - x[:, 0, 0, 0] # b
                dx = dx.unsqueeze(-1) # b 1
                dx_sq = dx**2
                quad_weights = self.IKF.quad_weights
                inv_quad_weights = 1 / quad_weights

                gradient = gradient / dx_sq.unsqueeze(-1).unsqueeze(-1) # b nx ny c
                gradient = torch.einsum('bijc,ij->bijc', gradient, inv_quad_weights) # b nx ny c

        elif self.quadrature == "riemann":
            dx = x[:, 1] - x[:, 0] # b 1
            gradient = gradient / dx.unsqueeze(-1)
            gradient[:, -1] = gradient[:, -2] # gradient is zero for last node, therefore set to previous node
            # this is because the last nodal position is not included in the left riemann sum, and therefore is not used in the forward pass/loss

        return gradient