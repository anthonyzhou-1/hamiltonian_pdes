import torch 
import torch.nn as nn
from common.derivatives import first_derivative, first_derivative_2d
from common.filter import SGolayFilter2

class Hamiltonian_Wrapper(nn.Module):

    def __init__(self, 
                 model, 
                 pde, 
                 derivative_mode="central", 
                 derivative_order = 2, 
                 optimize_grad=False, 
                 ablate_H=False,
                 ablate_grad=False,
                 filter=None):
        super(Hamiltonian_Wrapper, self).__init__()
        self.model = model
        self.pde = pde 
        self.derivative_mode = derivative_mode
        self.optimize_grad = optimize_grad
        self.derivative_order = derivative_order
        self.ablate_H = ablate_H
        self.ablate_grad = ablate_grad
        self.filer = filter

        if self.filter is not None:
            self.filter = SGolayFilter2(window_size=15, poly_order=3)
        
        print(f"Hamiltonian Wrapper: PDE: {self.pde}, Derivative Mode: {self.derivative_mode}, Derivative Order: {self.derivative_order}, Optimize Gradient: {self.optimize_grad}")
        print(f"Ablate H: {self.ablate_H}, Ablate Gradient: {self.ablate_grad}")

    def forward(self, u, x, cond=None, return_H=False, return_grad=False):
        '''
        args:
            u: shape (b, nx, c) or (b, nx, ny, c), nodal values
            x: shape (b, nx, 1) or (b, nx, ny, 2), coordinates
            return_H: bool, return H or not
            return_grad: bool, return gradient of H
        returns:
            du_dt: shape (b, nx, c) or (b, nx, ny, c), du_dt
            H: shape (b, 1), Hamiltonian
            gradient: shape (b, nx, c) or (b, nx, ny, c), gradient of H
        '''
        if self.ablate_H:
            return self.model(u, cond)

        u.requires_grad = True # for computing dH/du w/ autograd
        if self.ablate_grad:
            H = self.model(u, cond)
        else:
            H = self.model(u, x, cond)
        gradient = self.get_gradient(H, u, x)

        if self.optimize_grad: 
            return gradient
        
        du_dt = self.poisson_bracket(gradient, x, u)
        
        if return_grad:
            return du_dt, gradient
        elif return_H:
            return du_dt, H
        else:
            return du_dt
    
    def get_gradient(self, H, u, x):
        # if custom autograd function is defined
        if hasattr(self.model, 'get_derivative'):
            gradient = self.model.get_derivative(H, u, x) # b nx c or b nx ny c
        else:
            gradient = torch.autograd.grad(H, u, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        
        return gradient
    
    def poisson_bracket(self, gradient, x, u=None):
        '''
        args:
            gradient: shape (b, nx, c) or (b, nx, ny, c), dH/du
            x: shape (b, nx, 1) or (b, nx, ny, 2), coordinates
            u: shape (b, nx, c) or (b, nx, ny, c), nodal values, optional
        returns:
            pb: shape (b, nx, c)
        '''
        
        if self.pde == "kdv" or self.pde == "burgers" or self.pde == "advection": 
            dx = x[:, 1] - x[:, 0] # b 1
            dx = dx.unsqueeze(-1) # b 1 1

            pb = first_derivative(gradient, dx, mode=self.derivative_mode, order=self.derivative_order) 
        elif self.pde == "swe":
            # gradient has three channels: dH/dh, dH/du, dH/dv
            # Use the defined poisson bracket to compute dh/dt, du/dt, dv/dt
            # dh/dt = d/dx(dH/du) + d/dy(dH/dv)
            # du/dt = -q(dH/dv) + d/dx(dH/dh)
            # dv/dt = q(dH/du) + d/dy(dH/dh)
            # q = dv/dx - du/dy, vorticity
            dx = x[:, 1, 0, 0] - x[:, 0, 0, 0] # b 
            dx = dx.unsqueeze(-1).unsqueeze(-1) # b 1 1 

            dH_dh = gradient[..., 0]
            dH_du = gradient[..., 1]
            dH_dv = gradient[..., 2]
            u_x = u[..., 1]
            u_y = u[..., 2]

            du_dx, du_dy = first_derivative_2d(u_x, dx, order=self.derivative_order, mode=self.derivative_mode)
            dv_dx, dv_dy = first_derivative_2d(u_y, dx, order=self.derivative_order, mode=self.derivative_mode)

            q = dv_dx - du_dy

            dx_dH_du, dy_dH_du = first_derivative_2d(dH_du, dx, order=self.derivative_order, mode=self.derivative_mode)
            dx_dH_dv, dy_dH_dv = first_derivative_2d(dH_dv, dx, order=self.derivative_order, mode=self.derivative_mode)
            dx_dH_dh, dy_dH_dh = first_derivative_2d(dH_dh, dx, order=self.derivative_order, mode=self.derivative_mode)

            dh_dt = - dx_dH_du - dy_dH_dv
            du_dt = q * dH_dv - dx_dH_dh
            dv_dt = -q * dH_du - dy_dH_dh

            pb = torch.stack([dh_dt, du_dt, dv_dt], dim=-1) # b nx ny c
        else:
            raise ValueError("PDE not found")
        
        if self.filter is not None:
            pb = self.filter(pb, backend='torch')

        return pb

    