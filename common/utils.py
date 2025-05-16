import torch 
import yaml
import math
import numpy as np
from sympy import symbols, Poly
from einops import rearrange, repeat
from common.derivatives import second_derivative

# symbolic x for sympy
x_symbolic = symbols('x')

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def sinusoidal_embedding(timesteps, dim, max_period=10000, scale=1):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    timesteps = timesteps * scale # scale the timesteps since in our case they can be very small

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def richardson_extrapolation(u, idx, dt):
    '''
    Compute the Richardson extrapolation of a 1D tensor at timestep idx
    In effect, computing u'(t)|t=idx
        args:
        u: shape (b, nt, nx) or (b, nt, nx, ny) or (b, nt, nx, ny, c)
        idx: shape (b,)
        dt: scalar
    returns:
        du_dt: shape (b, nx) or (b, nx, ny) or (b, nx, ny, c)
    '''
    b = u.shape[0]
    nt = u.shape[1]
    batch_range = torch.arange(b)
    last_idx = nt-1 # nt is length of u, so nt-1 is the last index (i.e. u[:,nt] is out of bounds)

    is_not_zero = idx.all() # true if all elements are not zero, false if at least one element is zero
    is_not_one = (idx-1).all() # true if all elements are not one, false if at least one element is one
    is_not_n_minus_1 = (idx-(last_idx)).all() # true if all elements are not nt-1, false if at least one element is nt-1 
    is_not_n_minus_2 = (idx-(last_idx-1)).all() # true if all elements are not nt-2, false if at least one element is nt-2  
    # assume that idx is less than nt-1, so dont need to check if idx+1 is out of bounds 

    if is_not_zero and is_not_one and is_not_n_minus_2 and is_not_n_minus_1:
        # can compute batched Richardson extrapolation
        u_tminus1 = u[batch_range, idx-1]
        u_tminus2 = u[batch_range, idx-2]
        u_tplus1 = u[batch_range, idx+1]
        u_tplus2 = u[batch_range, idx+2]

        du_dt = 4/3 * (u_tplus1 - u_tminus1)/(2*dt) - 1/3 * (u_tplus2 - u_tminus2)/(4*dt)
    else:
        # compute elementwise Richardson extrapolation
        u_shape_no_time = u[:, 0].shape
        du_dt = torch.zeros(u_shape_no_time, device=u.device)
        for i in range(b):
            idx_i = idx[i]
            if idx_i == 0 or idx_i == 1: # use one-sided fourth order richardson extrapolation 
                # Kumar Rahul, S.N. Bhattacharyya, One-sided finite-difference approximations suitable for use with Richardson extrapolation
                # http://ftp.demec.ufpr.br/CFD/bibliografia/MER/Rahul_Bhattacharyya_2006.pdf

                u_t = u[i, idx_i]
                u_tplus1 = u[i, idx_i+1]
                u_tplus2 = u[i, idx_i+2]
                u_tplus3 = u[i, idx_i+3]
                u_tplus4 = u[i, idx_i+4]
                du_dt[i] = -1 * (25 * u_t - 48 * u_tplus1 + 36 * u_tplus2 - 16 * u_tplus3 + 3 * u_tplus4) / (12 * dt)
            elif idx_i == last_idx or idx_i == last_idx - 1: # use one-sided fourth order richardson extrapolation
                u_t = u[i, idx_i]
                u_tminus1 = u[i, idx_i-1]
                u_tminus2 = u[i, idx_i-2]
                u_tminus3 = u[i, idx_i-3]
                u_tminus4 = u[i, idx_i-4]
                du_dt[i] = (25 * u_t - 48 * u_tminus1 + 36 * u_tminus2 - 16 * u_tminus3 + 3 * u_tminus4) / (12 * dt)
            else:
                u_tminus1 = u[i, idx_i-1]
                u_tminus2 = u[i, idx_i-2]
                u_tplus1 = u[i, idx_i+1]
                u_tplus2 = u[i, idx_i+2]
                du_dt[i] = 4/3 * (u_tplus1 - u_tminus1)/(2*dt) - 1/3 * (u_tplus2 - u_tminus2)/(4*dt)

    return du_dt

def integrate_uniform(summand, dx, quadrature="trapezoidal"):
    '''
    Integrate a 1D tensor with uniform spacing
    args:
        summand: shape (b, nt, nx) 
        dx: shape (b, 1) 
        quadrature: str, quadrature method
    returns:
        integral: shape (b, nt)
    '''
    if quadrature == "riemann":
        summand = summand[:, :, :-1] # shape (b, nt, nx-1), left riemann sum
        h = dx * torch.sum(summand, dim = 2) # shape (b, nt)
    elif quadrature == "trapezoidal":
        summand_trap = 2*summand
        summand_trap[:, :, 0] = summand_trap[:, :, 0] / 2
        summand_trap[:, :, -1] = summand_trap[:, :, -1] / 2
        h = dx/2 * torch.sum(summand_trap, dim = 2) # shape (b, nt)
    else:
        raise ValueError("Quadrature not found")
    return h

def integrate_uniform_2d(summand, dx, quadrature="trapezoidal"):
    '''
    Integrate a 1D tensor with uniform spacing
    args:
        summand: shape (b, nt, nx, ny) 
        dx: shape (b, 1) 
        quadrature: str, quadrature method
    returns:
        integral: shape (b, nt)
    '''

    if quadrature == "trapezoidal": # assume uniform spacing
        weights = torch.ones(summand.shape[-2:], device=summand.device) # shape (nx, ny)
        weights[0, 0] = 1/4
        weights[0, -1] = 1/4
        weights[-1, 0] = 1/4
        weights[-1, -1] = 1/4
        weights[0, 1:-1] = 1/2
        weights[-1, 1:-1] = 1/2
        weights[1:-1, 0] = 1/2
        weights[1:-1, -1] = 1/2
        integrand = torch.einsum('btij,ij->bt', summand, weights) # shape (b, nt)
        integrand = dx**2 * integrand
        return integrand
    else:
        raise ValueError("Quadrature not found")

def calculate_dH_du(u, dx, pde="kdv", order=2):
    # u in shape (b nx)
    # dx in shape (b 1)

    if pde == "kdv":
        u_xx = second_derivative(u, dx, order=order)
        dH_du = -1/2 * u**2 - u_xx # shape (b, nx, 1)
    elif pde == "burgers":
        dH_du = -1/2 * u**2
    elif pde == "advection":
        c = -1 # advection speed/direction
        dH_du = -1 * c * u
    elif pde == "swe":
        # u in shape (b nx ny 3) height, velocity_x, velocity_y
        g = 1.0
        h = u[..., 0]
        u_x = u[..., 1]
        u_y = u[..., 2]

        dH_dh = 1/2 * (u_x**2 + u_y**2) + g * h
        dH_dux = h * u_x
        dH_duy = h * u_y
        
        dH_du = torch.stack([dH_dh, dH_dux, dH_duy], dim=-1) # shape (b, nx, ny, 3)
    return dH_du

def calculate_Hamiltonian(u, dx, pde="kdv", quadrature="trapezoidal", order=4):
    # u in shape (b nt nx)
    # dx in shape (b 1)

    if pde == "kdv":        
        # batch second derivative calculation across time dimension
        u_batch = rearrange(u, 'b nt nx -> (b nt) nx') # shape (b nt, nx)
        dx_batch = repeat(dx, 'b 1 -> (b nt) 1', nt = u.shape[1]) # shape (b nt, 1)
        u_xx = second_derivative(u_batch, dx_batch, order=order) # shape (b nt, nx)
        u_xx = rearrange(u_xx, '(b nt) nx -> b nt nx', b = u.shape[0]) # shape (b, nt, nx)

        summand = -1/6 * torch.pow(u,3) - 1/2 * u * u_xx # shape (b, nt, nx)
    elif pde == "burgers":
        summand = -1/6 * torch.pow(u,3)
    elif pde == "advection":
        summand = -1/2 * torch.pow(u,2) 
    elif pde == "swe":
        # u in shape (b nt nx ny 3) height, velocity_x, velocity_y
        g = 1.0 # data was generated with gravity = 1
        h = u[..., 0] # b nt nx ny
        u_x = u[..., 1] 
        u_y = u[..., 2]

        summand = 1/2 * h * (u_x**2 + u_y**2) + 1/2 * g * h**2 # shape (b, nt, nx, ny)
        h = integrate_uniform_2d(summand, dx, quadrature=quadrature) # shape (b, nt)
        return h
    else:
        raise ValueError("PDE not found")
    
    h = integrate_uniform(summand, dx, quadrature=quadrature) # shape (b, nt)
    return h
    
def get_u(p, x, use_sympy = False):
    # x in shape (nx)
    # p in shape (n)
    # returns u(x) = p[0]x^(n-1) + p[1]x^(n-2) + ... + p[n-1]x + p[n]
    if use_sympy:
        polynomial = Poly(p, x_symbolic) 

        eval_polynomial = lambda xx: float(polynomial.subs(x_symbolic, xx))
        vectorize_eval = np.vectorize(eval_polynomial) # not the most efficient since using sympy
        u = vectorize_eval(x)
        u = torch.tensor(u)
    
    else:
        u = torch.zeros_like(x, device=x.device)
        for i in range(len(p)):
            u += p[i] * torch.pow(x, len(p)-i-1)
    
    return u

def F_l(p, x):
    # x in shape (nx)
    # p in shape (n)
    # returns F[u] = \int u(x)g(x)dx, where u(x) = p[0]x^(n-1) + p[1]x^(n-2) + ... + p[n-1]x + p[n], and g(x)=x^2 is fixed

    g = x_symbolic**2
    polynomial = Poly(p, x_symbolic) * g
    integral = polynomial.integrate(x_symbolic)

    F = float(integral.subs(x_symbolic, x[-1])) - float(integral.subs(x_symbolic, x[0]))

    return torch.tensor([F])

def F_nl(p, x):
    # x in shape (nx)
    # p in shape (n)
    # returns F[u] = \int (u(x))^3dx, where u(x) = p[0]x^(n-1) + p[1]x^(n-2) + ... + p[n-1]x + p[n]

    polynomial = Poly(p, x_symbolic)
    polynomial = polynomial**3
    integral = polynomial.integrate(x_symbolic)

    F = float(integral.subs(x_symbolic, x[-1])) - float(integral.subs(x_symbolic, x[0]))

    return torch.tensor([F])

