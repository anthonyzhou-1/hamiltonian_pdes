import torch 
import math 
import torch.nn.functional as F

def first_derivative(u, dx, order=2, mode="central"):
    '''
    Compute the first derivative of a 1D tensor
    Assume u is evenly spaced and periodic BCs
    args:
        u: shape (b, n) or (b, n, 1)
        dx: (b, 1) or (b, 1, 1)
    returns:
        du_dx: shape (b, n) or (b, n, 1)
    Centered difference at interior points https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Uses a one-sided difference at edges http://ftp.demec.ufpr.br/CFD/bibliografia/MER/Rahul_Bhattacharyya_2006.pdf if the signal is not periodic
    Smooth, Noise-Robust differentiation from http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    '''
    du_dx = torch.zeros_like(u, device=u.device)

    if order == 2:
        u_plus_1 = torch.roll(u, shifts=-1, dims=1) 
        u_minus_1 = torch.roll(u, shifts=1, dims=1) 
        if mode == "periodic" or mode == "periodic_enforced":
            if mode == "periodic_enforced":
                u_plus_1[:, -1] = u_plus_1[:, 0] # enforce periodicity
                u_minus_1[:, 0] = u_minus_1[:, -1] 
            du_dx = (u_plus_1 - u_minus_1) / (2*dx)
        elif mode == "smooth":
            u_plus_2 = torch.roll(u, shifts=-2, dims=1)
            u_minus_2 = torch.roll(u, shifts=2, dims=1)
            du_dx = (2*(u_plus_1 - u_minus_1) + u_plus_2 - u_minus_2) / (8*dx)
        else:
            du_dx_center = (u_plus_1 - u_minus_1) / (2*dx)
            du_dx[:, 1:-1] = du_dx_center[:, 1:-1] # b n-2
            du_dx[:, 0] = -1 * (3*u[:, 0] - 4*u[:, 1] + u[:, 2]) / (2*dx[:, 0]) # b 
            du_dx[:, -1] = (3*u[:, -1] - 4*u[:, -2] + u[:, -3]) / (2*dx[:, 0]) # b
    elif order == 4:
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        if mode == "periodic" or mode == "periodic_enforced":
            if mode == "periodic_enforced":
                u_plus_1[:, -1] = u_plus_1[:, 0] # enforce periodicity
                u_minus_1[:, 0] = u_minus_1[:, -1] 
                u_plus_2 = torch.roll(u_plus_1, shifts=-1, dims=1)
                u_minus_2 = torch.roll(u_minus_1, shifts=1, dims=1)
                u_plus_2[:, -1] = u_plus_2[:, 0] # enforce periodicity
                u_minus_2[:, 0] = u_minus_2[:, -1]
            du_dx = (-u_plus_2 + 8*u_plus_1 - 8*u_minus_1 + u_minus_2) / (12*dx)
        elif mode == "smooth":
            u_plus_3 = torch.roll(u_plus_1, shifts=-3, dims=1)
            u_minus_3 = torch.roll(u_minus_1, shifts=3, dims=1)
            du_dx = (5*(u_plus_1 - u_minus_1) + 4*(u_plus_2 - u_minus_2) + (u_plus_3 - u_minus_3)) / (32*dx)
        else:
            du_dx_center = (-u_plus_2 + 8*u_plus_1 - 8*u_minus_1 + u_minus_2) / (12*dx) 
            du_dx[:, 2:-2] = du_dx_center[:, 2:-2] # b n-4
            du_dx[:, 0] = -1 * (25*u[:, 0] - 48*u[:, 1] + 36*u[:, 2] - 16*u[:, 3] + 3*u[:, 4]) / (12*dx[:, 0])
            du_dx[:, 1] = -1 * (25*u[:, 1] - 48*u[:, 2] + 36*u[:, 3] - 16*u[:, 4] + 3*u[:, 5]) / (12*dx[:, 0])
            du_dx[:, -2] = (25*u[:, -2] - 48*u[:, -3] + 36*u[:, -4] - 16*u[:, -5] + 3*u[:, -6]) / (12*dx[:, 0])
            du_dx[:, -1] = (25*u[:, -1] - 48*u[:, -2] + 36*u[:, -3] - 16*u[:, -4] + 3*u[:, -5]) / (12*dx[:, 0])
    elif order == 6:
        # default to smooth differentiation
        u_plus_3 = torch.roll(u, shifts=-3, dims=1)
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        u_minus_3 = torch.roll(u, shifts=3, dims=1)
        u_plus_4 = torch.roll(u, shifts=-4, dims=1)
        u_minus_4 = torch.roll(u, shifts=4, dims=1)
        du_dx = (14*(u_plus_1 - u_minus_1) + 14*(u_plus_2 - u_minus_2) + 6*(u_plus_3 - u_minus_3) + (u_plus_4 - u_minus_4)) / (128*dx)
    elif order == 8:
        # default to smooth differentiation
        u_plus_5 = torch.roll(u, shifts=-5, dims=1)
        u_plus_4 = torch.roll(u, shifts=-4, dims=1)
        u_plus_3 = torch.roll(u, shifts=-3, dims=1)
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        u_minus_3 = torch.roll(u, shifts=3, dims=1)
        u_minus_4 = torch.roll(u, shifts=4, dims=1)
        u_minus_5 = torch.roll(u, shifts=5, dims=1)
        du_dx = (42*(u_plus_1 - u_minus_1) + 48*(u_plus_2 - u_minus_2) + 27*(u_plus_3 - u_minus_3) + 8*(u_plus_4 - u_minus_4) + (u_plus_5 - u_minus_5)) / (512*dx)
    else:
        raise ValueError("Order not found")
    return du_dx

def first_derivative_2d(u, dx, dy=None, order=2, mode="central", boundary=False):
    '''
    Compute the first derivative of a 2D tensor
    Assume u is evenly spaced and periodic BCs
    args:
        u: shape (b, n, m) or (b, n, m, 1)
        dx: (b, 1, 1) or (b, 1, 1, 1)
    returns:
        du_dx: shape (b, n, m) or (b, n, m, 1)
        du_dy: shape (b, n, m) or (b, n, m, 1)
    '''
    if dy == None:
        dy = dx # assume isotropic, uniform grid

    if order == 2:
        u_xplus_1 = torch.roll(u, shifts=-1, dims=1)
        u_xminus_1 = torch.roll(u, shifts=1, dims=1)
        u_yplus_1 = torch.roll(u, shifts=-1, dims=2)
        u_yminus_1 = torch.roll(u, shifts=1, dims=2)
        if mode == "central":
            du_dx = (u_xplus_1 - u_xminus_1) / (2*dx)
            du_dy = (u_yplus_1 - u_yminus_1) / (2*dx)
        elif mode == "smooth":
            u_xplus_2 = torch.roll(u, shifts=-2, dims=1)
            u_xminus_2 = torch.roll(u, shifts=2, dims=1)
            u_yplus_2 = torch.roll(u, shifts=-2, dims=2)
            u_yminus_2 = torch.roll(u, shifts=2, dims=2)
            du_dx = (2*(u_xplus_1 - u_xminus_1) + u_xplus_2 - u_xminus_2) / (8*dx)
            du_dy = (2*(u_yplus_1 - u_yminus_1) + u_yplus_2 - u_yminus_2) / (8*dy)

    elif order == 4:
        u_xplus_1 = torch.roll(u, shifts=-1, dims=1)
        u_xminus_1 = torch.roll(u, shifts=1, dims=1)
        u_yplus_1 = torch.roll(u, shifts=-1, dims=2)
        u_yminus_1 = torch.roll(u, shifts=1, dims=2)
        u_xplus_2 = torch.roll(u, shifts=-2, dims=1)
        u_xminus_2 = torch.roll(u, shifts=2, dims=1)
        u_yplus_2 = torch.roll(u, shifts=-2, dims=2)
        u_yminus_2 = torch.roll(u, shifts=2, dims=2)

        if mode == "central":
            du_dx = (-u_xplus_2 + 8*u_xplus_1 - 8*u_xminus_1 + u_xminus_2) / (12*dx)
            du_dy = (-u_yplus_2 + 8*u_yplus_1 - 8*u_yminus_1 + u_yminus_2) / (12*dy)
        elif mode == "smooth":
            u_xplus_3 = torch.roll(u, shifts=-3, dims=1)
            u_xminus_3 = torch.roll(u, shifts=3, dims=1)
            u_yplus_3 = torch.roll(u, shifts=-3, dims=2)
            u_yminus_3 = torch.roll(u, shifts=3, dims=2)
            du_dx = (5*(u_xplus_1 - u_xminus_1) + 4*(u_xplus_2 - u_xminus_2) + (u_xplus_3 - u_xminus_3)) / (32*dx)
            du_dy = (5*(u_yplus_1 - u_yminus_1) + 4*(u_yplus_2 - u_yminus_2) + (u_yplus_3 - u_yminus_3)) / (32*dy)
    
    elif order == 6:
        u_xplus_1 = torch.roll(u, shifts=-1, dims=1)
        u_xminus_1 = torch.roll(u, shifts=1, dims=1)
        u_yplus_1 = torch.roll(u, shifts=-1, dims=2)
        u_yminus_1 = torch.roll(u, shifts=1, dims=2)
        u_xplus_2 = torch.roll(u, shifts=-2, dims=1)
        u_xminus_2 = torch.roll(u, shifts=2, dims=1)
        u_yplus_2 = torch.roll(u, shifts=-2, dims=2)
        u_yminus_2 = torch.roll(u, shifts=2, dims=2)
        u_xplus_3 = torch.roll(u, shifts=-3, dims=1)
        u_xminus_3 = torch.roll(u, shifts=3, dims=1)
        u_yplus_3 = torch.roll(u, shifts=-3, dims=2)
        u_yminus_3 = torch.roll(u, shifts=3, dims=2)
        u_xplus_4 = torch.roll(u, shifts=-4, dims=1)
        u_xminus_4 = torch.roll(u, shifts=4, dims=1)
        u_yplus_4 = torch.roll(u, shifts=-4, dims=2)
        u_yminus_4 = torch.roll(u, shifts=4, dims=2)

        du_dx = (14*(u_xplus_1 - u_xminus_1) + 14*(u_xplus_2 - u_xminus_2) + 6*(u_xplus_3 - u_xminus_3) + (u_xplus_4 - u_xminus_4)) / (128*dx)
        du_dy = (14*(u_yplus_1 - u_yminus_1) + 14*(u_yplus_2 - u_yminus_2) + 6*(u_yplus_3 - u_yminus_3) + (u_yplus_4 - u_yminus_4)) / (128*dy)

    elif order == 8:
        u_xplus_1 = torch.roll(u, shifts=-1, dims=1)
        u_xminus_1 = torch.roll(u, shifts=1, dims=1)
        u_yplus_1 = torch.roll(u, shifts=-1, dims=2)
        u_yminus_1 = torch.roll(u, shifts=1, dims=2)
        u_xplus_2 = torch.roll(u, shifts=-2, dims=1)
        u_xminus_2 = torch.roll(u, shifts=2, dims=1)
        u_yplus_2 = torch.roll(u, shifts=-2, dims=2)
        u_yminus_2 = torch.roll(u, shifts=2, dims=2)
        u_xplus_3 = torch.roll(u, shifts=-3, dims=1)
        u_xminus_3 = torch.roll(u, shifts=3, dims=1)
        u_yplus_3 = torch.roll(u, shifts=-3, dims=2)
        u_yminus_3 = torch.roll(u, shifts=3, dims=2)
        u_xplus_4 = torch.roll(u, shifts=-4, dims=1)
        u_xminus_4 = torch.roll(u, shifts=4, dims=1)
        u_yplus_4 = torch.roll(u, shifts=-4, dims=2)
        u_yminus_4 = torch.roll(u, shifts=4, dims=2)
        u_xplus_5 = torch.roll(u, shifts=-5, dims=1)
        u_xminus_5 = torch.roll(u, shifts=5, dims=1)
        u_yplus_5 = torch.roll(u, shifts=-5, dims=2)
        u_yminus_5 = torch.roll(u, shifts=5, dims=2)

        du_dx = (42*(u_xplus_1 - u_xminus_1) + 48*(u_xplus_2 - u_xminus_2) + 27*(u_xplus_3 - u_xminus_3) + 8*(u_xplus_4 - u_xminus_4) + (u_xplus_5 - u_xminus_5)) / (512*dx)
        du_dy = (42*(u_yplus_1 - u_yminus_1) + 48*(u_yplus_2 - u_yminus_2) + 27*(u_yplus_3 - u_yminus_3) + 8*(u_yplus_4 - u_yminus_4) + (u_yplus_5 - u_yminus_5)) / (512*dy)

    # post-process boundaries (if not periodic)
    if boundary:
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        # assume y is periodic
    return du_dx, du_dy

def second_derivative(u, dx, order=2):
    '''
    Compute the second derivative of a 1D tensor
    Assume u is evenly spaced and periodic BCs
    args:
        u: shape (b, n) or (b, n, 1)
        dx: (b, 1) or (b, 1, 1)
    returns: 
        du2_dx2: shape (b, n) or (b, n, 1)
    Centered difference at interior points https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Use a one-sided difference at edges https://www.mech.kth.se/~ardeshir/courses/literature/fd.pdf
    '''
    du2_dx2 = torch.zeros_like(u, device=u.device)
    if order == 2:
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        du2_dx2_center = (u_plus_1 - 2*u + u_minus_1) / (dx**2)
        du2_dx2[:, 1:-1] = du2_dx2_center[:, 1:-1] # b n-2
        du2_dx2[:, 0] = (2*u[:, 0] - 5*u[:, 1] + 4*u[:, 2] - u[:, 3]) / ((dx[:, 0])**2) # b
        du2_dx2[:, -1] = (2*u[:, -1] - 5*u[:, -2] + 4*u[:, -3] - u[:, -4]) / ((dx[:, 0])**2) # b
    elif order == 4:
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        du2_dx2_center = (-u_plus_2 + 16*u_plus_1 - 30*u + 16*u_minus_1 - u_minus_2) / (12*dx**2)
        du2_dx2[:, 2:-2] = du2_dx2_center[:, 2:-2] # b n-4
        du2_dx2[:, 0] = (15/4*u[:, 0] - 77/6*u[:, 1] + 107/6 *u[:, 2] - 13*u[:, 3] + 61/12*u[:, 4] - 5/6*u[:, 5]) / ((dx[:, 0])**2) # b
        du2_dx2[:, 1] = (15/4*u[:, 1] - 77/6*u[:, 2] + 107/6 *u[:, 3] - 13*u[:, 4] + 61/12*u[:, 5] - 5/6*u[:, 6]) / ((dx[:, 0])**2) # b
        du2_dx2[:, -2] = (15/4*u[:, -2] - 77/6*u[:, -3] + 107/6 *u[:, -4] - 13*u[:, -5] + 61/12*u[:, -6] - 5/6*u[:, -7]) / ((dx[:, 0])**2) # b
        du2_dx2[:, -1] = (15/4*u[:, -1] - 77/6*u[:, -2] + 107/6 *u[:, -3] - 13*u[:, -4] + 61/12*u[:, -5] - 5/6*u[:, -6]) / ((dx[:, 0])**2) # b
    elif order == 6:
        u_plus_3 = torch.roll(u, shifts=-3, dims=1)
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        u_minus_3 = torch.roll(u, shifts=3, dims=1)
        du2_dx2 = (-(u_plus_3 + u_minus_3) + 5*(u_plus_2 + u_minus_2) + (u_plus_1 + u_minus_1) - 10*u) / (12*dx**2)
    elif order == 8:
        u_plus_4 = torch.roll(u, shifts=-4, dims=1)
        u_plus_3 = torch.roll(u, shifts=-3, dims=1)
        u_plus_2 = torch.roll(u, shifts=-2, dims=1)
        u_plus_1 = torch.roll(u, shifts=-1, dims=1)
        u_minus_1 = torch.roll(u, shifts=1, dims=1)
        u_minus_2 = torch.roll(u, shifts=2, dims=1)
        u_minus_3 = torch.roll(u, shifts=3, dims=1)
        u_minus_4 = torch.roll(u, shifts=4, dims=1)
        du2_dx2 = (-7*(u_plus_4 + u_minus_4) + 12*(u_plus_3 + u_minus_3) + 52*(u_plus_2 + u_minus_2) - 12*(u_plus_1 + u_minus_1) - 90*u) / (192*dx**2)
    else:
        raise ValueError("Order not found")
    return du2_dx2

def second_derivative_2d(u, dx):
    '''
    Compute the second derivative of a 2D tensor
    Assume u is evenly spaced and periodic BCs
    args:
        u: shape (b, n, m)
        dx: (b, 1, 1)
    returns:
        du2_dx2: shape (b, n, m)
        du2_dy2: shape (b, n, m)
    '''
    u_xplus_1 = torch.roll(u, shifts=-1, dims=1)
    u_xminus_1 = torch.roll(u, shifts=1, dims=1)
    du2_dx2 = (u_xplus_1 - 2*u + u_xminus_1) / (dx**2)

    u_yplus_1 = torch.roll(u, shifts=-1, dims=2)
    u_yminus_1 = torch.roll(u, shifts=1, dims=2)
    du2_dy2 = (u_yplus_1 - 2*u + u_yminus_1) / (dx**2)

    du2_dxdy = (torch.roll(u, shifts=(-1, -1), dims=(1, 2)) + torch.roll(u, shifts=(1, 1), dims=(1, 2)) - torch.roll(u, shifts=(1, -1), dims=(1, 2)) - torch.roll(u, shifts=(-1, 1), dims=(1, 2))) / (4*dx**2)

    return du2_dx2, du2_dy2, du2_dxdy