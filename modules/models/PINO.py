import torch
from common.derivatives import first_derivative_2d

# Implement Physics-Informed loss for SWE (Shallow Water Equations)

def SWE_Loss(data, pred, dx, dt, order=8):
    """
    2D shallow water equations:

        h_t + (hu)_x + (hv)_y = 0 \\
        (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y = 0 \\
        (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y = 0.

    Given a vector pred = [n_x, n_y, 3], with 2 components [u, v, h], we want its residual to be zero.
    We assume data is u(t) and pred is u'(t+1)
    """
    g = 1 # SWE was solved w/ graviational constant of 1

    # Continuity equation: h_t + (hu)_x + (hv)_y = 0

    h_pred = pred[..., 0]  # h(t+1), shape (b, nx, ny)
    h_true = data[..., 0]  # h(t), shape (b, nx, ny)

    dh_dt = (h_pred - h_true)/dt  # dh/dt, shape (b, nx, ny)
    hu_pred = h_pred * pred[..., 1]  # hu(t+1), shape (b, nx, ny)
    hv_pred = h_pred * pred[..., 2]  # hv(t+1), shape (b, nx, ny)

    dhu_dx, dhu_dy = first_derivative_2d(hu_pred, dx, order=order)
    dhv_dx, dhv_dy = first_derivative_2d(hv_pred, dx, order=order)

    mass_conservation = dh_dt + dhu_dx + dhv_dy  # continuity equation, shape (b, nx, ny)
    mass_conservation_loss = torch.mean(mass_conservation**2)  # shape (1), squared to prevent negative values

    # Momentum equations: 
    # (u)_t + u*(u)_x + v*(u)_y + g*(h)_x = 0
    # (v)_t + u*(v)_x + v*(v)_y + g*(h)_y = 0

    u_true = data[..., 1]  # u(t), shape (b, nx, ny)
    u_pred = pred[..., 1]  # u(t+1), shape (b, nx, ny)
    v_true = data[..., 2]  # v(t), shape (b, nx, ny)
    v_pred = pred[..., 2]  # v(t+1), shape (b, nx, ny)

    du_dt = (u_pred - u_true) / dt  # du/dt, shape (b, nx, ny)
    du_dx, du_dy = first_derivative_2d(u_pred, dx, order=order)
    dh_dx, dh_dy = first_derivative_2d(h_pred, dx, order=order)

    momentum_conservation_x = du_dt + u_pred * du_dx + v_pred * du_dy + g * dh_dx  # momentum conservation in x, shape (b, nx, ny)

    dv_dt = (v_pred - v_true) / dt  # dv/dt, shape (b, nx, ny)
    dv_dx, dv_dy = first_derivative_2d(v_pred, dx, order=order)

    momentum_conservation_y = dv_dt + u_true * dv_dx + v_pred * dv_dy + g * dh_dy  # momentum conservation in y, shape (b, nx, ny)

    momentum_conservation = momentum_conservation_x**2 + momentum_conservation_y**2  # total momentum conservation, shape (b, nx, ny)
    momentum_conservation_loss = torch.mean(momentum_conservation)  # shape (1)

    total_loss = mass_conservation_loss + momentum_conservation_loss  # total loss, shape (b)
    return total_loss
