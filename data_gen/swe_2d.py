#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: radial dam break
==================================

Solve the 2D shallow water equations:

.. math::
    h_t + (hu)_x + (hv)_y = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y = 0 \\
    (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y = 0.

The initial condition is a circular area with high depth surrounded by lower-depth water.
The top and right boundary conditions reflect, while the bottom and left boundaries
are outflow.
"""

import numpy as np
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import depth, x_momentum, y_momentum, num_eqn
from clawpack import pyclaw
from tqdm import tqdm

def qinit_DamBreak(state,h_in=2.,h_out=1.,dam_radius=0.5):
    x0=0.
    y0=0.
    X, Y = state.p_centers
    r = np.sqrt((X-x0)**2 + (Y-y0)**2)

    state.q[depth     ,:,:] = h_in*(r<=dam_radius) + h_out*(r>dam_radius)
    state.q[x_momentum,:,:] = 0.
    state.q[y_momentum,:,:] = 0.

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def qinit_GaussianPulse(state, nx, ny, seed=None):
    if seed is not None:
        np.random.seed(seed)

    s = np.random.uniform(0.1, 0.5) # standard deviation

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    x, y = np.meshgrid(x, y) 
    u = gaus2d(x, y, sx=s, sy=s) # get 2D gaussian
    u = 0.5*(u / np.max(u)) # normalize the gaussian
    u = u + 1 # height field between 1 and 1.5
    state.q[depth     ,:,:] = u # set initial height
    state.q[x_momentum,:,:] = 0. # initial velocity is zero
    state.q[y_momentum,:,:] = 0. # initial velocity is zero


def qinit_RandomSines(state, nx, ny, J, lj, L, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    [xx, yy] = np.meshgrid(x, y)

    u = np.zeros((nx, ny))

    A = np.random.uniform(-0.1, 0.1, J)
    Kx = 2*np.pi*np.random.randint(1, lj, J)/L
    Ky = 2*np.pi*np.random.randint(1, lj, J)/L
    phi = np.random.uniform(0, 2*np.pi, J)

    for i in range(J):
        u = u + A[i]*np.sin(Kx[i] * xx + Ky[i] * yy + phi[i])
    u = u + 1
    state.q[depth     ,:,:] = u # set initial height
    state.q[x_momentum,:,:] = 0. # initial velocity is zero
    state.q[y_momentum,:,:] = 0. # initial velocity is zero

def setup():
    global outdir
    global seed

    rs = riemann.shallow_hlle_2D

    solver = pyclaw.ClawSolver2D(rs)
    solver.limiters = pyclaw.limiters.tvd.MC
    #solver.dimensional_split=1
    #solver = pyclaw.SharpClawSolver2D(rs)

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    solver.dt=0.00001
    solver.max_steps = 10000

    # Domain:
    xlower = -1
    xupper = 1
    mx = 256
    ylower = -1
    yupper = 1
    my = 256
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])

    state = pyclaw.State(domain,num_eqn)

    # Gravitational constant
    state.problem_data['grav'] = 1.0
    
    # Random initial state
    J = 5
    lj = 3
    L = 2

    #qinit_RandomSines(state, nx=mx, ny=my, J=J, lj=lj, L=L, seed=seed)
    qinit_GaussianPulse(state, nx=mx, ny=my, seed=seed)

    claw = pyclaw.Controller()
    claw.output_format = "hdf5"
    claw.tfinal = 2.0
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.num_output_times = 100
    claw.keep_copy = True
    claw.verbosity = 0

    return claw

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main

    split = 'valid'
    num_samples = 64
    
    if split == 'train':
        offset = 0 
    else:
        offset = 1000

    for i in tqdm(range(num_samples)):
        seed = i + offset
        outdir = f"swe_2d_gaussian_{split}_{seed}"

        output = run_app_from_main(setup)

    #outdir = "_output"
    #output = run_app_from_main(setup)