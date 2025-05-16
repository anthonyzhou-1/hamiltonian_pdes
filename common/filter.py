import collections
import typing as t
from math import ceil

import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndim
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d, conv1d
from torch.distributions import Normal

from einops import repeat

Param2 = collections.namedtuple('Param2', ('row', 'column'))
Polynom2 = collections.namedtuple('Polynom2', ('row_pows', 'column_pows', 'num_coeffs'))

_Param2Type = t.Union[Param2, t.Tuple[int, int]]
_ParamType = t.Union[int, _Param2Type]

_DIM = 2


class SGolayKernel2:
    """Computes two-dimensional kernel (weights) for Savitzky-Golay filter
    """

    def __init__(self, window_size: _Param2Type, poly_order: _Param2Type):
        self._window_size = Param2(*window_size)
        self._poly_order = Param2(*poly_order)

        self._kernel = None  # type: np.ndarray
        self.computed = False

    def __call__(self):
        self.compute()

    def compute(self, backend="scipy"):
        if self.computed:
            return

        polynom = self._make_polynom(self._poly_order)
        basis_matrix = self._make_basis_matrix(self._window_size, polynom)

        self._kernel = self._compute_kernel(self._window_size, basis_matrix)

        if backend == 'torch':
            # kernel is kx, ky -> 1, 1, kx, ky
            self._kernel = torch.from_numpy(self._kernel.copy()).unsqueeze(0).unsqueeze(0)
            self._kernel = self._kernel.float()

        self.computed = True

    @property
    def kernel(self) -> np.ndarray:
        """Returns 2D Savitzky-Golay kernel
        """
        self.compute()
        return self._kernel

    @staticmethod
    def _make_polynom(poly_order: Param2) -> Polynom2:
        """
        Creates 2-D polynom model (for example poly33):
            p = a00 + a10x + a01y + a20x^2 + a11xy + a02y^2 + a30x^3 + a21x^2y \
                + a12xy^2 + a03y^3
        """
        row_pows = []
        column_pows = []
        num_coeffs = 0

        for row in range(poly_order.row + 1):
            for column in range(poly_order.column + 1):
                if (row + column) > max(*poly_order):
                    continue

                row_pows.append(row)
                column_pows.append(column)

                num_coeffs += 1

        return Polynom2(row_pows, column_pows, num_coeffs)

    @staticmethod
    def _make_basis_matrix(window_size: Param2, poly: Polynom2) -> np.ndarray:
        """Creates basis polynomial matrix
        """
        basis_rows = window_size.row * window_size.column
        basis_columns = poly.num_coeffs

        basis_matrix = np.zeros((basis_rows, basis_columns))

        radius_row = (window_size.row - 1) // 2
        radius_column = (window_size.column - 1) // 2

        row_pows = np.array(poly.row_pows)
        column_pows = np.array(poly.column_pows)

        k = 0

        for row in range(-radius_row, radius_row + 1):
            for column in range(-radius_column, radius_column + 1):
                basis_matrix[k, :] = column ** column_pows * row ** row_pows
                k += 1

        return basis_matrix

    @staticmethod
    def _compute_kernel(window_size: Param2,
                        basis_matrix: np.ndarray) -> np.ndarray:
        """Computes filter 2D kernel via solving least squares problem
        """
        q, _ = la.qr(basis_matrix)

        iq = (window_size.row * window_size.column - 1) // 2
        kernel = q @ np.array(q[iq, :], ndmin=2).T
        kernel = np.fliplr(kernel.reshape(*window_size, order='F'))

        return kernel


class SGolayFilter2:
    """Two-dimensional Savitzky-Golay filter
    """

    def __init__(self, window_size: _ParamType, poly_order: _ParamType):
        
        self._window_size = self._canonize_param(
            'window_size', window_size, self._validate_window_size)
        self._poly_order = self._canonize_param(
            'poly_order', poly_order, self._validate_poly_order)

        self._kernel = SGolayKernel2(self._window_size, self._poly_order)

    def __call__(self, data: np.ndarray,
                 mode: str = 'reflect', 
                 cval: float = 0.0,
                 backend = 'scipy',
                 padding_mode='circular') -> np.ndarray:
        return self._filtrate(data, mode=mode, cval=cval, backend=backend, padding_mode=padding_mode)

    @property
    def window_size(self) -> Param2:
        return self._window_size

    @property
    def poly_order(self) -> Param2:
        return self._poly_order

    @property
    def kernel(self) -> SGolayKernel2:
        """Returns filter 2D kernel
        """
        return self._kernel

    @staticmethod
    def _canonize_param(name, value: _ParamType, validator) -> Param2:
        err = TypeError(
            'The parameter "{}" must be int scalar or Tuple[int, int]'.format(
                name))

        if isinstance(value, int):
            value = (value, value)

        if not isinstance(value, (list, tuple)):
            raise err
        if len(value) != _DIM:
            raise err
        if not all(isinstance(v, int) for v in value):
            raise err

        validator(value)

        return Param2(*value)

    @staticmethod
    def _validate_window_size(value):
        if not all(v >= 3 and bool(v % 2) for v in value):
            raise ValueError(
                'Window size values must be odd and >= 3 (Given: {})'.format(
                    value))

    @staticmethod
    def _validate_poly_order(value):
        if not all(v >= 1 for v in value):
            raise ValueError(
                'Polynom order values must be >= 1 (Given: {})'.format(value))

    def _filtrate(self, data: np.ndarray, mode, cval, backend = 'scipy', padding_mode='circular'):
        self._kernel.compute(backend)

        if backend == 'scipy':
            # data is nx, ny
            # kernel is kx, ky
            return ndim.correlate(data, self._kernel.kernel, mode=mode, cval=cval)
        elif backend == "torch":
            # data is b nx ny c
            # kernel is c_out c_in kx ky
            channel_dim = data.shape[-1]
            kernel_size = self._kernel.kernel.shape[-1] 
            padding = kernel_size // 2

            grouped_kernel = self._kernel.kernel.repeat(channel_dim, 1, 1, 1).to(data.device)

            data = data.permute(0, 3, 1, 2)
            data = F.pad(data, (padding, padding, padding, padding), mode=padding_mode)
            
            smoothed = F.conv2d(data, grouped_kernel, groups=channel_dim) # depthwise convolution
            smoothed = smoothed.permute(0, 2, 3, 1)
            return smoothed
        

def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_1d(img: torch.Tensor, sigma: float, reshape=True) -> torch.Tensor:
    # img in shape b nx 
    if reshape:
        img = img.unsqueeze(1)  # b 1 nx
    n_channels = img.shape[1]

    filter = gaussian_kernel_1d(sigma)  # kx
    padding = len(filter) // 2  # Ensure that image size does not change
    filter = repeat(filter, 'k -> b c k', c=n_channels, b=1).to(img.device)  # 1 c kx

    img = conv1d(img, weight=filter, padding=padding)
    if reshape:
        img = img.squeeze() # b nx
    return img 

def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    # img in shape b c nx ny
    n_channels = img.shape[1]
    
    filter = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    padding = len(filter) // 2  # Ensure that image size does not change
    filter_x = repeat(filter, 'k -> b c k 1', c=n_channels, b=1).to(img.device)  # 1 c kx 1
    filter_y = repeat(filter, 'k -> b c 1 k', c=n_channels, b=1).to(img.device)  # 1 c 1 ky

    # Convolve along columns and rows
    img = conv2d(img, weight=filter_x, padding=(padding, 0))
    img = conv2d(img, weight=filter_y, padding=(0, padding))
    return img  # Make 2D again