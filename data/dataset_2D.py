import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class PDEDataset2D(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 split: str,
                 resolution: Tuple[int, int, int],
                 t_start = 0,
                 t_end = -1,
                 data_key='u',
                 train_pct=1.0) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            split: [train, valid, test]
            resolution: resolution of the dataset [nt, nlat, nlon]
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.split = split
        self.pde = pde
        self.resolution = resolution
        self.nt, self.nx, self.ny = resolution
        self.train_pct = train_pct
        data = f[self.split]
        
        self.x = torch.tensor(np.array(data['x']), dtype=torch.float32) # nx, ny 2
        self.t = torch.tensor(np.array(data['t']), dtype=torch.float32) # nt

        self.u = data[data_key]
        self.n_samples = self.u.shape[0]

        if len(self.t.shape) > 1:
            self.t = self.t[0] # assume constant time across batch, take first one

        nt_data = self.t.shape[0]
        self.t_downsample = int(nt_data / self.nt) 
        self.t = self.t[::self.t_downsample] # downsample time
        self.dt = torch.tensor([self.t[1] - self.t[0]]) # shape (1)

        if t_end > 0:
            self.t_end = t_end
        else:
            self.t_end = len(self.t)

        self.t_start = t_start
        self.t = self.t[self.t_start:self.t_end] # truncate to t_start to t_end
        self.t_cutoff = int(self.train_pct * len(self.t)) # cutoff time for training

        nx_data = self.x.shape[0]
        ny_data = self.x.shape[1]
        self.x_downsample = int(nx_data / self.nx) 
        self.y_downsample = int(ny_data / self.ny)
        self.x = self.x[::self.x_downsample, ::self.y_downsample] # downsample x

        self.dx = torch.tensor([self.x[1, 0, 0] - self.x[0, 0, 0]]) # shape (1)

        print("Data loaded from: {}".format(path))
        print(f"PDE: {self.pde}, nt: {self.nt}, nx: {self.nx}, ny: {self.ny}, downsample t: {self.t_downsample}, downsample x: {self.x_downsample}, downsample y: {self.y_downsample}")
        print(f"Time ranges from {self.t[0]:.3f} to {self.t[-1]:.3f} = {self.dt.item():.3f} * {len(self.t)} dt * nt")
        print(f"Space ranged from {self.x[0, 0, 0]:.3f} to {self.x[-1, -1, 0]:.3f} = {self.dx.item():.3f} * {self.nx} dx * nx")
        if self.split == "train":
            print(f"Training with samples until: {self.t_cutoff}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            return_dict (dict): dictionary of data
        """
        u = self.u[idx] # shape (nt, nx, ny, c)
        if self.t_downsample > 1:
            u = u[::self.t_downsample] # truncate to nt by taking every t_downsample
        u = u[self.t_start:self.t_end] # truncate to t_start to t_end

        if self.x_downsample > 1:
            u = u[:, ::self.x_downsample, ::self.y_downsample] # downsample x
        
        if self.split == "train" and self.train_pct < 1.0:
            u = u[:self.t_cutoff]

        u = torch.from_numpy(u) # shape (nt, nx, ny, c)

        if self.pde == "swe":
            u[..., 1] = u[..., 1] / u[..., 0] # convert x-momentum to velocity
            u[..., 2] = u[..., 2] / u[..., 0] # convert y-momentum to velocity

        return_dict = {"u": u, "dt": self.dt, "x": self.x, "t": self.t, 'dx': self.dx}

        return return_dict