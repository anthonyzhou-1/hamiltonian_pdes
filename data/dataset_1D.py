import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
import pickle as pkl 

class PDEDataset1D(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 split: str,
                 resolution: Tuple[int, int],
                 start=0.0,
                 data_key='pde_400-256',
                 load_memory=True,
                 H_path=None,
                 H_filter=0.5,
                 train_pct=1.0,
                 norm_x = False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            split: [train, valid, test]
            resolution: resolution of the dataset [nt, nx]
            norm_dims: normalize x and t
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.split = split
        self.pde = pde
        self.resolution = resolution
        self.nt, self.nx = resolution
        self.start = start
        self.time_start = int(start * self.nt) # start time index
        self.load_memory = load_memory
        self.H_path = H_path
        self.H_filter = H_filter
        data = f[self.split]
        self.train_pct = train_pct
        self.norm_x = norm_x
        
        self.x = torch.tensor(np.array(data['x']), dtype=torch.float32) # nx
        self.t = torch.tensor(np.array(data['t']), dtype=torch.float32) # nt

        # optionally load dataset into a torch tensor
        if self.load_memory:
            self.u = torch.tensor(np.array(data[data_key]), dtype=torch.float32) # b nt nx
            f.close()
        else:
            self.u = data[data_key]

        # Optionally filter the samples by lowest variance in Hamiltonian. 
        # This is because some samples have numerical viscosity that may not conserve the Hamiltonian even if the PDE does.
        # This is unavoidable due to shock formation, but we can filter out the worst cases.
        if H_path is not None:
            with open(H_path, 'rb') as h_file:
                self.H_list = pkl.load(h_file)
            self.H_list = self.H_list[split]
            self.n_samples = int(len(self.H_list) * H_filter)
            self.H_list = self.H_list[:self.n_samples]
        else:
            self.n_samples = self.u.shape[0]

        if len(self.t.shape) > 1:
            self.t = self.t[0] # assume constant time across batch, take first one
        if len(self.x.shape) > 1:
            self.x = self.x[0]

        self.t = self.t - self.t[0] # start time from zero 
        self.x = self.x - self.x[0] # start x from zero

        if self.norm_x:
            self.scale_factor = self.x[-1] / 2 
            self.x = self.x / self.scale_factor # normalize x to [0, 2]
            self.x = self.x - 1 # normalize x to [-1, 1]
        else:
            self.scale_factor = torch.tensor([1.0])

        nt_data = self.t.shape[0]
        self.t_downsample = int(nt_data / self.nt) 
        self.t = self.t[::self.t_downsample] # downsample time
        self.t = self.t[self.time_start:]

        self.train_cutoff = int(len(self.t) * self.train_pct)

        self.dt = torch.tensor([self.t[1] - self.t[0]]) # shape (1)
        self.dx = torch.tensor([self.x[1] - self.x[0]]) # shape (1)

        print("Data loaded from: {}".format(path))
        print(f"PDE: {self.pde}, dx: {self.dx.item():.3f}, nt: {self.nt}, nx: {self.nx}, downsample: {self.t_downsample}")
        print(f"Time ranges from {self.t[0]:.3f} to {self.t[-1]:.3f} = {self.dt.item():.3f} * {len(self.t)} dt * nt")
        print(f"Space ranges from {self.x[0]:.3f} to {self.x[-1]:.3f} = {self.dx.item():.3f} * {len(self.x)} dx * nx")
        if self.split == "train":
            print(f"Training with samples until: {self.train_cutoff}")

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
        if self.H_path is not None:
            idx = self.H_list[idx][0] # H_list is a list of tuples (idx, H_variance)

        # unsqueeze to add channel dimension
        if self.load_memory:
            u = self.u[idx].unsqueeze(-1) # shape (nt, nx, 1) 
        else:
            u = torch.tensor(self.u[idx], dtype=torch.float32).unsqueeze(-1) # shape (nt, nx, 1)
        u = u[::self.t_downsample] # truncate to nt by taking every t_downsample
        u = u[self.time_start:]

        if self.split == "train" and self.train_pct < 1.0:
            u = u[:self.train_cutoff] # only use first half of training data

        return_dict = {"u": u, "dx": self.dx, "dt": self.dt, "x": self.x.unsqueeze(-1), "t": self.t}

        return return_dict