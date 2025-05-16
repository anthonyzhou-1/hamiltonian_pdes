from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os.path
import torch 
import numpy as np
from tqdm import tqdm
import numpy.ma as ma
from einops import repeat

class Normalizer:
    def __init__(self,
                 use_norm = False,
                 stat_path = "./",
                 dataset=None,
                 scaler = "normal",
                 recalculate = False,
                 scaling_factor=1):
        self.use_norm = use_norm
        self.scaler = scaler # normal or minmax
        self.scaling_factor = scaling_factor

        if not self.use_norm:
            print("Normalization is turned off")
            return 

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
            print("Statistics loaded from", stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            print("Calculating statistics for normalization")
            dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=0)
            
            if self.scaler == "normal":
                u_scaler = StandardScaler()
                v_scaler = StandardScaler()
                p_scaler = StandardScaler()
            elif self.scaler == "minmax":
                u_scaler = MinMaxScaler(feature_range=(-1, 1)) 
                v_scaler = MinMaxScaler(feature_range=(-1, 1))
                p_scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError("Scaler must be either 'normal' or 'minmax'")

            for batch in tqdm(dataloader):
                data = batch["u"]

                p, u, v = self.get_data(data)

                p_scaler.partial_fit(p.reshape(-1, 1))
                u_scaler.partial_fit(u.reshape(-1, 1))
                v_scaler.partial_fit(v.reshape(-1, 1))

            if self.scaler == "normal":
                self.p_mean = p_scaler.mean_.item()
                self.p_std = np.sqrt(p_scaler.var_).item()
                self.u_mean = u_scaler.mean_.item()
                self.u_std = np.sqrt(u_scaler.var_).item()
                self.v_mean = v_scaler.mean_.item()
                self.v_std = np.sqrt(v_scaler.var_).item()

            else:
                self.p_min = p_scaler.min_.item()
                self.p_scale = p_scaler.scale_.item()
                self.u_min = u_scaler.min_.item()
                self.u_scale = u_scaler.scale_.item()
                self.v_min = v_scaler.min_.item()
                self.v_scale = v_scaler.scale_.item()
                
            self.save_stats(path=stat_path)
            print("Statistics saved to", stat_path)

        self.print_stats()

        if self.scaler == "normal":
            self.u_mean = torch.tensor(self.u_mean)
            self.u_std = torch.tensor(self.u_std)
            self.v_mean = torch.tensor(self.v_mean)
            self.v_std = torch.tensor(self.v_std)
            self.p_mean = torch.tensor(self.p_mean)
            self.p_std = torch.tensor(self.p_std)

        else:
            self.u_min = torch.tensor(self.u_min)
            self.u_scale = torch.tensor(self.u_scale)
            self.v_min = torch.tensor(self.v_min)
            self.v_scale = torch.tensor(self.v_scale)
            self.p_min = torch.tensor(self.p_min)
            self.p_scale = torch.tensor(self.p_scale)


    def get_data(self, data):
        # data in shape [batch, t, nx, ny, c] or [batch, t, n, c]
        p = data[..., 0]
        u = data[..., 1]
        v = data[..., 2]

        return p, u, v
    
    def assemble_data(self, p, u, v):
        return torch.stack([p, u, v], dim=-1)

    def print_stats(self):
        if self.scaler == "minmax":
            print(f"p min: {self.p_min}, p scale: {self.p_scale}")
            print(f"u min: {self.u_min}, u scale: {self.u_scale}")
            print(f"v min: {self.v_min}, v scale: {self.v_scale}")
            print(f"scaling factor: {self.scaling_factor}") 
        else:
            print(f"p mean: {self.p_mean}, p std: {self.p_std}")
            print(f"u mean: {self.u_mean}, u std: {self.u_std}")
            print(f"v mean: {self.v_mean}, v std: {self.v_std}")
            print(f"scaling factor: {self.scaling_factor}")

    def save_stats(self, path):
        if self.scaler == "minmax":
            with open(path, "wb") as f:
                pickle.dump([self.p_min, self.p_scale, self.u_min, self.u_scale, self.v_min, self.v_scale], f)
        else:
            with open(path, "wb") as f:
                pickle.dump([self.p_mean, self.p_std, self.u_mean, self.u_std, self.v_mean, self.v_std], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            if self.scaler == "minmax":
                self.p_min, self.p_scale, self.u_min, self.u_scale, self.v_min, self.v_scale = pickle.load(f)
            else:
                self.p_mean, self.p_std, self.u_mean, self.u_std, self.v_mean, self.v_std = pickle.load(f)
    
    def normalize(self, x):
        if not self.use_norm:
            return x

        x_norm = x.clone()
        p_norm, u_norm, v_norm = self.get_data(x_norm)

        if self.scaler == "normal":
            p_norm = (p_norm - self.p_mean) / self.p_std
            u_norm = (u_norm - self.u_mean) / self.u_std
            v_norm = (v_norm - self.v_mean) / self.v_std

        else:
            p_norm = p_norm * self.p_scale + self.p_min
            u_norm = u_norm * self.u_scale + self.u_min
            v_norm = v_norm * self.v_scale + self.v_min

        x_norm = self.assemble_data(p_norm, u_norm, v_norm)
        x_norm = x_norm * self.scaling_factor

        return x_norm

    def denormalize(self, x): 
        if not self.use_norm:
            return x

        x_denorm = x.clone()
        x_denorm = x_denorm / self.scaling_factor
        p_denorm, u_denorm, v_denorm = self.get_data(x_denorm)

        if self.scaler == "normal":
            p_denorm = p_denorm * self.p_std + self.p_mean
            u_denorm = u_denorm * self.u_std + self.u_mean
            v_denorm = v_denorm * self.v_std + self.v_mean

        else:
            p_denorm = (p_denorm - self.p_min) / self.p_scale
            u_denorm = (u_denorm - self.u_min) / self.u_scale
            v_denorm = (v_denorm - self.v_min) / self.v_scale

        x_denorm = self.assemble_data(p_denorm, u_denorm, v_denorm)

        return x_denorm