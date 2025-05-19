import h5py 
import numpy as np
from tqdm import tqdm 
import torch 

def read_data(dir, nt, nx, ny):
    all_t = torch.zeros(nt, 3, nx, ny)
    for i in range(nt):
        if i < 10:
            load_path = f"{dir}/claw000{i}.hdf"
        elif i<100:
            load_path = f"{dir}/claw00{i}.hdf"
        else:
            load_path = f"{dir}/claw0{i}.hdf"

        f = h5py.File(load_path, "r")
        u_t = np.array(f['patch1']['q'])
        all_t[i] = torch.from_numpy(u_t)

    all_t = all_t.permute(0, 2, 3, 1) # shape (nt, nx, ny, 3)
    return all_t

num_samples = 64
nt = 101
nx = 256
ny = 256
split = "valid"

if split == "train":
    offset = 0
else:
    offset = 1000

path = f"SWE_Gaussian_{split}_{num_samples}.h5"
h5f = h5py.File(path, 'a')

dataset = h5f.create_group(split)

h5f_u = dataset.create_dataset(f'u', (num_samples, nt, nx, ny, 3), dtype='f4')
tcoord = dataset.create_dataset(f't', (nt), dtype='f4')
xcoord = dataset.create_dataset(f'x', (nx, ny, 2), dtype='f4')

t_coord = torch.linspace(0, 2, nt)
tcoord[:] = t_coord

x = torch.linspace(-1, 1, 256)
y = torch.linspace(-1, 1, 256)
X,Y = torch.meshgrid(x,y)
grid = torch.stack([X, Y], -1)  
xcoord[:] = grid

for i in tqdm(range(num_samples)):
    dir = f"swe_2d_gaussian_{split}_{i+offset}"
    u = read_data(dir, nt, nx, ny)
    h5f_u[i] = u

h5f.close()