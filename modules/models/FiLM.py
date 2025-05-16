import torch 
import torch.nn as nn
from modules.models.Basics import MLP, CNN
from einops import rearrange

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, in_features, hidden_dim, film_dims, film_type, kernel_size=3, dim=1):
        super().__init__()
        if film_type == "local":
            self.film_net = MLP([in_features] + film_dims + [hidden_dim * 2])
        elif film_type == "global":
            self.film_net = CNN(in_features=in_features,
                                layers=film_dims + [hidden_dim * 2],
                                kernel_size=kernel_size, 
                                dim=dim,)
        self.film_type = film_type
        self.dim = dim

    def forward(self, x, u):
        # x in shape (batch, hidden_dim)
        # u in shape (batch, in_features) or (batch, nx, in_features) or (batch, nx, ny, in_features)

        out = self.film_net(u) # (batch, hidden_dim * 2) or (batch, nx, hidden_dim * 2) or (batch, nx, ny, hidden_dim * 2)
        gammas, betas = torch.chunk(out, 2, dim=-1) # (batch, hidden_dim) or (batch, nx, hidden_dim) or (batch, nx, ny, hidden_dim)

        if self.film_type == "global":
            if self.dim == 1:
                gammas = rearrange(gammas, 'b nx c -> (b nx) c')
                betas = rearrange(betas, 'b nx c -> (b nx) c')
            elif self.dim == 2:
                gammas = rearrange(gammas, 'b nx ny c -> (b nx ny) c')
                betas = rearrange(betas, 'b nx ny c -> (b nx ny) c')

        return (gammas * x) + betas
