import torch 
import torch.nn as nn
import numpy as np
from modules.models.FiLM import FiLM
import torch.nn.functional as F

class MLP_Kernel(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 activation=F.gelu,
                 film_type="disabled", 
                 film_dims=[], 
                 kernel_size=3,
                 periodic_embedding=False,):
        super().__init__()


        self.net = nn.ModuleList()
        self.coord_dim = in_features # assume input is coordinates
        self.func_dim = out_features # assume output is function values
        self.activation = activation
        self.n_layers = hidden_layers + 2 # +2 for first and last layer
        self.periodic_embedding = periodic_embedding 

        # create first layer
        self.net.append(nn.Linear(in_features, hidden_features))

        # create hidden layers
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))

                
        self.net.append(nn.Linear(hidden_features, out_features))
            
        self.use_film = False if film_type == "disabled" else True
        self.film_type = film_type # disabled, local, global

        if self.use_film:
            self.film_layers = nn.ModuleList() 
            for i in range(hidden_layers + 2): # +2 for first and last layer
                if i == hidden_layers + 1: 
                    self.film_layers.append(FiLM(self.func_dim, 
                                                 out_features, 
                                                 film_dims, 
                                                 film_type, 
                                                 kernel_size=kernel_size,))
                else:
                    self.film_layers.append(FiLM(self.func_dim, 
                                                 hidden_features, 
                                                 film_dims, 
                                                 film_type, 
                                                 kernel_size=kernel_size,))

        print(f"MLP Kernel: film_type: {film_type}, periodic_embedding: {periodic_embedding}")

    def forward(self, x, u_global=None):
        # x in dim (batch, coord_dim). If use_film, then x in dim (batch, coord_dim + func_dim)

        if self.periodic_embedding:
            x_max = torch.max(x) # get max value
            x = torch.sin(2*torch.pi/x_max * x) # periodic embedding
        
        if self.use_film:
            if self.film_type == "local":
                x, u = x[..., :self.coord_dim], x[..., self.coord_dim:] # (batch, coord_dim), (batch, func_dim)
            else:
                u = u_global # (batch, nx, func_dim) or (batch, nx, ny, func_dim)

            for i, (layer, film_layer) in enumerate(zip(self.net, self.film_layers)):
                x = layer(x) # get activation x' = Wx + b
                if i < self.n_layers - 1:
                    x = self.activation(x) # get activation y = o(x')
                x = film_layer(x, u) # modulate activation z = gamma(u) * y + beta(u)
        else:
            for i, layer in enumerate(self.net):
                x = layer(x) # z = Wx + b
                if i < self.n_layers - 1:
                    x = self.activation(x) # y = o(Wx + b)

        return x


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True,
                 is_first=False, 
                 omega_0=30,):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        # input in dim (batch, coord_dim)
        return torch.sin(self.omega_0 * self.linear(input))

    
class Siren_Kernel(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 func_dim = None,
                 outermost_linear=False, 
                 first_omega_0=1.0, 
                 hidden_omega_0=1.0, 
                 film_type="disabled", 
                 film_dims=[], 
                 kernel_size=3,
                 dim=1,
                 periodic_embedding=False,):
        super().__init__()
        
        self.net = nn.ModuleList()
        self.coord_dim = in_features # assume input is coordinates
        self.func_dim = out_features if func_dim is None else func_dim # assume output is function values
        self.periodic_embedding = periodic_embedding

        # create first layer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        # create hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        # create final layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
            
        self.use_film = False if film_type == "disabled" else True
        self.film_type = film_type # disabled, local, global

        if self.use_film:
            self.film_layers = nn.ModuleList() 
            for i in range(hidden_layers + 2): # +2 for first and last layer
                if i == hidden_layers + 1: 
                    self.film_layers.append(FiLM(self.func_dim, 
                                                 out_features, 
                                                 film_dims, 
                                                 film_type, 
                                                 kernel_size=kernel_size, 
                                                 dim=dim,))
                else:
                    self.film_layers.append(FiLM(self.func_dim, 
                                                 hidden_features, 
                                                 film_dims, 
                                                 film_type, 
                                                 kernel_size=kernel_size, 
                                                 dim=dim,))
        
        
        print(f"Siren Kernel: film_type: {film_type}, periodic_embedding: {periodic_embedding}")

    def forward(self, x, u_global=None):
        # x in dim (batch, coord_dim). If use_film, then x in dim (batch, coord_dim + func_dim)

        if self.periodic_embedding:
            x_max = torch.max(x) # get max value (length of domain [0, L])
            x = torch.sin(2*torch.pi/x_max * x) # periodic embedding
        
        if self.use_film: # film
            if self.film_type == "local":
                x, u = x[..., :self.coord_dim], x[..., self.coord_dim:] # (batch, coord_dim), (batch, func_dim)
            else:
                u = u_global # (batch, nx, func_dim) or (batch, nx, ny, func_dim)

            for layer, film_layer in zip(self.net, self.film_layers):
                x = layer(x) # get activation x' = sin(Wx + b)
                x = film_layer(x, u) # modulate activation z = gamma(u) * x' + beta(u)
        else: # concat
            for layer in self.net:
                x = layer(x) # get activation z = sin(Wx + b)
        return x