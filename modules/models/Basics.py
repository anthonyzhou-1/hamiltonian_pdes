import torch 
import torch.nn as nn
import torch.nn.functional as F

def conv_nd(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dim=1):
    if dim == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    elif dim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    elif dim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
def avg_pool_nd(kernel_size, stride, dim=1):
    if dim == 1:
        return nn.AvgPool1d(kernel_size, stride)
    elif dim == 2:
        return nn.AvgPool2d(kernel_size, stride)
    elif dim == 3:
        return nn.AvgPool3d(kernel_size, stride)

class MLP(torch.nn.Module):
    def __init__(self, 
                 layers, 
                 non_linearity=F.gelu, 
                 dropout=0.0,):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
    
class CNN(torch.nn.Module):
    def __init__(self, 
                 layers,
                 in_features = 1,
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 non_linearity=F.gelu, 
                 dropout=0.0,
                 dim=1,
                 downsample = False,
                 latent_dim = 1,):
        super().__init__()
        
        self.num_layers = len(layers) - 1
        self.non_linearity = non_linearity
        self.kernel_size = kernel_size
        self.dim = dim  

        self.input_projection = nn.Linear(in_features, layers[0])

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(conv_nd(layers[i], 
                                         layers[i + 1], 
                                         kernel_size=kernel_size, 
                                         stride=stride, 
                                         padding=padding,
                                         dim=dim))
        self.downsample_layers = nn.ModuleList()
        self.downsample = downsample

        if downsample:
            self.downsample_layer = avg_pool_nd(kernel_size=2, stride=2) # downsample by 2
            self.final_downsample = nn.AdaptiveAvgPool1d(1) if dim == 1 else nn.AdaptiveAvgPool2d((1, 1))
            self.latent_projection = nn.Linear(layers[-1], latent_dim)
            
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_layers)])
            if dropout > 0.0
            else None
        )

    def pad(self, input: torch.Tensor, padding = 1) -> torch.Tensor:
        """
        Periodic padding for N-D convolution
        """
        # input in shape (batch, channel, x) or (batch, channel, x, y)
        if self.dim == 1:
            padding = padding+1
            left = input[..., -padding:-1]
            right = input[..., 1:padding]
            padded_input = torch.cat([left, input, right], -1) # (batch, channel, x+2*padding)
        elif self.dim == 2:
            padding = padding+1
            left = input[..., -padding:-1, :]
            right = input[..., 1:padding, :]
            padded_input = torch.cat([left, input, right], -2)
            top = padded_input[..., -padding:-1]
            bottom = padded_input[..., 1:padding]
            padded_input = torch.cat([top, padded_input, bottom], -1)

        return padded_input

        
    def forward(self, x):
        # x in shape (batch, nx, c) or (batch, nx, ny, c)
        x = self.input_projection(x) # (batch, nx, 1) -> (batch, nx, c)

        if self.dim == 1:
            x = x.permute(0, 2, 1) # (batch, c, nx)
        else:
            x = x.permute(0, 3, 1, 2) # (batch, c, nx, ny)
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = self.pad(x, padding = self.kernel_size//2) # same, circular padding
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
            if self.downsample:
                x = self.downsample_layer(x)

        if self.downsample:
            x = self.final_downsample(x) # (b c nx) -> (b c 1)
            x = torch.flatten(x, start_dim=1) # (b c 1) -> (b c)
            x = self.latent_projection(x) # (b c) -> (b d)
            return x

        if self.dim == 1:
            x = x.permute(0, 2, 1) # (batch, nx, c)
        else:
            x = x.permute(0, 2, 3, 1) # (batch, nx, ny, c)
        return x
