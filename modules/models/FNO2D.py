import torch
from torch import nn
from common.utils import sinusoidal_embedding

def batchmul2d(input, weights, emb=None):
    '''
    args:
        input: (batch, c_in, modes1, modes2)
        weights: (c_in, c_out, modes1, modes2)
        emb: (batch, modes1, modes2)
    '''
    if emb is not None:
        input = input * emb.unsqueeze(1)

    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    out = torch.einsum("bixy,ioxy->boxy", input, weights)
    return out

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)


class SpectralConv2d_cond(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, cond_channels = 0):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        if cond_channels > 0:
            self.cond_emb = FreqLinear(cond_channels, self.modes1, self.modes2)

    def forward(self, x, emb=None):
        if emb is not None:
            emb12 = self.cond_emb(emb)
            # emb12 has shape (batch, modes1, modes2, 2)
            emb1 = emb12[..., 0]
            emb2 = emb12[..., 1]
        else:
            emb1 = None
            emb2 = None

        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1, emb1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2, emb2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    
class FourierBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = nn.GELU()

        self.fourier1 = SpectralConv2d_cond(in_planes, planes, cond_channels=cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv2d_cond(planes, planes, cond_channels=cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)

        if cond_channels > 0:
            self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None):
        # x has shape [b, c, x, y]
        # emb has shape [b, cond_channels]
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)

        if emb is not None:
            emb_out = self.cond_emb(emb)
            while len(emb_out.shape) < len(x2.shape):
                emb_out = emb_out[..., None]
            out = self.activation(x1 + x2 + emb_out)
        else:
            out = self.activation(x1 + x2)

        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out

class FNO2d_cond(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 modes1: int = 6,
                 modes2: int = 6,
                 width: int = 24,
                 cond_channels: int = 0,
                 cond_dim: int = 0,
                 num_layers: int = 4,
                 time_scale: float = 1.0,
                 cond_scale: float = 1.0) -> None:
        super(FNO2d_cond, self).__init__()
        """
        Args:
            time_window (int): input/output timesteps of the trajectory
            modes1 (int): low frequency Fourier modes considered for multiplication in the Fourier space
            modes2 (int): low frequency Fourier modes considered for multiplication in the Fourier space
            width (int): hidden channel dimension
            num_layers (int): number of FNO layers
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.cond_dim = cond_dim
        self.time_scale = time_scale
        self.cond_scale = cond_scale

        self.conv_in1 = nn.Conv2d(
            self.in_channels,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.width,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

        if cond_channels > 0:
            self.emb_in = nn.Linear(cond_channels*(1+self.cond_dim), cond_channels)

        self.layers = nn.ModuleList(
            [
                FourierBasicBlock(self.width, self.width, cond_channels, modes1=self.modes1, modes2=self.modes2)
                for i in range(num_layers)
            ]
        )

        self.activation = nn.GELU()

    def forward(self, 
                x: torch.Tensor,
                c: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape [batch, x, y, time] or [batch, x, y, time * channels]
            c (torch.Tensor): condition tensor of shape [batch, ] or [batch, cond_dim] for multiple conditions
        Returns: torch.Tensor: output has the shape [batch, x, y, time] or [batch, x, y, time * channels]
        """
        x = x.permute(0, 3, 1, 2) # (batch, x, y, time) -> (batch, time, x, y)

        if self.cond_channels > 0:
            if len(c.shape) == 1:
                emb = sinusoidal_embedding(c, self.cond_channels, scale=self.cond_scale) # (b,) -> (b, cond_channels)
            else:
                c_emb = [sinusoidal_embedding(c[:, i], self.cond_channels, scale=self.cond_scale) for i in range(self.cond_dim)]
                emb = torch.cat(c_emb, dim=-1)
            emb = self.emb_in(emb) # (b, cond_channels * (1+cond_dim)) -> (b, cond_channels)
        else:
            emb = None

        x = self.activation(self.conv_in1(x)) # (batch, time_window, x, y) -> (batch, width, x, y)
        x = self.activation(self.conv_in2(x)) # (batch, width, x, y) -> (batch, width, x, y)

        for layer in self.layers:
            x = layer(x, emb) # (batch, width, x, y) -> (batch, width, x, y)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        x = x.permute(0, 2, 3, 1) # (batch, width, x, y) -> (batch, x, y, width)

        return x