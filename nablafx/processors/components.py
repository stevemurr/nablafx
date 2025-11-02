"""
Components Module for Neural Processors

This module contains general-purpose neural network components and utility functions
that are shared across different neural architectures in NablaFX.
"""

import torch
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Crop functions for residual connections
# -----------------------------------------------------------------------------


def center_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    """Center crop tensor to specified length."""
    if x.shape[-1] > length:
        start = (x.shape[-1] - length) // 2
        stop = start + length
        return x[..., start:stop]
    else:
        return x


def causal_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    """Causally crop tensor to specified length."""
    if x.shape[-1] > length:
        stop = x.shape[-1] - 1
        start = stop - length
        return x[..., start:stop]
    else:
        return x


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """
    MLP used in conditioning layers

    Args:
        input_dim: Input dimensions
        output_dim: Output dimensions
        num_layers: Number of layers
        hidden_dim: Hidden dimensions
        activation: Activation function
    """

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_layers: int = 3, 
        hidden_dim: int = 32, 
        activation: torch.nn.Module = torch.nn.ReLU()
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)

        self.mlp = torch.nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -----------------------------------------------------------------------------
# FiLM layer
# -----------------------------------------------------------------------------


class FiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond, modulates x.

    Args:
        nfeatures: Number of features (i.e., convolution channels)
        cond_dim: Number of conditioning features
    """

    def __init__(self, nfeatures: int, cond_dim: int):
        super(FiLM, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim

        self.bn = torch.nn.BatchNorm1d(nfeatures, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, nfeatures * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x = [batch, channels, length]
        # cond = [batch, cond_dim]
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)  # [batch, nfeatures, 1]
        b = b.unsqueeze(-1)  # [batch, nfeatures, 1]
        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # apply conditional affine
        return x


# -----------------------------------------------------------------------------
# Temporal FiLM layer
# -----------------------------------------------------------------------------


class TFiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    modulates x over time.

    Args:
        nfeatures: Number of features (i.e., convolution channels)
        cond_dim: Number of control parameters, if cond_dim = 0, modulation is only a function of features
        block_size: Size of blocks to modulate
        num_layers: Number of LSTM layers
    """

    def __init__(self, nfeatures: int, cond_dim: int, block_size: int, num_layers: int):
        super(TFiLM, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (hidden_state, cell_state)

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lstm = torch.nn.LSTM(
            input_size=nfeatures + cond_dim, hidden_size=nfeatures * 2, num_layers=num_layers, batch_first=False, bidirectional=False
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal FiLM modulation.
        
        Args:
            x: Input tensor [batch, nchannels, length]
            cond: Optional conditioning parameters [batch, nparams]
            
        Returns:
            Modulated tensor
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding_size = self.block_size - (x_in_shape[2] % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        # x_down = self.pool(torch.abs(x))
        x_down = self.pool(x)

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state is None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.nfeatures * 2).requires_grad_().type_as(x)
            c0 = torch.zeros(self.num_layers, x.size(0), self.nfeatures * 2).requires_grad_().type_as(x)
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back [batch, channels, length]
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        x_norm = torch.reshape(x_norm, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(x_norm, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out

    def reset_state(self) -> None:
        """Reset LSTM hidden state."""
        self.hidden_state = None


# -----------------------------------------------------------------------------
# Tiny Temporal FiLM layer
# -----------------------------------------------------------------------------


class TinyTFiLM(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    modulates x over time.

    Args:
        nfeatures: Number of features (i.e., convolution channels)
        bottleneck_dim: Bottleneck dimension
        cond_dim: Number of control parameters, if cond_dim = 0, modulation is only a function of features
        block_size: Size of blocks to modulate
        num_layers: Number of LSTM layers
    """

    def __init__(self, nfeatures: int, bottleneck_dim: int, cond_dim: int, block_size: int, num_layers: int):
        super(TinyTFiLM, self).__init__()
        self.nfeatures = nfeatures
        self.bottleneck_dim = bottleneck_dim
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (hidden_state, cell_state)

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # used for bottleneck
        self.conv = torch.nn.Conv1d(nfeatures, bottleneck_dim, kernel_size=1)

        self.lstm = torch.nn.LSTM(
            input_size=bottleneck_dim + cond_dim, hidden_size=bottleneck_dim, num_layers=num_layers, batch_first=False, bidirectional=False
        )

        # used to scale back up to nfeatures*2
        self.adaptor = torch.nn.Sequential(
            torch.nn.Conv1d(bottleneck_dim, 16, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, nfeatures * 2, kernel_size=1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply tiny temporal FiLM modulation.
        
        Args:
            x: Input tensor [batch, nchannels, length]
            cond: Optional conditioning parameters [batch, nparams]
            
        Returns:
            Modulated tensor
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding_size = self.block_size - (x_in_shape[2] % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.pool(x)

        # bottleneck
        x_down = self.conv(x_down)

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.hidden_state is None:  # state was reset
            # init hidden and cell states with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.bottleneck_dim).requires_grad_().type_as(x)
            c0 = torch.zeros(self.num_layers, x.size(0), self.bottleneck_dim).requires_grad_().type_as(x)
            x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back [batch, channels, length]
        x_norm = x_norm.permute(1, 2, 0)

        # mlp adaptor
        x_norm = self.adaptor(x_norm)

        # reshape input and modulation sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        x_norm = torch.reshape(x_norm, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(x_norm, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out

    def reset_state(self) -> None:
        """Reset LSTM hidden state."""
        self.hidden_state = None


# -----------------------------------------------------------------------------
# Time-Varying FiLM - modulation layer
# -----------------------------------------------------------------------------


class TVFiLMMod(torch.nn.Module):
    """
    Given an input sequence x and a conditioning sequence cond_seq,
    returns a modulated sequence.

    Args:
        nfeatures: Number of input channels
        cond_dim: Number of channels for conditioning sequence
        block_size: Size of blocks to modulate
    """

    def __init__(self, nfeatures: int, cond_dim: int, block_size: int):
        super(TVFiLMMod, self).__init__()
        self.nfeatures = nfeatures
        self.cond_dim = cond_dim
        self.block_size = block_size

        self.adaptor = torch.nn.Conv1d(cond_dim, nfeatures * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_seq: torch.Tensor) -> torch.Tensor:
        # x = [batch, channels, length]
        # cond_seq = [batch, cond_dim, length]

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[-1] % self.block_size) != 0:
            padding = self.block_size - (x_in_shape[-1] % self.block_size)
            x = torch.nn.functional.pad(x, [0, padding])

        x_pad_shape = x.shape
        nsteps = int(x_pad_shape[-1] / self.block_size)

        cond_seq = self.adaptor(cond_seq)
        cond_seq = cond_seq[..., -nsteps:]  # crop to match input length
        assert nsteps == cond_seq.shape[-1]

        # reshape input and conditioning sequence into blocks
        # [batch, channels, nsteps, block_size]
        x_in = torch.reshape(x, shape=(-1, self.nfeatures, nsteps, self.block_size))
        # [batch, channels*2, nsteps, 1]
        cond_seq = torch.reshape(cond_seq, shape=(-1, self.nfeatures * 2, nsteps, 1))

        # split modulation sequence along channels
        g, b = torch.chunk(cond_seq, 2, dim=1)

        # modulate
        x_out = (x_in * g) + b

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_pad_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out


# -----------------------------------------------------------------------------
# Time-Varying FiLM - conditioning layer
# -----------------------------------------------------------------------------


class TVFiLMCond(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    returns a conditioning sequence that modulates FiLM layers
    over time.

    Args:
        input_dim: Number of input channels
        output_dim: Number of channels for conditioning sequence
        cond_dim: Number of control parameters, if cond_dim = 0, modulation is only a function of x
        block_size: Size of blocks to modulate
        num_layers: Number of LSTM layers
    """

    def __init__(self, input_dim: int, output_dim: int, cond_dim: int, block_size: int, num_layers: int):
        super(TVFiLMCond, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers

        # initialized as a tensor for torchscript tracing
        self.hidden_state: Tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )
        self.is_hidden_state_init: bool = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lstm = torch.nn.LSTM(
            input_size=input_dim + cond_dim, hidden_size=output_dim, num_layers=num_layers, batch_first=False, bidirectional=False
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate conditioning sequence.
        
        Args:
            x: Input tensor [batch, nchannels, length]
            cond: Optional conditioning parameters [batch, nparams]
            
        Returns:
            Conditioning sequence [batch, output_dim, nsteps]
        """
        # x = [batch, nchannels, length]
        # cond = [batch, nparams]

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[-1] % self.block_size) != 0:
            padding = self.block_size - (x_in_shape[-1] % self.block_size)
            x = torch.nn.functional.pad(x, [0, padding])

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.pool(torch.abs(x))

        # upsample params [batch, nparams, nsteps]
        if self.cond_dim > 0 and cond is not None:
            cond_up = cond.unsqueeze(-1).repeat(1, 1, nsteps)

            # concat along channel dim [batch, nchannels+nparams, nsteps]
            x_down = torch.cat((x_down, cond_up), dim=1)

        # shape for LSTM [length, batch, channels]
        x_down = x_down.permute(2, 0, 1)

        # conditioning sequence
        if self.is_hidden_state_init:
            cond_seq, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            # default to zeros if hidden state is not provided
            cond_seq, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        cond_seq = cond_seq.permute(1, 2, 0)

        self.update_state(new_hidden_state)
        return cond_seq  # [batch, output_dim, nsteps]

    def update_state(self, new_hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Update LSTM hidden state."""
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True

    def detach_state(self) -> None:
        """Detach LSTM hidden state from computation graph."""
        if self.is_hidden_state_init:
            # TODO(cm): check whether clone is required or not
            self.hidden_state = tuple((h.detach().clone() for h in self.hidden_state))

    def reset_state(self) -> None:
        """Reset LSTM hidden state."""
        self.is_hidden_state_init = False
