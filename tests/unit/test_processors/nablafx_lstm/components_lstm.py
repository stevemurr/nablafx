"""
Components Module for Neural Processors

This module contains general-purpose neural network components and utility functions
that are shared across different neural architectures in NablaFX.
"""

import torch

# -----------------------------------------------------------------------------
# Time-Varying FiLM - conditioning layer
# -----------------------------------------------------------------------------


class TVFiLMCond(torch.nn.Module):
    """
    Given an input sequence x and conditioning parameters cond,
    returns a conditioning sequence that modulates FiLM layers
    over time.

    input_dim: number of input channels
    output_dim: number of channels for conditioning sequence
    cond_dim: number of control parameters, if cond_dim = 0, modulation is only a function of x
    block_size: size of blocks to modulate
    num_layers: number of LSTM layers
    """

    def __init__(self, input_dim, output_dim, cond_dim, block_size, num_layers):
        super(TVFiLMCond, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.block_size = block_size
        self.num_layers = num_layers

        # initialized as a tensor for torchscript tracing
        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(kernel_size=block_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lstm = torch.nn.LSTM(
            input_size=input_dim + cond_dim, hidden_size=output_dim, num_layers=num_layers, batch_first=False, bidirectional=False
        )

    def forward(self, x, cond=None):
        """
        when cond is None, modulation is only a function of x
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

    def update_state(self, new_hidden):
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            # TODO(cm): check whether clone is required or not
            self.hidden_state = tuple((h.detach().clone() for h in self.hidden_state))

    def reset_state(self):
        self.is_hidden_state_init = False
