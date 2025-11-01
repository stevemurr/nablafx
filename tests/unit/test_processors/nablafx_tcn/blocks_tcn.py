"""
Architecture-Specific Blocks Module

This module contains conditional blocks and specialized components that are
specific to particular neural architectures (TCN, GCN, S4) in NablaFX.
"""

import torch
from rational.torch import Rational

from .components_tcn import FiLM, TFiLM, TinyTFiLM, TVFiLMMod, center_crop, causal_crop


# -----------------------------------------------------------------------------
# TCN Conditional Block
# -----------------------------------------------------------------------------


class TCNCondBlock(torch.nn.Module):
    """
    cond. types: None, FiLM, TFiLM, TVFiLM

    cond_dim:   if None -> cond_dim = 0
                if film -> cond_dim = conditioning MLP output size
                if tfilm -> cond_dim = number of control parameters
                if ttfilm -> cond_dim = number of control parameters
                if tvfilm -> cond_dim = TVFiLMCond output_dim
    batchnorm:  available only for non-conditional models

    activation types: Tanh, PReLU, Rational
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        causal,
        batchnorm,
        residual,
        kernel_size,
        padding,
        dilation,
        groups,
        bias,
        cond_type,
        cond_dim,
        cond_block_size,
        cond_num_layers,
        act_type,
    ):
        super(TCNCondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0
        assert cond_num_layers > 0
        assert act_type in ["tanh", "prelu", "rational"]

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.causal = causal
        self.batchnorm = batchnorm
        self.residual = residual
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers
        self.act_type = act_type

        # CONV
        self.conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=out_ch, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=out_ch, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=out_ch, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=out_ch, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        # ACTIVATIONS
        if act_type == "tanh":
            self.act = torch.nn.Tanh()
        elif act_type == "prelu":
            self.act = torch.nn.PReLU(num_parameters=out_ch)
        elif act_type == "rational":
            self.act = Rational(approx_func="tanh", degrees=[4, 3], version="A")

        # RESIDUAL
        if residual:
            self.res = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x, cond=None):
        """
        cond:   if film -> cond = conditioning MLP output
                if tfilm -> cond = control parameters
                if tvfilm -> cond = TVFiLMCond conditioning sequence
        """

        x_in = x

        # CONV
        x = self.conv(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            x = self.film(x, cond)
        elif self.cond_type is None and self.batchnorm:
            x = self.bn(x)

        # ACTIVATIONS
        x = self.act(x)

        # OUTPUT
        if self.residual:
            x_res = self.res(x_in)

            if self.causal:
                x = x + causal_crop(x_res, x.shape[-1])
            else:
                x = x + center_crop(x_res, x.shape[-1])

        return x
