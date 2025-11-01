"""
Architecture-Specific Blocks Module

This module contains conditional blocks and specialized components that are
specific to particular neural architectures (TCN, GCN, S4) in NablaFX.
"""

import torch

from .components_gcn import FiLM, TFiLM, TinyTFiLM, TVFiLMMod, center_crop, causal_crop

# -----------------------------------------------------------------------------
# GCN Conditional Block
# -----------------------------------------------------------------------------


class GCNCondBlock(torch.nn.Module):
    """
    cond. types: None, FiLM, TFiLM, TVFiLM

    cond_dim:   if None -> cond_dim = 0
                if film -> cond_dim = conditioning MLP output size
                if tfilm -> cond_dim = number of control parameters
                if ttfilm -> cond_dim = number of control parameters
                if tvfilm -> cond_dim = TVFiLMCond output_dim
    batchnorm:  available only for non-conditional models
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
    ):
        super(GCNCondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0

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

        # CONV
        self.conv = torch.nn.Conv1d(
            in_ch,
            out_ch * 2,  # for the gated activation
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=out_ch * 2, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=out_ch * 2, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=out_ch * 2, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=out_ch * 2, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(out_ch * 2)

        # GATED ACTIVATION
        self.tanh = torch.nn.Tanh()
        self.sigm = torch.nn.Sigmoid()

        # MIX
        self.mix = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x, cond=None):
        """
        cond:   if film -> cond = conditioning MLP output
                if tfilm -> cond = control parameters
                if tvfilm -> cond = TVFiLMCond conditioning sequence
        """

        x_res = x

        # CONV
        y = self.conv(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            y = self.film(y, cond)
        elif self.cond_type is None and self.batchnorm:
            y = self.bn(y)

        # GATED ACTIVATION
        z = self.tanh(y[:, : self.out_ch, :]) * self.sigm(y[:, self.out_ch :, :])

        # OUTPUT
        if self.residual:
            if self.causal:
                x = self.mix(z) + causal_crop(x_res, z.shape[-1])
            else:
                x = self.mix(z) + center_crop(x_res, z.shape[-1])
        else:
            x = self.mix(z)

        return x, z
