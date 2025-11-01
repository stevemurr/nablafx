# from https://github.com/mcomunita/gcn-tfilm

import torch
import math
import argparse

""" 
Gated convolutional layer, zero pads and then applies a causal convolution to the input 
"""


class GatedConv1d(torch.nn.Module):

    def __init__(self, in_ch, out_ch, dilation, kernel_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size

        # Layers: Conv1D -> Activations -> Mix + Residual

        self.conv = torch.nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch * 2, kernel_size=kernel_size, stride=1, padding=0, dilation=dilation
        )

        self.mix = torch.nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, : self.out_ch, :]) * torch.sigmoid(y[:, self.out_ch :, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0], self.out_ch, residual.shape[2] - z.shape[2]), z), dim=2)

        x = self.mix(z) + residual

        return x, z


""" 
Gated convolutional neural net block, applies successive gated convolutional layers to the input, a total of 'layers'
layers are applied, with the filter size 'kernel_size' and the dilation increasing by a factor of 'dilation_growth' for
each successive layer.
"""


class GCNBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, nlayers, kernel_size, dilation_growth):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth

        dilations = [dilation_growth**l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch, out_ch=out_ch, dilation=d, kernel_size=kernel_size))
            in_ch = out_ch

    def forward(self, x):
        # [batch, channels, length]
        z = torch.empty([x.shape[0], self.nlayers * self.out_ch, x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, n * self.out_ch : (n + 1) * self.out_ch, :] = zn

        return x, z


""" 
Gated Convolutional Neural Net class, based on the 'WaveNet' architecture, takes a single channel of audio as input and
produces a single channel of audio of equal length as output. one-sided zero-padding is used to ensure the network is 
causal and doesn't reduce the length of the audio.

Made up of 'blocks', each one applying a series of dilated convolutions, with the dilation of each successive layer 
increasing by a factor of 'dilation_growth'. 'layers' determines how many convolutional layers are in each block,
'kernel_size' is the size of the filters. Channels is the number of convolutional channels.

The output of the model is creating by the linear mixer, which sums weighted outputs from each of the layers in the 
model
"""


class GCN(torch.nn.Module):
    def __init__(self, nblocks=2, nlayers=9, nchannels=8, kernel_size=3, dilation_growth=2, **kwargs):
        super(GCN, self).__init__()
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(
                GCNBlock(
                    in_ch=1 if b == 0 else nchannels,
                    out_ch=nchannels,
                    nlayers=nlayers,
                    kernel_size=kernel_size,
                    dilation_growth=dilation_growth,
                )
            )

        # output mixing layer
        self.blocks.append(torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # x.shape = [length, batch, channels]
        x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
            z[:, n * self.nchannels * self.nlayers : (n + 1) * self.nchannels * self.nlayers, :] = zn

        # back to [length, batch, channels]
        return self.blocks[-1](z).permute(2, 0, 1)

    def compute_receptive_field(self):
        """Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf
