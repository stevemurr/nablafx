"""
Architecture-Specific Blocks Module

This module contains conditional blocks and specialized components that are
specific to particular neural architectures (TCN, GCN, S4) in NablaFX.
"""

import math
import torch
from einops import rearrange, repeat
from torch import Tensor
from rational.torch import Rational

from .components_s4 import FiLM, TFiLM, TinyTFiLM, TVFiLMMod

# -----------------------------------------------------------------------------
# Diagonal State-Space Model Block
# -----------------------------------------------------------------------------

c2r = torch.view_as_real
r2c = torch.view_as_complex


class DSSM(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()

        H = input_dim
        self.H = H
        N = state_dim
        self.N = N

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = torch.nn.Parameter(c2r(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(torch.arange(N), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        self.D = torch.nn.Parameter(torch.randn(H))

    def get_kernel(self, length: int):  # `length` is `L`
        dt = torch.exp(self.log_dt)  # (H)
        C = r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        P = dtA.unsqueeze(-1) * torch.arange(length, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(P)).real
        return K

    def register(self, name: str, tensor: Tensor, lr: float = None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, torch.nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}  # Never use weight decay
            if lr is not None:  # Use custom learning rate when a learning rate is given
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u: Tensor, length: Tensor = None):
        # Input and output shape (B H L)
        assert u.dim() == 3
        B, H, L = u.size()
        assert H == self.H

        # length shape (L)
        if length is None:
            length = torch.empty(B).fill_(L)
        assert length.dim() == 1
        assert length.size(0) == B
        assert torch.all(length <= L)
        length = length.to(torch.long).cpu()

        l_s, i_s = length.sort(stable=True, descending=True)
        l_s, i_s = l_s.tolist(), i_s.tolist()
        prev_i = 0
        pair: list[tuple[int, list[int]]] = []
        for i in range(1, B):
            if l_s[i] == l_s[i - 1]:
                continue
            pair.append((l_s[prev_i], i_s[prev_i:i]))
            prev_i = i
        pair.append((l_s[prev_i], i_s[prev_i:]))

        kernel = self.get_kernel(L)  # (H L)
        out = torch.zeros_like(u)  # (B H L)

        for l, idxs in pair:
            _k = kernel[:, :l]
            _u = u[idxs, :, :l]

            k_f = torch.fft.rfft(_k, n=2 * l)  # (H l)
            u_f = torch.fft.rfft(_u, n=2 * l)  # (B H l)
            y = torch.fft.irfft(u_f * k_f, n=2 * l)[..., :l]  # (B H l)
            y += _u * self.D.unsqueeze(-1)  # (B H l)

            out[idxs, :, :l] = y[...]

        return out


# -----------------------------------------------------------------------------
# S4 Conditional Block
# -----------------------------------------------------------------------------


class S4CondBlock(torch.nn.Module):
    def __init__(
        self,
        channel_width: int,
        batchnorm: bool,
        residual: bool,
        s4_state_dim: int,
        s4_learning_rate: float,
        cond_type: str,
        cond_dim: int,
        cond_block_size: int,
        cond_num_layers: int,
        act_type: str,
    ):
        super(S4CondBlock, self).__init__()
        assert cond_dim >= 0
        assert cond_type in [None, "film", "tfilm", "ttfilm", "tvfilm"]
        assert cond_block_size > 0
        assert cond_num_layers > 0
        assert act_type in ["tanh", "prelu", "rational"]

        self.channel_width = channel_width
        self.batchnorm = batchnorm
        self.residual = residual
        self.s4_state_dim = s4_state_dim
        self.s4_learning_rate = s4_learning_rate
        self.cond_type = cond_type
        self.cond_dim = cond_dim
        self.cond_block_size = cond_block_size
        self.cond_num_layers = cond_num_layers
        self.act_type = act_type

        # LINEAR
        self.linear = torch.nn.Linear(channel_width, channel_width)

        # S4
        self.s4 = DSSM(input_dim=channel_width, state_dim=s4_state_dim, lr=s4_learning_rate)

        # CONDITIONING/MODULATION
        if cond_type == "film":
            self.film = FiLM(nfeatures=channel_width, cond_dim=cond_dim)
        elif cond_type == "tfilm":
            self.film = TFiLM(nfeatures=channel_width, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers)
        elif cond_type == "ttfilm":
            self.film = TinyTFiLM(
                nfeatures=channel_width, bottleneck_dim=4, cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        elif cond_type == "tvfilm":
            self.film = TVFiLMMod(nfeatures=channel_width, cond_dim=cond_dim, block_size=cond_block_size)
        elif cond_type is None and batchnorm:
            self.bn = torch.nn.BatchNorm1d(channel_width)

        # ACTIVATIONS
        if act_type == "tanh":
            self.act1 = torch.nn.Tanh()
            self.act2 = torch.nn.Tanh()
        elif act_type == "prelu":
            self.act1 = torch.nn.PReLU(num_parameters=1)
            self.act2 = torch.nn.PReLU(num_parameters=1)
        elif act_type == "rational":
            self.act1 = Rational(approx_func="tanh", degrees=[4, 3], version="A")
            self.act2 = Rational(approx_func="tanh", degrees=[4, 3], version="A")

        # RESIDUAL
        if residual:
            self.res = torch.nn.Conv1d(
                channel_width,
                channel_width,
                kernel_size=1,
                groups=channel_width,
                bias=False,
            )

    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:

        x_in = x

        # LINEAR
        x = rearrange(x, "B H L -> B L H")
        x = self.linear(x)
        x = rearrange(x, "B L H -> B H L")

        # ACTIVATION
        x = self.act1(x)

        # S4
        x = self.s4(x)

        # CONDITIONING/MODULATION
        if self.cond_type is not None:
            x = self.film(x, cond)
        elif self.cond_type is None and self.batchnorm:
            x = self.bn(x)

        # ACTIVATION
        x = self.act2(x)

        # OUTPUT
        if self.residual:
            x = x + self.res(x_in)

        return x
