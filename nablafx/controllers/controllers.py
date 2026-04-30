"""
Controllers Module for NablaFX

This module contains all controller implementations for neural audio processing.
Controllers generate control parameters for processors based on input signals and/or external controls.
"""

import torch
from typing import Union, Optional, Tuple

from nablafx.processors.components import MLP

# -----------------------------------------------------------------------------
# Dummy Controller
# -----------------------------------------------------------------------------


class DummyController(torch.nn.Module):
    """Dummy controller that returns None"""

    def __init__(self):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = 0

    def forward(self, x: torch.Tensor) -> None:
        return None


# -----------------------------------------------------------------------------
# Static Controller
# -----------------------------------------------------------------------------


class StaticController(torch.nn.Module):
    """Non-conditional controller that maps an internal tensor
    to a set of control parameters.
    """

    def __init__(
        self,
        num_control_params: int,
        control_params_initial: Union[str, float],
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = num_control_params
        if control_params_initial == "rnd":
            self.control_params = torch.nn.Parameter(torch.randn(num_control_params))
        else:
            self.control_params = torch.nn.Parameter(torch.ones(num_control_params) * control_params_initial)
        self.lr_multiplier = lr_multiplier

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, chs, seq_len = x.size()
        return self.act(self.control_params).unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1)


# -----------------------------------------------------------------------------
# Static Conditional Controller
# -----------------------------------------------------------------------------


class StaticCondController(torch.nn.Module):
    """Conditional controller that maps input controls
    to a set of control parameters.
    """

    def __init__(
        self,
        num_controls: int,
        num_control_params: int,
        num_layers: int = 3,
        hidden_dim: int = 16,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.num_control_params = num_control_params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr_multiplier = lr_multiplier

        self.projection = MLP(
            input_dim=num_controls,
            output_dim=num_control_params,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=torch.nn.Sigmoid(),
        )

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        bs, chs = controls.shape
        assert chs == self.num_controls

        return self.projection(controls).unsqueeze(-1)


# -----------------------------------------------------------------------------
#  Dynamic Controller
# -----------------------------------------------------------------------------


class DynamicController(torch.nn.Module):
    """Non-conditional controller that maps an input signal
    to a set of time-varying control parameters.
    """

    def __init__(
        self,
        num_control_params: int,
        block_size: int = 128,
        num_layers: int = 1,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = num_control_params
        self.block_size = block_size
        self.num_layers = num_layers
        self.lr_multiplier = lr_multiplier

        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )  # initialized as a tensor for torchscript tracing
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(
            kernel_size=block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=num_control_params,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, chs, seq_len = x.shape
        assert chs == 1

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (seq_len % self.block_size) != 0:
            padding_size = self.block_size - (seq_len % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        nsteps = int(x.shape[-1] / self.block_size)

        # downsample signal
        x_down = self.pool(torch.abs(x))

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # control sequence
        if self.is_hidden_state_init:
            control_params, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            control_params, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        control_params = control_params.permute(1, 2, 0)

        # limit
        control_params = self.act(control_params)

        # upsample to original size
        control_params = control_params.repeat_interleave(self.block_size, dim=-1)
        self.update_state(new_hidden_state)
        return control_params[..., :seq_len]

    def reset_states(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        self.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple((h.detach() for h in self.hidden_state))

    def update_state(self, new_hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True


# -----------------------------------------------------------------------------
# Spectral Dynamic Controller
# -----------------------------------------------------------------------------


class SpectralDynamicController(torch.nn.Module):
    """Time-varying controller whose per-block input feature is a windowed
    log-magnitude spectrum (rfft) instead of a single peak scalar.

    DynamicController feeds the LSTM `max(|x|)` per block — a single number
    that throws away all spectral information. For tasks where the controller
    must discriminate spectral content (e.g. auto-EQ) this is structurally
    insufficient and the model collapses to outputting the conditional-mean
    correction. SpectralDynamicController replaces the pool-of-abs frontend
    with a Hann-windowed rfft per block, projects the magnitude spectrum to
    a small embedding, and feeds a wider LSTM with a linear head.

    Same hidden-state contract as DynamicController so the export pipeline
    (`STATEFUL_CLASSES` / `collect_stateful`) picks it up automatically.
    """

    def __init__(
        self,
        num_control_params: int,
        block_size: int = 128,
        num_layers: int = 2,
        hidden_dim: int = 64,
        feat_dim: int = 32,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = 0
        self.num_control_params = num_control_params
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.lr_multiplier = lr_multiplier

        n_freq = block_size // 2 + 1

        # Hann window applied per-block before the DFT.
        window = torch.hann_window(block_size)
        # Manual real-input DFT basis ([block_size, n_freq] each for real/imag
        # components). torch.fft.rfft is not exportable via TorchScript ONNX at
        # opset 17, but a fixed-size matmul is — and at block_size=128 this is
        # cheap. The window is folded into the basis so the runtime cost is a
        # single (block_size, n_freq) GEMM per block, twice (real + imag).
        idx_t = torch.arange(block_size, dtype=torch.float32)
        idx_f = torch.arange(n_freq, dtype=torch.float32)
        ang = 2.0 * torch.pi * idx_t.unsqueeze(1) * idx_f.unsqueeze(0) / block_size
        # [block_size, n_freq]
        dft_real = torch.cos(ang) * window.unsqueeze(1)
        dft_imag = -torch.sin(ang) * window.unsqueeze(1)
        self.register_buffer("dft_real", dft_real, persistent=False)
        self.register_buffer("dft_imag", dft_imag, persistent=False)

        self.feat_proj = torch.nn.Linear(n_freq, feat_dim)
        self.lstm = torch.nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.head = torch.nn.Linear(hidden_dim, num_control_params)
        self.act = torch.nn.Sigmoid()

        self.hidden_state = (torch.zeros(1,), torch.zeros(1,))
        self.is_hidden_state_init = False

    def _featurize(self, x: torch.Tensor) -> torch.Tensor:
        bs, chs, T = x.shape
        n_blocks = T // self.block_size
        # [bs, n_blocks, block_size]
        x_blocks = x.view(bs, n_blocks, self.block_size)
        # [bs, n_blocks, n_freq] — manual real-input DFT (window folded in).
        re = x_blocks @ self.dft_real
        im = x_blocks @ self.dft_imag
        mag = torch.sqrt(re * re + im * im + 1e-12)
        log_mag = torch.log1p(mag)
        return self.feat_proj(log_mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, chs, seq_len = x.shape
        assert chs == 1
        if (seq_len % self.block_size) != 0:
            pad = self.block_size - (seq_len % self.block_size)
            x = torch.nn.functional.pad(x, (0, pad))

        feat = self._featurize(x)         # [bs, n_blocks, feat_dim]
        feat = feat.permute(1, 0, 2)      # [n_blocks, bs, feat_dim]
        if self.is_hidden_state_init:
            out, new_hidden = self.lstm(feat, self.hidden_state)
        else:
            out, new_hidden = self.lstm(feat)
        params = self.act(self.head(out))  # [n_blocks, bs, num_control_params]
        params = params.permute(1, 2, 0)   # [bs, num_control_params, n_blocks]
        params = params.repeat_interleave(self.block_size, dim=-1)
        self.update_state(new_hidden)
        return params[..., :seq_len]

    def reset_states(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        self.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple(h.detach() for h in self.hidden_state)

    def update_state(self, new_hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True


# -----------------------------------------------------------------------------
# Dynamic Conditional Controller
# -----------------------------------------------------------------------------


class DynamicCondController(torch.nn.Module):
    """Conditional controller that maps input signal and input controls
    to a set of time-varying control parameters.
    """

    def __init__(
        self,
        num_controls: int,
        num_control_params: int,
        block_size: int = 128,
        num_layers: int = 1,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.num_control_params = num_control_params
        self.block_size = block_size
        self.num_layers = num_layers
        self.lr_multiplier = lr_multiplier

        self.hidden_state = (
            torch.zeros(
                1,
            ),
            torch.zeros(
                1,
            ),
        )  # initialized as a tensor for torchscript tracing
        self.is_hidden_state_init = False

        # used to downsample input
        self.pool = torch.nn.MaxPool1d(
            kernel_size=block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.lstm = torch.nn.LSTM(
            input_size=1 + num_controls,
            hidden_size=num_control_params,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )

        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        bs_x, chs_x, seq_len_x = x.shape
        bs_c, chs_c = controls.shape
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_controls

        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (seq_len_x % self.block_size) != 0:
            padding_size = self.block_size - (seq_len_x % self.block_size)
            x = torch.nn.functional.pad(x, (0, padding_size))

        nsteps = int(x.shape[-1] / self.block_size)

        # downsample signal
        x_down = self.pool(torch.abs(x))

        # upsample params [batch, nparams, nsteps]
        controls_up = controls.unsqueeze(-1).repeat(1, 1, nsteps)

        # concat along channel dim [batch, nchannels+nparams, nsteps]
        x_down = torch.cat((x_down, controls_up), dim=1)

        # shape for LSTM [length, batch, nchannels]
        x_down = x_down.permute(2, 0, 1)

        # control sequence
        if self.is_hidden_state_init:
            control_params, new_hidden_state = self.lstm(x_down, self.hidden_state)
        else:  # state was reset
            control_params, new_hidden_state = self.lstm(x_down)

        # put shape back [batch, channels, length]
        control_params = control_params.permute(1, 2, 0)

        # limit
        control_params = self.act(control_params)

        # upsample to original size
        control_params = control_params.repeat_interleave(self.block_size, dim=-1)
        self.update_state(new_hidden_state)
        return control_params[..., :seq_len_x]

    def reset_states(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self.is_hidden_state_init = False

    def detach_states(self) -> None:
        self.detach_state()

    def detach_state(self) -> None:
        if self.is_hidden_state_init:
            self.hidden_state = tuple((h.detach() for h in self.hidden_state))

    def update_state(self, new_hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self.hidden_state = new_hidden
        self.is_hidden_state_init = True
