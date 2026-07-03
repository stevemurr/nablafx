import torch
from rational.torch import Rational
from typing import Tuple, Dict, Optional

from .dsp import denormalize_parameters, biquad, sosfilt, sosfilt_via_fsm, lfilter_via_fsm
from .siren import Modulator, SirenNet

# -----------------------------------------------------------------------------
# BASIC
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Phase Shift (WIP)
# -----------------------------------------------------------------------------


class PhaseShift(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_shift: float = -180.0,
        max_shift: float = +180.0,
        control_type: str = "static",
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"shift": (min_shift, max_shift)}
        self.control_type = control_type
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {"shift": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, shift: torch.Tensor, train: bool = False) -> torch.Tensor:
        bs, chs, seq_len = x.size()
        shift_rad = shift.view(bs, chs, -1) * (torch.pi / 180.0)
        shift_rad = shift_rad.repeat(1, 1, seq_len)
        return x * torch.cos(shift_rad) + torch.roll(x, 1, dims=-1) * torch.sin(shift_rad)


# -----------------------------------------------------------------------------
# Phase Invertion
# -----------------------------------------------------------------------------


class PhaseInversion(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_control_params = 0
        self.control_type = None

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return -x


# -----------------------------------------------------------------------------
# Gain
# -----------------------------------------------------------------------------


class Gain(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -32.0,
        max_gain_db: float = 32.0,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"gain_db": (min_gain_db, max_gain_db)}
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {"gain_db": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, gain_db: torch.Tensor, train: bool = False) -> torch.Tensor:
        bs, chs, seq_len = x.size()
        gain_ln = 10 ** (gain_db.view(bs, chs, -1) / 20.0)
        return x * gain_ln


# -----------------------------------------------------------------------------
# DC Offset
# -----------------------------------------------------------------------------


class DCOffset(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_offset: float = -2.0,
        max_offset: float = +2.0,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {"offset": (min_offset, max_offset)}
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 1

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {"offset": params[:, 0, :]}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x
        param_dict = self.get_param_dict(control_params)
        output = self.process(x, **param_dict, train=train)
        return output, param_dict

    def process(self, x: torch.Tensor, offset: torch.Tensor, train: bool = False) -> torch.Tensor:
        bs, chs, seq_len = x.size()
        return x + offset.view(bs, chs, -1)


# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parametric EQ
# -----------------------------------------------------------------------------


class ParametricEQ(torch.nn.Module):
    """Parametric EQ with low-shelving, 3 peakings, and high-shelving filters."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
        freeze_freqs: bool = False,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond", "dynamic-spectral"]
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.freeze_freqs = freeze_freqs
        self.param_ranges = {
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20.0, 2000.0),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (20.0, 200.0),
            "band0_q_factor": (min_q_factor, max_q_factor),
            "band1_gain_db": (min_gain_db, max_gain_db),
            "band1_cutoff_freq": (200.0, 2000.0),
            "band1_q_factor": (min_q_factor, max_q_factor),
            "band2_gain_db": (min_gain_db, max_gain_db),
            "band2_cutoff_freq": (2000.0, 12000.0),
            "band2_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000.0, 16000.0),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
        }
        # When freeze_freqs=True, collapse each cutoff range to its midpoint so
        # denormalize_parameters always returns the fixed frequency regardless of
        # the incoming [0,1] control value.
        if freeze_freqs:
            for key in list(self.param_ranges.keys()):
                if key.endswith("_cutoff_freq"):
                    lo, hi = self.param_ranges[key]
                    mid = 0.5 * (lo + hi)
                    self.param_ranges[key] = (mid, mid)
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 15

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond", "dynamic-spectral"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "low_shelf_gain_db": params[:, 0, :],
            "low_shelf_cutoff_freq": params[:, 1, :],
            "low_shelf_q_factor": params[:, 2, :],
            "band0_gain_db": params[:, 3, :],
            "band0_cutoff_freq": params[:, 4, :],
            "band0_q_factor": params[:, 5, :],
            "band1_gain_db": params[:, 6, :],
            "band1_cutoff_freq": params[:, 7, :],
            "band1_q_factor": params[:, 8, :],
            "band2_gain_db": params[:, 9, :],
            "band2_cutoff_freq": params[:, 10, :],
            "band2_q_factor": params[:, 11, :],
            "high_shelf_gain_db": params[:, 12, :],
            "high_shelf_cutoff_freq": params[:, 13, :],
            "high_shelf_q_factor": params[:, 14, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        band1_gain_db: torch.Tensor,
        band1_cutoff_freq: torch.Tensor,
        band1_q_factor: torch.Tensor,
        band2_gain_db: torch.Tensor,
        band2_cutoff_freq: torch.Tensor,
        band2_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        train: bool = False,
    ):
        # since we are learning parameter we compute coefficients on the fly
        sos = self.compute_coefficients(
            self.sample_rate,
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            band1_gain_db,
            band1_cutoff_freq,
            band1_q_factor,
            band2_gain_db,
            band2_cutoff_freq,
            band2_q_factor,
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        band1_gain_db: torch.Tensor,
        band1_cutoff_freq: torch.Tensor,
        band1_q_factor: torch.Tensor,
        band2_gain_db: torch.Tensor,
        band2_cutoff_freq: torch.Tensor,
        band2_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
    ):
        bs = low_shelf_gain_db.size(0)

        # five second order sections
        sos = torch.zeros(bs, 5, 6).type_as(low_shelf_gain_db)
        # ------------ low shelf ------------
        b, a = biquad(
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 1, :] = torch.cat((b, a), dim=-1)
        # ------------ band1 ------------
        b, a = biquad(
            band1_gain_db,
            band1_cutoff_freq,
            band1_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 2, :] = torch.cat((b, a), dim=-1)
        # ------------ band2 ------------
        b, a = biquad(
            band2_gain_db,
            band2_cutoff_freq,
            band2_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 3, :] = torch.cat((b, a), dim=-1)
        # ------------ high shelf ------------
        b, a = biquad(
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            sample_rate,
            "high_shelf",
        )
        sos[:, 4, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# SSL 9000 J Console EQ (grey-box, knob-conditioned)
# -----------------------------------------------------------------------------


class SSLConsoleEQ(torch.nn.Module):
    """Differentiable model of the SSL 9000 J channel-strip EQ topology:

        HPF -> LF(shelf|bell) -> LMF(bell) -> HMF(bell) -> HF(shelf|bell) -> LPF

    Driven by a knob-conditioned controller (``control_type="static-cond"``): a
    small MLP maps the console knob vector to the ``num_control_params`` [0,1]
    values below, which denormalize to physical (freq/gain/Q) and build a biquad
    cascade recomputed at ``sample_rate`` — so the model is sample-rate correct
    and its coefficients transfer to the C++ runtime (mirrors ``dsp.biquad``).

    LF and HF each switch shelf<->bell via a learned blend channel ``*_bellmix``
    (0 = shelf, 1 = bell): the section sos is ``mix*bell + (1-mix)*shelf`` (both
    biquad-normalized to a0=1, so the blend is well-defined and exact at 0/1).
    HPF/LPF are ``hpf_sections``/``lpf_sections`` cascaded biquads sharing one
    cutoff+Q (the SSL HPF measures ~18 dB/oct — see docs/ssl_eq_phase0_findings).

    The harmonic/analog coloration (``thd_db``) is intentionally NOT here; it is a
    separate nonlinearity processor in the grey-box chain (see the ssl_eq config).
    """

    _CTRL = ["static", "static-cond", "dynamic", "dynamic-cond"]

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        block_size: int = 128,
        control_type: str = "static-cond",
        lr_multiplier: float = 1.0,
        hpf_sections: int = 2,
        lpf_sections: int = 1,
    ):
        super().__init__()
        assert control_type in self._CTRL
        self.sample_rate = sample_rate
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.block_size = block_size
        self.hpf_sections = hpf_sections
        self.lpf_sections = lpf_sections
        g = (min_gain_db, max_gain_db)
        # SSL-physical ranges (encompass the console's measured behaviour; the
        # controller maps the knob vector into these). See Phase-0 findings.
        self.param_ranges = {
            "hpf_freq": (10.0, 800.0),   "hpf_q": (0.4, 1.4),
            "lf_gain": g, "lf_freq": (30.0, 600.0),    "lf_q": (0.2, 2.0),  "lf_bellmix": (0.0, 1.0),
            "lmf_gain": g, "lmf_freq": (60.0, 3000.0), "lmf_q": (0.1, 4.0),
            "hmf_gain": g, "hmf_freq": (400.0, 20000.0), "hmf_q": (0.1, 4.0),
            "hf_gain": g, "hf_freq": (1500.0, 20000.0), "hf_q": (0.2, 2.0),  "hf_bellmix": (0.0, 1.0),
            "lpf_freq": (2000.0, 23000.0), "lpf_q": (0.4, 1.4),
        }
        self._keys = list(self.param_ranges.keys())
        self.num_control_params = len(self._keys)  # 18

        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    # -- param unpacking -------------------------------------------------------
    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {k: params[:, i, :] for i, k in enumerate(self._keys)}
        # bell-mix channels stay in [0,1]; everything else denormalizes physically
        mix = {k: param_dict.pop(k) for k in ("lf_bellmix", "hf_bellmix")}
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        param_dict.update(mix)
        return param_dict

    # -- coefficient construction ---------------------------------------------
    def compute_coefficients(self, sample_rate, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        bs = p["lf_gain"].size(0)
        zero = torch.zeros_like(p["lf_gain"])

        def blended(gain, freq, q, mix, high):
            shelf = torch.cat(biquad(gain, freq, q, sample_rate,
                                     "high_shelf" if high else "low_shelf"), dim=-1)
            bell = torch.cat(biquad(gain, freq, q, sample_rate, "peaking"), dim=-1)
            m = mix.view(bs, 1)
            return m * bell + (1.0 - m) * shelf

        def passf(freq, q, high):
            return torch.cat(biquad(zero, freq, q, sample_rate,
                                    "high_pass" if high else "low_pass"), dim=-1)

        sections = []
        sections += [passf(p["hpf_freq"], p["hpf_q"], True)] * self.hpf_sections
        sections.append(blended(p["lf_gain"], p["lf_freq"], p["lf_q"], p["lf_bellmix"], False))
        sections.append(torch.cat(biquad(p["lmf_gain"], p["lmf_freq"], p["lmf_q"], sample_rate, "peaking"), dim=-1))
        sections.append(torch.cat(biquad(p["hmf_gain"], p["hmf_freq"], p["hmf_q"], sample_rate, "peaking"), dim=-1))
        sections.append(blended(p["hf_gain"], p["hf_freq"], p["hf_q"], p["hf_bellmix"], True))
        sections += [passf(p["lpf_freq"], p["lpf_q"], False)] * self.lpf_sections

        return torch.stack(sections, dim=1)  # (bs, n_sos, 6)

    def process(self, x, param_dict, train: bool = False):
        sos = self.compute_coefficients(self.sample_rate, param_dict)
        out = sosfilt_via_fsm(sos, x) if train else sosfilt(sos, x)
        # The frequency-sampling filter returns a non-contiguous view; a
        # downstream nonlinearity (rational-activations CUDA kernel) calls
        # x.view(-1) which requires contiguity. Make the audio well-behaved.
        return out.contiguous()

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False):
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c and chs_x == 1 and chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, param_dict, train=train)
        else:
            if (seq_len_x % self.block_size) != 0:
                pad = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, pad))
                control_params = torch.nn.functional.pad(control_params, (0, pad), mode="replicate")
            control_params = self.pool(control_params)
            num_blocks = x.shape[-1] // self.block_size
            output, pdl = [], []
            for i in range(num_blocks):
                xb = x[:, :, i * self.block_size:(i + 1) * self.block_size]
                pd = self.get_param_dict(control_params[:, :, i:i + 1])
                output.append(self.process(xb, pd, train=train))
                pdl.append(pd)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            param_dict = {k: torch.cat([pd[k] for pd in pdl], dim=-1) for k in pdl[0]}
        return output, param_dict

    # -- analytic helper: magnitude response in dB at arbitrary freqs ----------
    def magnitude_db(self, control_params: torch.Tensor, freqs_hz: torch.Tensor) -> torch.Tensor:
        """(bs, n_freq) dB magnitude of the cascade — for TF-loss / anchoring / tests."""
        pd = self.get_param_dict(control_params)
        sos = self.compute_coefficients(self.sample_rate, pd)  # (bs, n_sos, 6)
        w = (2 * torch.pi * freqs_hz / self.sample_rate).view(1, 1, -1)
        z1 = torch.exp(-1j * w)
        z2 = z1 * z1
        b = sos[..., 0:3].unsqueeze(-1)   # (bs, n_sos, 3, 1)
        a = sos[..., 3:6].unsqueeze(-1)
        num = b[:, :, 0] + b[:, :, 1] * z1 + b[:, :, 2] * z2
        den = a[:, :, 0] + a[:, :, 1] * z1 + a[:, :, 2] * z2
        h = (num / den).prod(dim=1)       # (bs, n_freq)
        return 20 * torch.log10(h.abs().clamp_min(1e-9))


# -----------------------------------------------------------------------------
# Spectral Mask EQ
# -----------------------------------------------------------------------------


class SpectralMaskEQ(torch.nn.Module):
    """STFT-domain magnitude-mask EQ with N mel-spaced bands.

    The controller emits ``num_bands`` sigmoid values per block. Those are
    de-normalized to per-band gain (dB), interpolated/expanded to a per-FFT-bin
    gain mask, and applied to the magnitude spectrum (phase preserved). iSTFT
    with overlap-add reconstructs the audio. Differentiable via torch.stft /
    torch.istft.

    Designed to mirror the C++ ``spectral_mask_eq.hpp`` runtime: same n_fft,
    hop, mel band edges, and zero-phase magnitude-only application. The C++
    runtime computes its mel matrix the same way (HTK formula) so the trained
    weights are directly usable.
    """

    def __init__(
        self,
        sample_rate: float,
        n_bands: int = 32,
        n_fft: int = 1024,
        hop: int = 512,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        block_size: int = 128,
        control_type: str = "dynamic-spectral",
        lr_multiplier: float = 1.0,
        f_min: float = 30.0,
        f_max: float | None = None,
    ):
        super().__init__()
        assert control_type in ["dynamic", "dynamic-spectral", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.hop = hop
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = n_bands

        # Mel filterbank: each row picks/weights linear FFT bins for one band.
        n_freq = n_fft // 2 + 1
        if f_max is None:
            f_max = sample_rate / 2.0
        mel_min = 2595.0 * float(torch.log10(torch.tensor(1.0 + f_min / 700.0)))
        mel_max = 2595.0 * float(torch.log10(torch.tensor(1.0 + f_max / 700.0)))
        mel_pts = torch.linspace(mel_min, mel_max, n_bands + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        bin_pts = (hz_pts * (n_fft / sample_rate)).clamp(0, n_freq - 1)
        # band_to_bin: triangular weighting matrix [n_bands, n_freq]
        band_to_bin = torch.zeros(n_bands, n_freq)
        idx = torch.arange(n_freq, dtype=torch.float32)
        for b in range(n_bands):
            left, center, right = bin_pts[b], bin_pts[b + 1], bin_pts[b + 2]
            up = (idx - left) / torch.clamp(center - left, min=1e-6)
            dn = (right - idx) / torch.clamp(right - center, min=1e-6)
            band_to_bin[b] = torch.clamp(torch.minimum(up, dn), min=0.0)
        # bin_to_band: maps per-bin gain back to per-band query — but we need the
        # opposite direction (per-band -> per-bin). Build that via row-normalized
        # transpose so each FFT bin sums its assigned bands' gains weighted by
        # the triangular overlap.
        # Specifically: bin_gain[k] = sum_b band_gain[b] * band_to_bin[b, k] /
        # sum_b band_to_bin[b, k]. Precompute the normalization here.
        bin_norm = band_to_bin.sum(dim=0).clamp(min=1e-6)  # [n_freq]
        self.register_buffer("band_to_bin", band_to_bin, persistent=False)
        self.register_buffer("bin_norm", bin_norm, persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        # No human-readable param dict — gains are anonymous mel bands. Provide
        # a flat dict for parity with the EQ logging callbacks.
        return {f"band_{b:02d}_gain_db": params[:, b, :] for b in range(self.n_bands)}

    def _band_to_bin_mask(self, band_gain_db: torch.Tensor) -> torch.Tensor:
        """band_gain_db: [bs, n_bands] -> per-bin linear gain [bs, n_freq]."""
        # band_gain_db @ band_to_bin -> per-bin sum-of-band gains in dB-weighted
        # form; normalize by overlap so gain stays in dB units.
        per_bin_db = (band_gain_db @ self.band_to_bin) / self.bin_norm
        return torch.pow(10.0, per_bin_db / 20.0)  # [bs, n_freq]

    def forward(
        self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, T = x.shape
        bs_c, n_chs_c, T_c = control_params.shape
        assert chs == 1
        assert bs_c == bs
        assert n_chs_c == self.num_control_params

        # Pad audio so length is a multiple of hop and we can run a clean iSTFT.
        # torch.stft with center=True, hop=H, n_fft=N gives n_frames = T // H + 1.
        if T % self.hop != 0:
            pad = self.hop - (T % self.hop)
            x = torch.nn.functional.pad(x, (0, pad))
            control_params = torch.nn.functional.pad(
                control_params, (0, pad), mode="replicate"
            )
            T_padded = T + pad
        else:
            T_padded = T

        # Convert sigmoid [0,1] params to per-band gain in dB.
        gain_db = self.min_gain_db + control_params * (self.max_gain_db - self.min_gain_db)
        # Downsample to per-frame: each STFT frame i (centered) covers samples
        # around [i*hop - n_fft/2, i*hop + n_fft/2]. Take the gain at i*hop as
        # representative. With center=True, frame 0 is at sample 0.
        # Simpler: average within each hop-sized window.
        # gain_db: [bs, n_bands, T_padded]
        n_frames = T_padded // self.hop + 1  # matches torch.stft(center=True)
        # Reshape into [bs, n_bands, n_frames-1, hop] for averaging the first
        # (n_frames-1) frames; the last frame uses the trailing hop's mean.
        # Take gain at frame center indices [0, hop, 2*hop, ...].
        # Prepend a hop/2-padded version so frame 0 gets a leading mean too.
        # Cheap version: pick gain at each frame's hop-aligned sample.
        idx = torch.arange(n_frames, device=x.device) * self.hop
        idx = idx.clamp(max=T_padded - 1)
        gain_db_frames = gain_db.index_select(-1, idx)   # [bs, n_bands, n_frames]

        # STFT
        x_flat = x.view(bs, T_padded)
        spec = torch.stft(
            x_flat, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=self.window,
            return_complex=True, center=True,
        )  # [bs, n_freq, n_frames]
        # Per-frame per-bin gain mask.
        # gain_db_frames -> [bs, n_frames, n_bands]
        gain_db_frames_t = gain_db_frames.permute(0, 2, 1)
        # mask: [bs, n_frames, n_freq]
        mask = self._band_to_bin_mask(
            gain_db_frames_t.reshape(bs * gain_db_frames_t.shape[1], self.n_bands)
        ).view(bs, gain_db_frames_t.shape[1], -1)
        # Align mask to spec frames
        mask = mask.permute(0, 2, 1)  # [bs, n_freq, n_frames]
        # Defensive: if mask has more or fewer frames than spec (rare with
        # center=True boundary), trim/pad to match.
        if mask.shape[-1] != spec.shape[-1]:
            target = spec.shape[-1]
            if mask.shape[-1] > target:
                mask = mask[..., :target]
            else:
                last = mask[..., -1:].expand(-1, -1, target - mask.shape[-1])
                mask = torch.cat([mask, last], dim=-1)

        spec_out = spec * mask
        y = torch.istft(
            spec_out, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=self.window,
            length=T_padded, center=True,
        )
        y = y.view(bs, 1, T_padded)
        y = y[..., :T]
        return y, self.get_param_dict(control_params[..., :T])


# -----------------------------------------------------------------------------
# Shelving EQ
# -----------------------------------------------------------------------------


class ShelvingEQ(torch.nn.Module):
    """EQ with high-pass, low-shelving, high-shelving and low-pass filters."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.param_ranges = {
            "highpass_cutoff_freq": (20.0, 2000.0),
            "highpass_q_factor": (min_q_factor, max_q_factor),
            "low_shelf_gain_db": (min_gain_db, max_gain_db),
            "low_shelf_cutoff_freq": (20.0, 4000.0),
            "low_shelf_q_factor": (min_q_factor, max_q_factor),
            "high_shelf_gain_db": (min_gain_db, max_gain_db),
            "high_shelf_cutoff_freq": (4000.0, 24000.0),
            "high_shelf_q_factor": (min_q_factor, max_q_factor),
            "lowpass_cutoff_freq": (4000.0, 24000.0),
            "lowpass_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 10

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "highpass_cutoff_freq": params[:, 0, :],
            "highpass_q_factor": params[:, 1, :],
            "low_shelf_gain_db": params[:, 2, :],
            "low_shelf_cutoff_freq": params[:, 3, :],
            "low_shelf_q_factor": params[:, 4, :],
            "high_shelf_gain_db": params[:, 5, :],
            "high_shelf_cutoff_freq": params[:, 6, :],
            "high_shelf_q_factor": params[:, 7, :],
            "lowpass_cutoff_freq": params[:, 8, :],
            "lowpass_q_factor": params[:, 9, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        highpass_cutoff_freq: torch.Tensor,
        highpass_q_factor: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        lowpass_cutoff_freq: torch.Tensor,
        lowpass_q_factor: torch.Tensor,
        train: bool = False,
    ):
        # since we are learning parameter we compute coefficients on the fly
        sos = self.compute_coefficients(
            self.sample_rate,
            highpass_cutoff_freq,
            highpass_q_factor,
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            lowpass_cutoff_freq,
            lowpass_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        highpass_cutoff_freq: torch.Tensor,
        highpass_q_factor: torch.Tensor,
        low_shelf_gain_db: torch.Tensor,
        low_shelf_cutoff_freq: torch.Tensor,
        low_shelf_q_factor: torch.Tensor,
        high_shelf_gain_db: torch.Tensor,
        high_shelf_cutoff_freq: torch.Tensor,
        high_shelf_q_factor: torch.Tensor,
        lowpass_cutoff_freq: torch.Tensor,
        lowpass_q_factor: torch.Tensor,
    ):
        bs = low_shelf_gain_db.size(0)

        # four second order sections
        sos = torch.zeros(bs, 4, 6).type_as(highpass_cutoff_freq)
        # ------------ highpass ------------
        b, a = biquad(
            torch.zeros_like(highpass_cutoff_freq),  # gain_db
            highpass_cutoff_freq,
            highpass_q_factor,
            sample_rate,
            "highpass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)
        # ------------ low shelf ------------
        b, a = biquad(
            low_shelf_gain_db,
            low_shelf_cutoff_freq,
            low_shelf_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 1, :] = torch.cat((b, a), dim=-1)
        # ------------ high shelf ------------
        b, a = biquad(
            high_shelf_gain_db,
            high_shelf_cutoff_freq,
            high_shelf_q_factor,
            sample_rate,
            "high_shelf",
        )
        sos[:, 2, :] = torch.cat((b, a), dim=-1)
        # ------------ lowpass ------------
        b, a = biquad(
            torch.zeros_like(lowpass_cutoff_freq),  # gain_db
            lowpass_cutoff_freq,
            lowpass_q_factor,
            sample_rate,
            "lowpass",
        )
        sos[:, 3, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Peak/Notch
# -----------------------------------------------------------------------------


class Peaking(torch.nn.Module):
    """Peaking filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 20000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_gain_db.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_gain_db)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "peaking",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Lowpass
# -----------------------------------------------------------------------------


class Lowpass(torch.nn.Module):
    """Lowpass filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_cutoff_freq: float = 2000.0,
        max_cutoff_freq: float = 20000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 2

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "cutoff_freq": params[:, 0, :],
            "q_factor": params[:, 1, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        cutoff_freq: torch.Tensor,
        q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            cutoff_freq,
            q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        cutoff_freq: torch.Tensor,
        q_factor: torch.Tensor,
    ):
        bs = cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            torch.zeros_like(cutoff_freq),  # gain_db
            cutoff_freq,
            q_factor,
            sample_rate,
            "low_pass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Highpass
# -----------------------------------------------------------------------------


class Highpass(torch.nn.Module):
    """Highpass filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 2

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "band0_cutoff_freq": params[:, 0, :],
            "band0_q_factor": params[:, 1, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            torch.zeros_like(band0_cutoff_freq),  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "high_pass",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Lowshelf
# -----------------------------------------------------------------------------


class Lowshelf(torch.nn.Module):
    """Lowshelf filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Highshelf
# -----------------------------------------------------------------------------


class Highshelf(torch.nn.Module):
    """Highshelf filter from biquad section."""

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        min_cutoff_freq: float = 200.0,
        max_cutoff_freq: float = 2000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 10.0,
        block_size: int = 128,
        control_type: str = "static",
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        assert control_type in ["static", "static-cond", "dynamic", "dynamic-cond"]
        self.sample_rate = sample_rate
        self.param_ranges = {
            "band0_gain_db": (min_gain_db, max_gain_db),
            "band0_cutoff_freq": (min_cutoff_freq, max_cutoff_freq),
            "band0_q_factor": (min_q_factor, max_q_factor),
        }
        self.block_size = block_size
        self.control_type = control_type
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 3

        # used to downsample control_params
        if control_type in ["dynamic", "dynamic-cond"]:
            self.pool = torch.nn.AvgPool1d(kernel_size=block_size)

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        param_dict = {
            "band0_gain_db": params[:, 0, :],
            "band0_cutoff_freq": params[:, 1, :],
            "band0_q_factor": params[:, 2, :],
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        return param_dict

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs_x, chs_x, seq_len_x = x.size()
        bs_c, chs_c, seq_len_c = control_params.size()
        assert bs_x == bs_c
        assert chs_x == 1
        assert chs_c == self.num_control_params
        assert seq_len_c == 1 if self.control_type in ["static", "static-cond"] else seq_len_c == seq_len_x

        if self.control_type in ["static", "static-cond"]:
            param_dict = self.get_param_dict(control_params)
            output = self.process(x, **param_dict, train=train)
        else:
            # pad if not multiple of block_size
            if (seq_len_x % self.block_size) != 0:
                padding_size = self.block_size - (seq_len_x % self.block_size)
                x = torch.nn.functional.pad(x, (0, padding_size))
                control_params = torch.nn.functional.pad(control_params, (0, padding_size), mode="replicate")

            # downsample control_params
            control_params = self.pool(control_params)

            # process block by block
            num_blocks = x.shape[-1] // self.block_size
            output = []
            param_dict_list = []
            for i in range(num_blocks):
                x_block = x[:, :, i * self.block_size : (i + 1) * self.block_size]
                control_params_block = control_params[:, :, i : i + 1]
                param_dict = self.get_param_dict(control_params_block)
                output_block = self.process(x_block, **param_dict, train=train)
                output.append(output_block)
                param_dict_list.append(param_dict)
            output = torch.cat(output, dim=-1)[..., :seq_len_x]
            # concat all parameters along sequence axis
            param_dict = {}
            for k in param_dict_list[0].keys():
                param_dict[k] = torch.concat([param_dict[k] for param_dict in param_dict_list], dim=-1)

        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
        train: bool = False,
    ):
        sos = self.compute_coefficients(
            self.sample_rate,
            band0_gain_db,
            band0_cutoff_freq,
            band0_q_factor,
        )

        # apply filters
        if train:
            x_out = sosfilt_via_fsm(sos, x)
        else:
            x_out = sosfilt(sos, x)

        return x_out

    @staticmethod
    def compute_coefficients(
        sample_rate,
        band0_gain_db: torch.Tensor,
        band0_cutoff_freq: torch.Tensor,
        band0_q_factor: torch.Tensor,
    ):
        bs = band0_cutoff_freq.size(0)

        # one second order sections
        sos = torch.zeros(bs, 1, 6).type_as(band0_cutoff_freq)
        # ------------ band0 ------------
        b, a = biquad(
            band0_gain_db,  # gain_db
            band0_cutoff_freq,
            band0_q_factor,
            sample_rate,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        return sos


# -----------------------------------------------------------------------------
# Static FIR Filter
# -----------------------------------------------------------------------------


class StaticFIRFilter(torch.nn.Module):

    def __init__(
        self,
        sample_rate: float,
        n_taps: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        w0_initial: float = 30.0,
        pretrained: str = None,
        lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_taps = n_taps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.w0_initial = w0_initial
        self.pretrained = pretrained
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 0
        self.control_type = None

        self.net = SirenNet(1, hidden_dim, 1, num_layers, w0_initial=w0_initial)

        coords = torch.linspace(-1, 1, n_taps).view(1, 1, -1)
        self.register_buffer("coords", coords)

        if pretrained is not None:
            state_dict_path = pretrained
            print(f"\nLoading weights from {state_dict_path}")
            self.load_state_dict(torch.load(state_dict_path))

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def extract_impulse_response(self, x: torch.Tensor):
        bs = x.size(0)
        impulse = self.net(self.coords.repeat(bs, 1, 1).permute(0, 2, 1))
        impulse = impulse.permute(0, 2, 1).squeeze(1)
        return impulse

    def process(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        b = self.extract_impulse_response(x)
        output = lfilter_via_fsm(x, b)
        return output


# -----------------------------------------------------------------------------
# NONLINEARITIES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Tanh
# -----------------------------------------------------------------------------


class TanhNonlinearity(torch.nn.Module):
    def __init__(self, sample_rate: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_control_params = 0
        self.control_type = None

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train)

    def process(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return torch.tanh(x), self.get_param_dict(None)


# -----------------------------------------------------------------------------
# Static MLP Nonlinearity
# -----------------------------------------------------------------------------


class StaticMLPNonlinearity(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        hidden_dim: int = 64,
        num_layers: int = 3,
        w0_initial: float = 30.0,
        pretrained: str = None,
        lr_multiplier=1.0,
    ):
        """
        Single learnable nonlinearity with no external conditioning parameters..

        Notes: For efficiecny, we could consider creating a lookup table for the
        output of the MLP and then use that to compute the output of the MLP.

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.w0_initial = w0_initial
        self.pretrained = pretrained
        self.lr_multiplier = lr_multiplier
        self.num_control_params = 0
        self.control_type = None

        self.net = SirenNet(1, hidden_dim, 1, num_layers, w0_initial=w0_initial)

        if pretrained is not None:
            state_dict_path = pretrained
            print(f"Loading weights from {state_dict_path}")
            self.load_state_dict(torch.load(state_dict_path))

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# -----------------------------------------------------------------------------
# Static Rational Nonlinearity
# -----------------------------------------------------------------------------


class StaticRationalNonlinearity(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        degrees: Tuple[int, int] = (6, 5),
        init_approx_func: str = "tanh",
        lr_multiplier=1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.degrees = degrees
        self.init_approx_func = init_approx_func
        self.num_control_params = 0
        self.lr_multiplier = lr_multiplier
        self.control_type = None

        self.net = Rational(init_approx_func, degrees, version="A")

    def get_param_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, x: torch.Tensor, control_params: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()
        assert chs == 1
        assert control_params is None
        return self.process(x, train=train), self.get_param_dict(control_params)

    def process(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return self.net(x)
