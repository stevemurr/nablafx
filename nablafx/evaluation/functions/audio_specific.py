"""
Audio-Specific Evaluation Functions

This module contains audio-specific evaluation functions particularly useful
for guitar/instrument effects modeling.
"""

import torch
from auraloss.utils import apply_reduction
from ..registry import register_function


@register_function("edc_loss", differentiable=True)
class EDCLoss(torch.nn.Module):
    """Energy Decay Curve loss function module.

    Particularly useful for guitar/instrument effects that alter decay characteristics
    such as reverb pedals, amp modeling, and sustain effects.
    """

    def __init__(self, remove_dc=True, eps=1e-8, clamp=True, min_db=-80, max_db=0, error_fcn=None, reduction="mean"):
        super().__init__()
        self.remove_dc = remove_dc
        self.eps = eps
        self.clamp = clamp
        self.min_db = min_db
        self.max_db = max_db
        self.error_fcn = error_fcn if error_fcn is not None else torch.nn.MSELoss()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.remove_dc:
            input = input - torch.mean(input, dim=-1, keepdim=True)
            target = target - torch.mean(target, dim=-1, keepdim=True)

        if self.clamp:
            input = torch.abs(input)
            input = 20.0 * torch.log10(input + self.eps)
            input = torch.clamp(input, self.min_db, self.max_db)
            input = torch.pow(10.0, input / 20.0)

            target = torch.abs(target)
            target = 20.0 * torch.log10(target + self.eps)
            target = torch.clamp(target, self.min_db, self.max_db)
            target = torch.pow(10.0, target / 20.0)

        # Schroeder's energy decay curve
        input_energy = 10.0 * torch.log10(torch.sum(torch.square(input)) + self.eps)
        target_energy = 10.0 * torch.log10(torch.sum(torch.square(target)) + self.eps)

        input_edc = (
            10.0 * torch.log10(torch.flip(torch.cumsum(torch.flip(torch.square(input), dims=[-1]), dim=-1), dims=[-1])) - input_energy
        )
        target_edc = (
            10.0 * torch.log10(torch.flip(torch.cumsum(torch.flip(torch.square(target), dims=[-1]), dim=-1), dims=[-1])) - target_energy
        )

        losses = self.error_fcn(input_edc, target_edc)
        losses = apply_reduction(losses, reduction=self.reduction)

        return losses


@register_function("spectral_centroid_loss", differentiable=True)
class SpectralCentroidLoss(torch.nn.Module):
    """Spectral Centroid loss function.

    Measures the difference in the "brightness" of the sound, which is particularly
    important for guitar effects that change the tonal character (EQ, distortion, etc.).
    """

    def __init__(self, n_fft=1024, hop_length=None, eps=1e-8, reduction="mean"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle batch dimensions
        original_shape = input.shape
        if len(original_shape) > 1:
            input = input.reshape(-1, original_shape[-1])
            target = target.reshape(-1, original_shape[-1])

        # Compute STFT for each signal in the batch
        input_centroids = []
        target_centroids = []

        for i in range(input.size(0)):
            input_stft = torch.stft(input[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
            target_stft = torch.stft(target[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

            # Compute magnitude spectrograms
            input_mag = torch.abs(input_stft)
            target_mag = torch.abs(target_stft)

            # Frequency bins (normalized)
            freqs = torch.linspace(0, 1, input_mag.size(0), device=input.device).unsqueeze(1)

            # Compute spectral centroids
            input_centroid = torch.sum(freqs * input_mag, dim=0) / (torch.sum(input_mag, dim=0) + self.eps)
            target_centroid = torch.sum(freqs * target_mag, dim=0) / (torch.sum(target_mag, dim=0) + self.eps)

            input_centroids.append(input_centroid.mean())
            target_centroids.append(target_centroid.mean())

        input_centroids = torch.stack(input_centroids)
        target_centroids = torch.stack(target_centroids)

        # Compute loss
        loss = torch.nn.functional.mse_loss(input_centroids, target_centroids)
        loss = apply_reduction(loss, reduction=self.reduction)

        return loss


@register_function("spectral_rolloff_loss", differentiable=True)
class SpectralRolloffLoss(torch.nn.Module):
    """Spectral Rolloff loss function.

    Measures the frequency below which a specified percentage of the total spectral energy lies.
    Important for modeling effects that change the high-frequency content (filters, amp simulation).
    """

    def __init__(self, n_fft=1024, hop_length=None, rolloff_percent=0.85, eps=1e-8, reduction="mean"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.rolloff_percent = rolloff_percent
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle batch dimensions
        original_shape = input.shape
        if len(original_shape) > 1:
            input = input.reshape(-1, original_shape[-1])
            target = target.reshape(-1, original_shape[-1])

        # Compute STFT for each signal in the batch
        input_rolloffs = []
        target_rolloffs = []

        def compute_rolloff(magnitude):
            # Compute cumulative energy
            cumsum = torch.cumsum(magnitude, dim=0)
            total_energy = cumsum[-1:, :]

            # Find rolloff frequency
            threshold = self.rolloff_percent * total_energy
            rolloff_idx = torch.argmax((cumsum >= threshold).float(), dim=0)
            return rolloff_idx.float() / magnitude.size(0)

        for i in range(input.size(0)):
            input_stft = torch.stft(input[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
            target_stft = torch.stft(target[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

            # Compute magnitude spectrograms
            input_mag = torch.abs(input_stft)
            target_mag = torch.abs(target_stft)

            input_rolloff = compute_rolloff(input_mag).mean()
            target_rolloff = compute_rolloff(target_mag).mean()

            input_rolloffs.append(input_rolloff)
            target_rolloffs.append(target_rolloff)

        input_rolloffs = torch.stack(input_rolloffs)
        target_rolloffs = torch.stack(target_rolloffs)

        # Compute loss
        loss = torch.nn.functional.mse_loss(input_rolloffs, target_rolloffs)
        loss = apply_reduction(loss, reduction=self.reduction)

        return loss


@register_function("a_weighting_loss", differentiable=True)
class AWeightingLoss(torch.nn.Module):
    """A-weighting perceptual loss function.

    Applies A-weighting curve to emphasize frequencies that are more perceptually important.
    Useful for guitar modeling where perceptual accuracy is more important than raw spectral accuracy.
    """

    def __init__(self, n_fft=1024, hop_length=None, sample_rate=22050, reduction="mean"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.sample_rate = sample_rate
        self.reduction = reduction

        # Compute A-weighting curve
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        # A-weighting formula (approximation)
        f2 = freqs**2
        a_weight = 1.2588966 * 148840000 * f2**2 / ((f2 + 424.36) * torch.sqrt((f2 + 11599.29) * (f2 + 544496.41)) * (f2 + 148840000))
        a_weight_db = 20 * torch.log10(a_weight + 1e-10)
        a_weight_db = a_weight_db - torch.max(a_weight_db)  # Normalize to 0 dB max
        self.register_buffer("a_weight", 10 ** (a_weight_db / 20))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle batch dimensions
        original_shape = input.shape
        if len(original_shape) > 1:
            input = input.reshape(-1, original_shape[-1])
            target = target.reshape(-1, original_shape[-1])

        # Compute STFT for each signal in the batch
        input_losses = []
        target_losses = []

        for i in range(input.size(0)):
            input_stft = torch.stft(input[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
            target_stft = torch.stft(target[i], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

            # Compute magnitude spectrograms
            input_mag = torch.abs(input_stft)
            target_mag = torch.abs(target_stft)

            # Apply A-weighting
            input_weighted = input_mag * self.a_weight.unsqueeze(1)
            target_weighted = target_mag * self.a_weight.unsqueeze(1)

            # Compute MSE loss in A-weighted domain
            loss = torch.nn.functional.mse_loss(input_weighted, target_weighted)
            input_losses.append(loss)

        # Average loss across batch
        loss = torch.stack(input_losses).mean()
        loss = apply_reduction(loss, reduction=self.reduction)

        return loss


@register_function("zero_crossing_rate_metric", differentiable=False, requires_no_grad=True)
class ZeroCrossingRateMetric(torch.nn.Module):
    """Zero Crossing Rate metric.

    Measures the rate at which the signal changes sign. Useful for analyzing
    the roughness/noisiness of distorted guitar sounds.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def compute_zcr(signal):
            # Compute zero crossings
            sign_changes = torch.diff(torch.sign(signal), dim=-1)
            zcr = torch.sum(torch.abs(sign_changes), dim=-1) / (2 * signal.size(-1))
            return zcr

        input_zcr = compute_zcr(input)
        target_zcr = compute_zcr(target)

        # Compute absolute difference as metric
        diff = torch.abs(input_zcr - target_zcr)
        diff = apply_reduction(diff, reduction=self.reduction)

        return diff


@register_function("fad_metric", differentiable=False, requires_no_grad=True)
class FADMetric(torch.nn.Module):
    """Frechet Audio Distance metric.

    Measures the distance between distributions of audio features extracted by
    pre-trained models. Particularly useful for evaluating perceptual quality
    of generated audio in guitar/instrument effects modeling.

    This metric works with pre-saved audio directories (like those created by
    the system's audio logging) rather than computing from tensors directly.
    """

    def __init__(self, model_name="vggish", sample_rate=16000, ckpt_dir=None, use_pca=False, use_activation=False, verbose=False, **kwargs):
        super().__init__()
        try:
            from frechet_audio_distance import FrechetAudioDistance

            self.FrechetAudioDistance = FrechetAudioDistance
        except ImportError:
            raise ImportError(
                "frechet_audio_distance package is required for FAD metric. " "Install with: pip install frechet-audio-distance"
            )

        # Validate model_name against supported models
        supported_models = ["vggish", "pann", "clap"]
        if model_name not in supported_models:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                f"Currently supported models: {supported_models}. "
                f"Note: AFX-Rep is not supported by the current package version."
            )

        self.model_name = model_name
        self.sample_rate = sample_rate

        # Set default checkpoint directory following system.py pattern
        if ckpt_dir is None:
            import os

            parent_dir = os.path.abspath(os.getcwd())
            self.ckpt_dir = os.path.join(parent_dir, "checkpoints_fad")
        else:
            self.ckpt_dir = ckpt_dir

        self.use_pca = use_pca
        self.use_activation = use_activation
        self.verbose = verbose
        self.kwargs = kwargs

        # Initialize FAD model
        self._init_fad_model()

    def _init_fad_model(self):
        """Initialize the FAD model based on the specified model name."""
        import os

        # Filter kwargs to only include parameters supported by FrechetAudioDistance
        supported_kwargs = {}
        fad_supported_params = ["use_pca", "use_activation", "audio_load_worker", "submodel_name", "enable_fusion"]
        for param in fad_supported_params:
            if param in self.kwargs:
                supported_kwargs[param] = self.kwargs[param]

        if self.model_name == "vggish":
            self.fad_model = self.FrechetAudioDistance(
                ckpt_dir=self.ckpt_dir,
                model_name="vggish",
                sample_rate=self.sample_rate,
                use_pca=self.use_pca,
                use_activation=self.use_activation,
                verbose=self.verbose,
                **supported_kwargs,
            )
        elif self.model_name == "pann":
            self.fad_model = self.FrechetAudioDistance(
                ckpt_dir=self.ckpt_dir, model_name="pann", sample_rate=self.sample_rate, verbose=self.verbose, **supported_kwargs
            )
        elif self.model_name == "clap":
            # CLAP is now supported with the updated package
            clap_kwargs = supported_kwargs.copy()
            # Ensure required CLAP parameters
            if "submodel_name" not in clap_kwargs:
                clap_kwargs["submodel_name"] = "630k-audioset"  # Default submodel
            self.fad_model = self.FrechetAudioDistance(
                ckpt_dir=self.ckpt_dir, model_name="clap", sample_rate=self.sample_rate, verbose=self.verbose, **clap_kwargs
            )
        elif self.model_name == "afx-rep":
            # Note: AFX-Rep is still not supported by the current package version
            raise ValueError(
                f"AFX-Rep model is not supported by the current frechet_audio_distance package. " f"Supported models: vggish, pann, clap."
            )
        else:
            raise ValueError(
                f"Unsupported FAD model: {self.model_name}. " f"Supported models: vggish, pann, clap (AFX-Rep is not supported)"
            )

    def compute_fad_from_directories(self, target_dir: str, pred_dir: str) -> torch.Tensor:
        """
        Compute FAD score from pre-saved audio directories.

        Args:
            target_dir: Path to directory containing target/reference audio files
            pred_dir: Path to directory containing predicted/generated audio files

        Returns:
            FAD score as a torch.Tensor
        """
        import os

        # Check if directories exist and contain files
        if not os.path.exists(target_dir) or not os.path.exists(pred_dir):
            if self.verbose:
                print(f"Warning: Audio directories not found. Target: {target_dir}, Pred: {pred_dir}")
            return torch.tensor(float("inf"), dtype=torch.float32)

        # Check if directories contain audio files
        target_files = [f for f in os.listdir(target_dir) if f.endswith((".wav", ".mp3", ".flac"))]
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith((".wav", ".mp3", ".flac"))]

        if len(target_files) == 0 or len(pred_files) == 0:
            if self.verbose:
                print(f"Warning: No audio files found. Target files: {len(target_files)}, Pred files: {len(pred_files)}")
            return torch.tensor(float("inf"), dtype=torch.float32)

        try:
            if self.verbose:
                print(f"Computing FAD ({self.model_name}) between {target_dir} and {pred_dir}")

            fad_score = self.fad_model.score(target_dir, pred_dir)

            if self.verbose:
                print(f"FAD score ({self.model_name}): {fad_score}")

            return torch.tensor(fad_score, dtype=torch.float32)

        except Exception as e:
            if self.verbose:
                print(f"FAD computation failed: {e}")
            return torch.tensor(float("inf"), dtype=torch.float32)

    def compute_fad_from_logger_context(self, logger, mode: str = "val") -> torch.Tensor:
        """
        Compute FAD score using the logger context (following system.py pattern).

        Args:
            logger: The logger object with experiment.dir attribute
            mode: Evaluation mode ("val", "test", etc.)

        Returns:
            FAD score as a torch.Tensor
        """
        import os

        # Extract directories following system.py pattern
        run_dir = logger.experiment.dir
        pred_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/pred")
        target_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/target")

        return self.compute_fad_from_directories(target_dir, pred_dir)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward method for compatibility with metric interface.

        Note: This method cannot compute FAD from tensors directly as FAD requires
        pre-saved audio files. Use compute_fad_from_directories() or
        compute_fad_from_logger_context() instead.
        """
        if self.verbose:
            print("Warning: FADMetric.forward() called with tensors. " "FAD requires pre-saved audio files. Returning infinity.")
        return torch.tensor(float("inf"), dtype=torch.float32)


@register_function("fad_vggish_metric", differentiable=False, requires_no_grad=True)
class FADVGGishMetric(FADMetric):
    """VGGish-based Frechet Audio Distance metric."""

    def __init__(self, ckpt_dir=None, use_pca=False, use_activation=False, **kwargs):
        super().__init__(
            model_name="vggish", sample_rate=16000, ckpt_dir=ckpt_dir, use_pca=use_pca, use_activation=use_activation, **kwargs
        )


@register_function("fad_pann_metric", differentiable=False, requires_no_grad=True)
class FADPANNMetric(FADMetric):
    """PANN-based Frechet Audio Distance metric."""

    def __init__(self, ckpt_dir=None, **kwargs):
        super().__init__(model_name="pann", sample_rate=32000, ckpt_dir=ckpt_dir, **kwargs)


# Note: AFX-Rep metric is disabled - not supported by current frechet_audio_distance package


@register_function("fad_clap_metric", differentiable=False, requires_no_grad=True)
class FADCLAPMetric(FADMetric):
    """CLAP-based Frechet Audio Distance metric."""

    def __init__(self, ckpt_dir=None, submodel_name="630k-audioset", enable_fusion=False, **kwargs):
        super().__init__(
            model_name="clap", sample_rate=48000, ckpt_dir=ckpt_dir, submodel_name=submodel_name, enable_fusion=enable_fusion, **kwargs
        )


# @register_function("fad_afxrep_metric", differentiable=False, requires_no_grad=True)
# class FADAFXRepMetric(FADMetric):
#     """AFX-Rep-based Frechet Audio Distance metric."""
#
#     def __init__(self, ckpt_dir=None, **kwargs):
#         super().__init__(model_name="afx-rep", sample_rate=48000, ckpt_dir=ckpt_dir, **kwargs)
