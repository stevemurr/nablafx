// STFT-domain magnitude-mask EQ with N mel-spaced bands.
//
// Mirrors the Python `SpectralMaskEQ` processor: same n_fft, hop, mel band
// edges (HTK formula, f_min..f_max), Hann analysis+synthesis windows, and
// zero-phase magnitude-only application. Backed by Apple Accelerate vDSP for
// the real-input FFT — macOS-only, which matches build.sh's macOS-only
// constraint.
//
// Streaming contract:
//   - The host calls process(in, out, n) with arbitrary `n` (commonly 128).
//   - Internally we accumulate samples into an n_fft-sized analysis ring; on
//     every `hop` accumulated samples we run an FFT frame, apply the per-band
//     gain mask (set via set_params), inverse-FFT, window, and OLA into an
//     output ring.
//   - process() pulls `n` samples of finished output. Latency = n_fft - hop.
//
// Pure DSP — no CLAP / ORT / std::variant deps so it can be unit-tested
// standalone.

#pragma once

#include <Accelerate/Accelerate.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "meta.hpp"

namespace nablafx {

class SpectralMaskEq {
public:
    SpectralMaskEq() = default;
    ~SpectralMaskEq() {
        if (fft_setup_) vDSP_destroy_fftsetup(fft_setup_);
    }

    SpectralMaskEq(const SpectralMaskEq&)            = delete;
    SpectralMaskEq& operator=(const SpectralMaskEq&) = delete;

    void reset(const SpectralMaskEqParams& cfg) {
        cfg_ = cfg;
        if (cfg_.n_fft <= 0 || (cfg_.n_fft & (cfg_.n_fft - 1)) != 0) {
            throw std::runtime_error(
                "spectral_mask_eq: n_fft must be a power of two, got " +
                std::to_string(cfg_.n_fft));
        }
        if (cfg_.hop <= 0 || cfg_.hop > cfg_.n_fft) {
            throw std::runtime_error("spectral_mask_eq: hop must be in (0, n_fft]");
        }

        n_fft_     = cfg_.n_fft;
        hop_       = cfg_.hop;
        n_bands_   = cfg_.n_bands;
        n_freq_    = n_fft_ / 2 + 1;
        log2_nfft_ = static_cast<vDSP_Length>(std::log2(static_cast<double>(n_fft_)));

        if (fft_setup_) vDSP_destroy_fftsetup(fft_setup_);
        fft_setup_ = vDSP_create_fftsetup(log2_nfft_, kFFTRadix2);
        if (!fft_setup_) {
            throw std::runtime_error("spectral_mask_eq: vDSP_create_fftsetup failed");
        }

        window_.assign(n_fft_, 0.0f);
        for (int n = 0; n < n_fft_; ++n) {
            window_[n] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * n / n_fft_));
        }

        in_ring_.assign(n_fft_, 0.0f);
        out_ring_.assign(n_fft_ + hop_, 0.0f);
        norm_ring_.assign(n_fft_ + hop_, 0.0f);  // OLA window² accumulator
        in_fill_         = 0;
        samples_since_   = 0;
        out_write_       = 0;
        out_read_        = 0;
        out_avail_       = 0;

        // Mel filterbank.
        build_mel_(cfg_.sample_rate, n_fft_, n_bands_, cfg_.f_min, cfg_.f_max);

        // Per-bin gain mask (linear), starts as unity.
        bin_gain_.assign(n_freq_, 1.0f);

        // Scratch buffers for FFT.
        windowed_.assign(n_fft_, 0.0f);
        split_real_.assign(n_fft_ / 2, 0.0f);
        split_imag_.assign(n_fft_ / 2, 0.0f);
        time_out_.assign(n_fft_, 0.0f);

        // vDSP forward+inverse round-trip scale is 2*n_fft (Apple vDSP guide:
        // "divide by 2n to recover original values after inverse"). We apply
        // 1/(2*n_fft) in the OLA write and then divide per-sample by the
        // accumulated sum of window² — matching torch.istft's normalization —
        // so output is correctly scaled regardless of the COLA sum varying
        // between 0.5 and 1.0 over the hop cycle.
        ola_scale_ = 1.0f / (2.0f * static_cast<float>(n_fft_));
    }

    // Apply latest controller output: ``params`` holds n_bands sigmoid values
    // in [0, 1]. Updates the per-bin linear gain mask.
    void set_params(const float* params, std::size_t n) {
        if (static_cast<int>(n) != cfg_.num_control_params) {
            throw std::runtime_error(
                "spectral_mask_eq::set_params: expected " +
                std::to_string(cfg_.num_control_params) +
                ", got " + std::to_string(n));
        }
        // Per-band gain in dB, then per-bin via mel_band_to_bin_ matrix.
        const float gain_span = cfg_.max_gain_db - cfg_.min_gain_db;
        // Per-band dB
        std::vector<float> band_db(n_bands_, 0.0f);
        for (int b = 0; b < n_bands_; ++b) {
            float g = params[b];
            if (g < 0.0f) g = 0.0f;
            if (g > 1.0f) g = 1.0f;
            band_db[b] = cfg_.min_gain_db + g * gain_span;
        }
        // bin_db[k] = sum_b band_to_bin_[b, k] * band_db[b] / bin_norm_[k]
        for (int k = 0; k < n_freq_; ++k) {
            float sum = 0.0f;
            for (int b = 0; b < n_bands_; ++b) {
                sum += band_to_bin_[b * n_freq_ + k] * band_db[b];
            }
            const float bin_db = (bin_norm_[k] > 1e-6f) ? (sum / bin_norm_[k]) : 0.0f;
            bin_gain_[k] = std::pow(10.0f, bin_db / 20.0f);
        }
    }

    // In-place safe.
    void process(const float* in, float* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            // Push input into analysis ring.
            in_ring_[in_fill_] = in[i];
            in_fill_ = (in_fill_ + 1) % n_fft_;
            ++samples_since_;

            // Run an FFT frame every `hop` samples.
            if (samples_since_ >= hop_) {
                samples_since_ -= hop_;
                run_frame_();
            }

            // Hand back one sample. Divide by accumulated window² to match
            // torch.istft per-sample normalisation; guards against near-zero
            // norm at Hann window edges.
            if (out_avail_ > 0) {
                const int   rd   = out_read_;
                const float norm = norm_ring_[rd];
                out[i] = (norm > 1e-8f) ? (out_ring_[rd] / norm) : 0.0f;
                out_ring_[rd]  = 0.0f;
                norm_ring_[rd] = 0.0f;
                out_read_ = (rd + 1) % static_cast<int>(out_ring_.size());
                --out_avail_;
            } else {
                out[i] = 0.0f;
            }
        }
    }

    // Sample the current per-bin gain mask at n arbitrary frequencies (Hz)
    // and return linear→dB values. Used to populate the 5-band display.
    void sample_gains_db(const float* hz_arr, float* db_arr, int n) const {
        for (int i = 0; i < n; ++i) {
            int bin = static_cast<int>(
                std::round(hz_arr[i] * n_fft_ / static_cast<float>(cfg_.sample_rate)));
            bin = std::max(0, std::min(bin, n_freq_ - 1));
            const float g = bin_gain_[bin];
            db_arr[i] = (g > 1e-8f) ? 20.0f * std::log10(g) : -80.0f;
        }
    }

    int latency_samples() const { return n_fft_ - hop_; }
    int block_size() const { return cfg_.block_size; }
    int num_control_params() const { return cfg_.num_control_params; }

private:
    void run_frame_() {
        // Copy ring (oldest first) into windowed_.
        for (int n = 0; n < n_fft_; ++n) {
            const int src = (in_fill_ + n) % n_fft_;
            windowed_[n] = in_ring_[src] * window_[n];
        }

        // Pack real input into split form for vDSP.
        DSPSplitComplex split{split_real_.data(), split_imag_.data()};
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(windowed_.data()), 2,
                  &split, 1, n_fft_ / 2);

        // Forward FFT (in-place split-complex).
        vDSP_fft_zrip(fft_setup_, &split, 1, log2_nfft_, kFFTDirection_Forward);

        // After zrip-forward: split_real_[0] = DC,
        //                    split_imag_[0] = Nyquist real (packed),
        //                    split_real_[k]+i*split_imag_[k] = bin k for 0 < k < n_fft/2
        // Apply mask. Note: vDSP packs Nyquist into [0].imag.
        const float dc_mag      = split.realp[0];
        const float nyquist_mag = split.imagp[0];
        split.realp[0] = dc_mag      * bin_gain_[0];
        split.imagp[0] = nyquist_mag * bin_gain_[n_freq_ - 1];
        for (int k = 1; k < n_fft_ / 2; ++k) {
            split.realp[k] *= bin_gain_[k];
            split.imagp[k] *= bin_gain_[k];
        }

        // Inverse FFT.
        vDSP_fft_zrip(fft_setup_, &split, 1, log2_nfft_, kFFTDirection_Inverse);

        // Unpack split-complex back into time domain (interleaved).
        vDSP_ztoc(&split, 1,
                  reinterpret_cast<DSPComplex*>(time_out_.data()), 2,
                  n_fft_ / 2);

        // Hann²-OLA: accumulate audio (scaled by 1/(2N)) and window² into
        // parallel rings. Per-sample division in process() normalises away the
        // varying Hann² COLA sum (0.5–1.0 for hop=N/2), mirroring torch.istft.
        const int ring_sz = static_cast<int>(out_ring_.size());
        for (int n = 0; n < n_fft_; ++n) {
            const int idx = (out_write_ + n) % ring_sz;
            out_ring_[idx]  += time_out_[n] * window_[n] * ola_scale_;
            norm_ring_[idx] += window_[n] * window_[n];
        }
        out_write_ = (out_write_ + hop_) % ring_sz;
        out_avail_ += hop_;
    }

    void build_mel_(int sr, int n_fft, int n_bands, float f_min, float f_max) {
        const int n_freq = n_fft / 2 + 1;
        const float mel_min = 2595.0f * std::log10(1.0f + f_min / 700.0f);
        const float mel_max = 2595.0f * std::log10(1.0f + f_max / 700.0f);
        std::vector<float> mel_pts(n_bands + 2, 0.0f);
        std::vector<float> hz_pts(n_bands + 2, 0.0f);
        std::vector<float> bin_pts(n_bands + 2, 0.0f);
        for (int i = 0; i < n_bands + 2; ++i) {
            mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_bands + 1);
            hz_pts[i]  = 700.0f * (std::pow(10.0f, mel_pts[i] / 2595.0f) - 1.0f);
            bin_pts[i] = hz_pts[i] * (n_fft / static_cast<float>(sr));
            if (bin_pts[i] < 0.0f) bin_pts[i] = 0.0f;
            if (bin_pts[i] > n_freq - 1) bin_pts[i] = n_freq - 1;
        }
        band_to_bin_.assign(n_bands * n_freq, 0.0f);
        for (int b = 0; b < n_bands; ++b) {
            const float left   = bin_pts[b];
            const float center = bin_pts[b + 1];
            const float right  = bin_pts[b + 2];
            const float l_span = std::max(center - left,  1e-6f);
            const float r_span = std::max(right  - center, 1e-6f);
            for (int k = 0; k < n_freq; ++k) {
                const float kf = static_cast<float>(k);
                const float up = (kf - left) / l_span;
                const float dn = (right - kf) / r_span;
                float w = std::min(up, dn);
                if (w < 0.0f) w = 0.0f;
                band_to_bin_[b * n_freq + k] = w;
            }
        }
        bin_norm_.assign(n_freq, 0.0f);
        for (int b = 0; b < n_bands; ++b) {
            for (int k = 0; k < n_freq; ++k) {
                bin_norm_[k] += band_to_bin_[b * n_freq + k];
            }
        }
    }

    SpectralMaskEqParams cfg_{};
    int n_fft_{0}, hop_{0}, n_bands_{0}, n_freq_{0};
    vDSP_Length log2_nfft_{0};
    FFTSetup fft_setup_{nullptr};

    std::vector<float> window_;       // Hann
    std::vector<float> in_ring_;      // n_fft circular
    int                in_fill_{0};
    int                samples_since_{0};

    std::vector<float> out_ring_;     // OLA audio accumulator (n_fft + hop)
    std::vector<float> norm_ring_;   // OLA window² accumulator (same size)
    int                out_write_{0};
    int                out_read_{0};
    int                out_avail_{0};

    std::vector<float> band_to_bin_;  // [n_bands * n_freq]
    std::vector<float> bin_norm_;     // [n_freq]
    std::vector<float> bin_gain_;     // [n_freq] linear

    // FFT scratch
    std::vector<float> windowed_;
    std::vector<float> split_real_;
    std::vector<float> split_imag_;
    std::vector<float> time_out_;

    float ola_scale_{1.0f};
};

}  // namespace nablafx
