// Composite TONE CLAP plugin: 1 dylib that wires
//
//   audio → LufsLeveler → ort(autoeq controller) → ParametricEq5Band
//                       → RationalA (saturator)  → ort(la2a)
//                       → TruePeakCeiling → output trim
//
// The composite has two host-exposed knobs (AMT, TRM) defined in the
// composite_meta. AMT remaps to per-stage params (saturator pre/post + wet/dry,
// LA-2A peak reduction, auto-EQ wet/dry); TRM is a final linear gain.
//
// Block-rate streaming: the auto-EQ controller and the LA-2A LSTM both want
// fixed 128-sample blocks (cond_block_size). The plugin accumulates host
// audio into a 128-sample input ring per channel and flushes the chain block
// by block; output samples come out of an output ring with the same depth.
// Total internal latency = 128 (one block of accumulator) + ceiling lookahead.
//
// v1 limitations:
//   - arm64 macOS only (parent CMakeLists guards against other platforms)
//   - CPU execution provider only
//   - per-block parameter snapshot (no sample-accurate smoothing)
//   - refuses activation if host sample rate != composite_meta.sample_rate

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlfcn.h>

#include <clap/clap.h>
#include <onnxruntime_cxx_api.h>

#include "composite_meta.hpp"
#include "lufs_leveler.hpp"
#include "meta.hpp"
#include "param_id.hpp"
#include "parametric_eq_5band.hpp"
#include "rational_a.hpp"
#include "true_peak_ceiling.hpp"

namespace nablafx_tone {

namespace fs = std::filesystem;
using nablafx::CompositeMeta;
using nablafx::ControlSpec;
using nablafx::LufsLeveler;
using nablafx::ParametricEq5Band;
using nablafx::ParametricEq5BandParams;
using nablafx::PluginMeta;
using nablafx::RationalA;
using nablafx::RationalAParams;
using nablafx::TruePeakCeiling;
using nablafx::load_composite_meta;
using nablafx::load_meta;
using nablafx::param_id_for;

// All ORT sessions in this plugin process audio in fixed kBlockSize chunks,
// which matches both the auto-EQ controller's cond_block_size and the LA-2A
// processor's TVFiLM cond_block_size. Changing this requires re-exporting both
// ONNX bundles at the new block size.
constexpr int kBlockSize = 128;

// ---------------------------------------------------------------------------
// Module-global state (loaded once at module init)
// ---------------------------------------------------------------------------

struct ModuleState {
    CompositeMeta              tone_meta;
    PluginMeta                 autoeq_meta;
    PluginMeta                 sat_meta;
    PluginMeta                 la2a_meta;
    std::string                bundle_dir;            // .../TONE.clap/Contents
    std::string                resources_dir;         // .../Contents/Resources
    std::string                plugin_id_str;         // "com.nablafx.<model_id>"
    clap_plugin_descriptor_t   descriptor{};
    std::vector<const char*>   feature_ptrs;
    std::array<const char*, 3> feature_storage{};
    std::unique_ptr<Ort::Env>  ort_env;
    // Pulled out of sat_meta once at load.
    RationalAParams            sat_rational;
    // Pulled out of autoeq_meta once at load.
    ParametricEq5BandParams    autoeq_eq;
};

static ModuleState* g_state = nullptr;

static std::string find_bundle_contents_() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&find_bundle_contents_), &info) == 0
        || !info.dli_fname) {
        return {};
    }
    fs::path dylib = info.dli_fname;
    // .clap/Contents/MacOS/<dylib> → .clap/Contents
    return dylib.parent_path().parent_path().string();
}

static void populate_descriptor_(ModuleState& st) {
    st.plugin_id_str = "com.nablafx." + st.tone_meta.model_id;

    st.feature_storage[0] = CLAP_PLUGIN_FEATURE_AUDIO_EFFECT;
    st.feature_storage[1] = CLAP_PLUGIN_FEATURE_MASTERING;
    st.feature_storage[2] = nullptr;
    st.feature_ptrs.assign(st.feature_storage.begin(), st.feature_storage.end());

    st.descriptor.clap_version = CLAP_VERSION_INIT;
    st.descriptor.id           = st.plugin_id_str.c_str();
    st.descriptor.name         = st.tone_meta.effect_name.c_str();
    st.descriptor.vendor       = "nablafx";
    st.descriptor.url          = "https://github.com/mcomunita/nablafx";
    st.descriptor.manual_url   = "";
    st.descriptor.support_url  = "";
    st.descriptor.version      = "1.0.0";
    st.descriptor.description  = "Composite TONE mastering plugin";
    st.descriptor.features     = st.feature_ptrs.data();
}

// ---------------------------------------------------------------------------
// Per-channel chain state. One of these per audio channel.
// ---------------------------------------------------------------------------

struct StateBuf {
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

class OrtMiniSession {
    // Thin wrapper around an Ort::Session for fixed-shape audio + state I/O.
    // Owns the input/output state buffers; you set audio + controls (if any),
    // then run() reads/writes state, and call swap() to make this run's
    // outputs the next run's inputs.
public:
    OrtMiniSession(Ort::Env& env, const std::string& model_path, const PluginMeta& meta)
        : env_(env), cpu_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
          meta_(meta) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ORT_SEQUENTIAL);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        for (const auto& nm : meta.input_names)  in_names_owned_.push_back(nm);
        for (const auto& nm : meta.output_names) out_names_owned_.push_back(nm);
        for (auto& s : in_names_owned_)  in_names_.push_back(s.c_str());
        for (auto& s : out_names_owned_) out_names_.push_back(s.c_str());

        // Allocate state buffers in/out. Shapes come from the meta; "_in"
        // matches "_out" 1:1 by stripping the suffix.
        for (const auto& s : meta.state_tensors) {
            int64_t n = 1;
            for (auto d : s.shape) n *= d;
            in_states_[s.name].shape  = s.shape;
            in_states_[s.name].data.assign(n, 0.0f);
            out_states_[s.name].shape = s.shape;
            out_states_[s.name].data.assign(n, 0.0f);
        }
    }

    StateBuf& in_state(const std::string& name) { return in_states_.at(name); }

    void reset_state() {
        for (auto& [_, b] : in_states_)  std::fill(b.data.begin(), b.data.end(), 0.0f);
        for (auto& [_, b] : out_states_) std::fill(b.data.begin(), b.data.end(), 0.0f);
    }

    // Run with caller-owned audio (and optional controls) buffers. Outputs
    // are written into `audio_out` (and the internal state-out buffers).
    void run(const float* audio_in, int audio_in_len,
             float* audio_out, int audio_out_len,
             const float* controls /*nullable*/, int n_controls,
             const std::string& audio_out_name = "audio_out") {
        std::vector<Ort::Value>      inputs;
        std::vector<const char*>     in_names;
        std::vector<const char*>     out_names;
        in_names.reserve(in_names_.size());
        out_names.reserve(out_names_.size());

        std::array<int64_t, 3> aud_shape{1, 1, audio_in_len};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu_, const_cast<float*>(audio_in), audio_in_len,
            aud_shape.data(), aud_shape.size()));
        in_names.push_back("audio_in");

        std::array<int64_t, 2> ctl_shape{1, n_controls};
        if (n_controls > 0) {
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, const_cast<float*>(controls), n_controls,
                ctl_shape.data(), ctl_shape.size()));
            in_names.push_back("controls");
        }

        // Add state inputs in the order the meta declared them.
        for (const auto& s : meta_.state_tensors) {
            auto& buf = in_states_[s.name];
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, buf.data.data(), static_cast<int64_t>(buf.data.size()),
                buf.shape.data(), buf.shape.size()));
            in_names.push_back(in_name_for_state_(s.name).c_str());
        }

        // Output: audio first, then states in declared order.
        for (const auto& nm : out_names_) out_names.push_back(nm);

        auto outs = session_->Run(Ort::RunOptions{nullptr},
                                  in_names.data(), inputs.data(), inputs.size(),
                                  out_names.data(), out_names.size());

        // Copy audio out (must match the requested length).
        // The first output is named audio_out_name; locate it by index.
        std::size_t audio_out_idx = 0;
        for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
            if (out_names_owned_[i] == audio_out_name) {
                audio_out_idx = i; break;
            }
        }
        const float* aud_out = outs[audio_out_idx].GetTensorData<float>();
        std::copy_n(aud_out, audio_out_len, audio_out);

        // Read back states by output-name (always "<state>_out").
        for (const auto& s : meta_.state_tensors) {
            const std::string out_name = out_name_for_state_(s.name);
            std::size_t idx = 0;
            for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
                if (out_names_owned_[i] == out_name) { idx = i; break; }
            }
            const float* p = outs[idx].GetTensorData<float>();
            std::copy_n(p, out_states_[s.name].data.size(), out_states_[s.name].data.begin());
        }
    }

    void swap_state() {
        for (const auto& s : meta_.state_tensors) {
            std::swap(in_states_[s.name].data, out_states_[s.name].data);
        }
    }

    // Run-arbitrary variant for the auto-EQ controller, where the audio
    // output channel name is "params_proc_0" and represents [1, 15, T]
    // sigmoid params instead of audio. We expose only the first sample
    // (all samples in a block are identical post-repeat_interleave).
    void run_controller(const float* audio_in, int audio_in_len,
                        float* params_out_first, int params_out_count) {
        std::vector<Ort::Value>  inputs;
        std::vector<const char*> in_names;
        std::vector<const char*> out_names;

        std::array<int64_t, 3> aud_shape{1, 1, audio_in_len};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu_, const_cast<float*>(audio_in), audio_in_len,
            aud_shape.data(), aud_shape.size()));
        in_names.push_back("audio_in");

        for (const auto& s : meta_.state_tensors) {
            auto& buf = in_states_[s.name];
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, buf.data.data(), static_cast<int64_t>(buf.data.size()),
                buf.shape.data(), buf.shape.size()));
            in_names.push_back(in_name_for_state_(s.name).c_str());
        }
        for (const auto& nm : out_names_) out_names.push_back(nm);

        auto outs = session_->Run(Ort::RunOptions{nullptr},
                                  in_names.data(), inputs.data(), inputs.size(),
                                  out_names.data(), out_names.size());

        // First output is the params tensor [1, 15, T]. We want the first
        // sample (all are identical within a block per the controller's
        // repeat_interleave structure).
        const float* p = outs[0].GetTensorData<float>();
        // Stride: T = audio_in_len, channel-major contiguous so element
        // [channel c, sample 0] is at offset c * T.
        for (int c = 0; c < params_out_count; ++c) {
            params_out_first[c] = p[c * audio_in_len + 0];
        }

        for (const auto& s : meta_.state_tensors) {
            const std::string out_name = out_name_for_state_(s.name);
            std::size_t idx = 0;
            for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
                if (out_names_owned_[i] == out_name) { idx = i; break; }
            }
            const float* sp = outs[idx].GetTensorData<float>();
            std::copy_n(sp, out_states_[s.name].data.size(), out_states_[s.name].data.begin());
        }
    }

private:
    static std::string in_name_for_state_(const std::string& base) {
        // e.g. "processor_h" → "processor_h_in"
        return base + "_in";
    }
    static std::string out_name_for_state_(const std::string& base) {
        return base + "_out";
    }

    Ort::Env&                     env_;
    Ort::MemoryInfo               cpu_;
    PluginMeta                    meta_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string>      in_names_owned_;
    std::vector<std::string>      out_names_owned_;
    std::vector<const char*>      in_names_;
    std::vector<const char*>      out_names_;
    std::unordered_map<std::string, StateBuf> in_states_;
    std::unordered_map<std::string, StateBuf> out_states_;
};

struct ChannelChain {
    // Per-channel stage instances. LufsLeveler and TruePeakCeiling each have
    // their own state and run unsmoothed across the host buffer (the leveler's
    // attack/release covers gain ramps; the ceiling has its own lookahead).
    LufsLeveler                            leveler;
    std::unique_ptr<OrtMiniSession>        autoeq_ort;
    ParametricEq5Band                      autoeq_eq;
    RationalA                              saturator;
    std::unique_ptr<OrtMiniSession>        la2a_ort;
    TruePeakCeiling                        ceiling;

    // 128-sample accumulator: kBlockSize input samples in, then a chain pass,
    // then kBlockSize output samples ready. Output ring fills before any reads
    // so the first kBlockSize host samples produce silence (latency reported
    // to the host so DAWs compensate).
    std::array<float, kBlockSize> in_buf{};
    int                           in_fill = 0;

    std::array<float, kBlockSize> out_buf{};
    int                           out_avail = 0;
    int                           out_read  = 0;
};

// ---------------------------------------------------------------------------
// Per-instance state
// ---------------------------------------------------------------------------

struct Plugin {
    clap_plugin_t      plugin{};
    const clap_host_t* host{nullptr};

    const CompositeMeta* meta{nullptr};
    int                  channels{2};
    double               sample_rate{};
    bool                 activated{false};

    // AMT, TRM in host-rate units (default 0.5, 0.0).
    std::vector<float>   control_values;

    std::vector<ChannelChain> chains;
};

// ---------------------------------------------------------------------------
// CLAP extension: audio ports — 1 stereo input, 1 stereo output
// ---------------------------------------------------------------------------

static uint32_t audio_ports_count(const clap_plugin_t*, bool /*is_input*/) { return 1; }

static bool audio_ports_get(const clap_plugin_t*, uint32_t index, bool is_input,
                            clap_audio_port_info_t* info) {
    if (index != 0) return false;
    info->id            = is_input ? 0 : 1;
    std::snprintf(info->name, sizeof(info->name), "%s", is_input ? "in" : "out");
    info->channel_count = 2;
    info->flags         = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type     = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    return true;
}
static const clap_plugin_audio_ports_t s_ext_audio_ports = {audio_ports_count, audio_ports_get};

// ---------------------------------------------------------------------------
// CLAP extension: params (AMT, TRM)
// ---------------------------------------------------------------------------

static uint32_t params_count(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    return static_cast<uint32_t>(plug->meta->controls.size());
}

static bool params_get_info(const clap_plugin_t* p, uint32_t index, clap_param_info_t* info) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (index >= plug->meta->controls.size()) return false;
    const auto& c = plug->meta->controls[index];
    info->id        = param_id_for(plug->meta->effect_name, c.id);
    info->flags     = CLAP_PARAM_IS_AUTOMATABLE;
    info->cookie    = nullptr;
    info->min_value = c.min;
    info->max_value = c.max;
    info->default_value = c.def;
    std::snprintf(info->name,   sizeof(info->name),   "%s", c.name.c_str());
    std::snprintf(info->module, sizeof(info->module), "%s", "");
    return true;
}

static bool params_get_value(const clap_plugin_t* p, clap_id id, double* value) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (param_id_for(plug->meta->effect_name, plug->meta->controls[i].id) == id) {
            *value = plug->control_values[i];
            return true;
        }
    }
    return false;
}

static bool params_value_to_text(const clap_plugin_t*, clap_id, double value, char* out, uint32_t out_size) {
    std::snprintf(out, out_size, "%.3f", value);
    return true;
}

static bool params_text_to_value(const clap_plugin_t*, clap_id, const char* text, double* out) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    *out = v;
    return true;
}

static void params_flush(const clap_plugin_t*, const clap_input_events_t*, const clap_output_events_t*) {}

static const clap_plugin_params_t s_ext_params = {
    params_count, params_get_info, params_get_value, params_value_to_text,
    params_text_to_value, params_flush,
};

// ---------------------------------------------------------------------------
// CLAP extension: latency
// ---------------------------------------------------------------------------

static uint32_t latency_get(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (plug->chains.empty()) return 0;
    // One full block of accumulator + the ceiling's lookahead.
    return static_cast<uint32_t>(kBlockSize + plug->chains[0].ceiling.latency_samples());
}

static const clap_plugin_latency_t s_ext_latency = {latency_get};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static bool plugin_init(const clap_plugin_t*) { return true; }
static void plugin_destroy(const clap_plugin_t* p) { delete static_cast<Plugin*>(p->plugin_data); }

static bool plugin_activate(const clap_plugin_t* p, double sample_rate,
                            uint32_t /*min_frames*/, uint32_t /*max_frames*/) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (static_cast<int>(std::lround(sample_rate)) != plug->meta->sample_rate) return false;

    plug->sample_rate = sample_rate;
    plug->control_values.assign(plug->meta->controls.size(), 0.0f);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        plug->control_values[i] = plug->meta->controls[i].def;
    }

    plug->chains.clear();
    plug->chains.resize(plug->channels);
    for (auto& ch : plug->chains) {
        ch.leveler = LufsLeveler(LufsLeveler::Config{
            /*target_lufs=*/g_state->tone_meta.leveler.target_lufs,
        });
        ch.leveler.reset(sample_rate, g_state->tone_meta.leveler.target_lufs);

        ch.autoeq_ort = std::make_unique<OrtMiniSession>(
            *g_state->ort_env,
            g_state->resources_dir + "/" + g_state->tone_meta.sub_bundles.at("auto_eq")
                + "/model.onnx",
            g_state->autoeq_meta);
        ch.autoeq_eq.reset(g_state->autoeq_eq);
        ch.saturator.reset(g_state->sat_rational.numerator,
                           g_state->sat_rational.denominator);

        ch.la2a_ort = std::make_unique<OrtMiniSession>(
            *g_state->ort_env,
            g_state->resources_dir + "/" + g_state->tone_meta.sub_bundles.at("la2a")
                + "/model.onnx",
            g_state->la2a_meta);

        TruePeakCeiling::Config tcfg{
            /*ceiling_dbtp=*/g_state->tone_meta.ceiling.ceiling_dbtp,
            /*lookahead_ms=*/g_state->tone_meta.ceiling.lookahead_ms,
            /*attack_ms=*/g_state->tone_meta.ceiling.attack_ms,
            /*release_ms=*/g_state->tone_meta.ceiling.release_ms,
        };
        ch.ceiling = TruePeakCeiling(tcfg);
        ch.ceiling.reset(sample_rate);

        ch.in_fill   = 0;
        ch.out_avail = 0;
        ch.out_read  = 0;
    }

    plug->activated = true;
    return true;
}

static void plugin_deactivate(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    plug->chains.clear();
    plug->activated = false;
}

static bool plugin_start_processing(const clap_plugin_t*) { return true; }
static void plugin_stop_processing(const clap_plugin_t*) {}

static void plugin_reset(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (auto& ch : plug->chains) {
        ch.leveler.reset(plug->sample_rate, g_state->tone_meta.leveler.target_lufs);
        if (ch.autoeq_ort) ch.autoeq_ort->reset_state();
        if (ch.la2a_ort)   ch.la2a_ort->reset_state();
        ch.ceiling.reset(plug->sample_rate);
        ch.in_fill   = 0;
        ch.out_avail = 0;
        ch.out_read  = 0;
        std::fill(ch.in_buf.begin(),  ch.in_buf.end(),  0.0f);
        std::fill(ch.out_buf.begin(), ch.out_buf.end(), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Block flush — chain one 128-sample block through every stage.
// ---------------------------------------------------------------------------

namespace {

// Snapshot of host-rate AMT/TRM resolved to per-stage parameters. Recomputed
// on every block; cheap.
struct AmountSnapshot {
    float autoeq_wet_mix;
    float sat_pre_db;
    float sat_post_db;
    float sat_wet_mix;
    float la2a_pr_norm;
    float la2a_comp_or_limit;
    float trim_lin;
};

AmountSnapshot resolve_amount_(const Plugin& plug) {
    // controls is keyed by index into meta->controls (AMT first, then TRM by
    // construction in composite.py). Look up by id to be safe.
    float amt = 0.5f, trm_db = 0.0f;
    for (size_t i = 0; i < plug.meta->controls.size(); ++i) {
        const auto& c = plug.meta->controls[i];
        if (c.id == "AMT") amt = std::clamp(plug.control_values[i], c.min, c.max);
        if (c.id == "TRM") trm_db = std::clamp(plug.control_values[i], c.min, c.max);
    }
    AmountSnapshot s{};
    s.autoeq_wet_mix     = amt * g_state->tone_meta.amt_autoeq.wet_mix_max;
    s.sat_pre_db         = amt * g_state->tone_meta.amt_sat.pre_gain_db_max;
    s.sat_post_db        = amt * g_state->tone_meta.amt_sat.post_gain_db_max;
    s.sat_wet_mix        = amt * g_state->tone_meta.amt_sat.wet_mix_max;
    const auto& la = g_state->tone_meta.amt_la2a;
    float pr_raw       = la.peak_reduction_min + amt * (la.peak_reduction_max - la.peak_reduction_min);
    s.la2a_pr_norm     = pr_raw / 100.0f;
    s.la2a_comp_or_limit = la.comp_or_limit;
    s.trim_lin         = std::pow(10.0f, trm_db / 20.0f);
    return s;
}

void flush_block_(ChannelChain& ch, const AmountSnapshot& amt) {
    // Scratch buffers reused per call (sized to kBlockSize at compile time).
    std::array<float, kBlockSize> dry{};
    std::array<float, kBlockSize> wet{};
    std::array<float, kBlockSize> blk{};

    // Stage 1: leveler — sample-by-sample, in place.
    ch.leveler.process(ch.in_buf.data(), blk.data(), kBlockSize);

    // Stage 2: auto-EQ — run controller ORT to get the per-block sigmoid
    // params, then biquad cascade. Wet/dry blend with the leveler output.
    std::array<float, 15> eq_params{};
    ch.autoeq_ort->run_controller(blk.data(), kBlockSize, eq_params.data(), 15);
    ch.autoeq_ort->swap_state();
    ch.autoeq_eq.set_params(eq_params.data(), eq_params.size());
    std::copy_n(blk.data(), kBlockSize, dry.data());
    ch.autoeq_eq.process(blk.data(), wet.data(), kBlockSize);
    for (int i = 0; i < kBlockSize; ++i) {
        blk[i] = (1.0f - amt.autoeq_wet_mix) * dry[i] + amt.autoeq_wet_mix * wet[i];
    }

    // Stage 3: saturator — pre/post gain + wet/dry mix.
    const float pre_lin  = std::pow(10.0f, amt.sat_pre_db  / 20.0f);
    const float post_lin = std::pow(10.0f, amt.sat_post_db / 20.0f);
    std::copy_n(blk.data(), kBlockSize, dry.data());
    for (int i = 0; i < kBlockSize; ++i) {
        wet[i] = static_cast<float>(ch.saturator.eval(static_cast<double>(blk[i] * pre_lin))) * post_lin;
    }
    for (int i = 0; i < kBlockSize; ++i) {
        blk[i] = (1.0f - amt.sat_wet_mix) * dry[i] + amt.sat_wet_mix * wet[i];
    }

    // Stage 4: LA-2A LSTM with controls = [C, P_norm].
    std::array<float, 2> la2a_controls{amt.la2a_comp_or_limit, amt.la2a_pr_norm};
    ch.la2a_ort->run(blk.data(), kBlockSize,
                     wet.data(), kBlockSize,
                     la2a_controls.data(), 2,
                     /*audio_out_name=*/"audio_out");
    ch.la2a_ort->swap_state();

    // Stage 5: ceiling — sample-by-sample, in place.
    ch.ceiling.process(wet.data(), ch.out_buf.data(), kBlockSize);
    ch.out_avail = kBlockSize;
    ch.out_read  = 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// process — accumulator-driven 128-sample block flushes
// ---------------------------------------------------------------------------

static void apply_events_(Plugin* plug, const clap_input_events_t* in_events) {
    if (!in_events) return;
    const uint32_t n = in_events->size(in_events);
    for (uint32_t i = 0; i < n; ++i) {
        const auto* hdr = in_events->get(in_events, i);
        if (!hdr) continue;
        if (hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;
        if (hdr->type != CLAP_EVENT_PARAM_VALUE) continue;
        const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        for (size_t k = 0; k < plug->meta->controls.size(); ++k) {
            if (param_id_for(plug->meta->effect_name, plug->meta->controls[k].id) == pv->param_id) {
                plug->control_values[k] = static_cast<float>(pv->value);
                break;
            }
        }
    }
}

static clap_process_status plugin_process(const clap_plugin_t* p, const clap_process_t* process) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    apply_events_(plug, process->in_events);

    const uint32_t n_frames = process->frames_count;
    if (n_frames == 0) return CLAP_PROCESS_CONTINUE;
    if (process->audio_inputs_count == 0 || process->audio_outputs_count == 0) {
        return CLAP_PROCESS_ERROR;
    }

    const float* const* in_ch  = process->audio_inputs[0].data32;
    float* const*       out_ch = process->audio_outputs[0].data32;
    const uint32_t in_channels  = std::min<uint32_t>(plug->channels, process->audio_inputs[0].channel_count);
    const uint32_t out_channels = std::min<uint32_t>(plug->channels, process->audio_outputs[0].channel_count);

    AmountSnapshot amt = resolve_amount_(*plug);

    for (uint32_t ch = 0; ch < in_channels && ch < out_channels; ++ch) {
        auto& chain = plug->chains[ch];
        const float* in_p = in_ch[ch];
        float*       out_p = out_ch[ch];
        uint32_t     i = 0;

        while (i < n_frames) {
            // Drain output ring first.
            while (i < n_frames && chain.out_read < chain.out_avail) {
                out_p[i++] = chain.out_buf[chain.out_read++];
            }
            if (i >= n_frames) break;

            // Push input samples until we either fill the input block or
            // exhaust the host frames.
            const uint32_t take = std::min<uint32_t>(n_frames - i, kBlockSize - chain.in_fill);
            std::copy_n(in_p + i, take, chain.in_buf.data() + chain.in_fill);
            i              += take;
            chain.in_fill  += take;

            if (chain.in_fill == kBlockSize) {
                flush_block_(chain, amt);
                chain.in_fill = 0;
            }
        }
    }
    return CLAP_PROCESS_CONTINUE;
}

static const void* plugin_get_extension(const clap_plugin_t*, const char* id) {
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &s_ext_audio_ports;
    if (std::strcmp(id, CLAP_EXT_PARAMS)       == 0) return &s_ext_params;
    if (std::strcmp(id, CLAP_EXT_LATENCY)      == 0) return &s_ext_latency;
    return nullptr;
}

static void plugin_on_main_thread(const clap_plugin_t*) {}

// ---------------------------------------------------------------------------
// Factory + entry
// ---------------------------------------------------------------------------

static const clap_plugin_t* factory_create_plugin(const clap_plugin_factory_t*,
                                                  const clap_host_t* host,
                                                  const char* plugin_id) {
    if (!g_state) return nullptr;
    if (std::strcmp(plugin_id, g_state->plugin_id_str.c_str()) != 0) return nullptr;

    auto* plug = new Plugin{};
    plug->host     = host;
    plug->meta     = &g_state->tone_meta;
    plug->channels = 2;

    plug->plugin.desc             = &g_state->descriptor;
    plug->plugin.plugin_data      = plug;
    plug->plugin.init             = plugin_init;
    plug->plugin.destroy          = plugin_destroy;
    plug->plugin.activate         = plugin_activate;
    plug->plugin.deactivate       = plugin_deactivate;
    plug->plugin.start_processing = plugin_start_processing;
    plug->plugin.stop_processing  = plugin_stop_processing;
    plug->plugin.reset            = plugin_reset;
    plug->plugin.process          = plugin_process;
    plug->plugin.get_extension    = plugin_get_extension;
    plug->plugin.on_main_thread   = plugin_on_main_thread;
    return &plug->plugin;
}

static uint32_t factory_get_plugin_count(const clap_plugin_factory_t*) { return g_state ? 1 : 0; }
static const clap_plugin_descriptor_t* factory_get_plugin_descriptor(const clap_plugin_factory_t*,
                                                                     uint32_t index) {
    if (!g_state || index != 0) return nullptr;
    return &g_state->descriptor;
}

static const clap_plugin_factory_t s_factory = {
    factory_get_plugin_count, factory_get_plugin_descriptor, factory_create_plugin,
};

static bool entry_init(const char* /*plugin_path*/) {
    if (g_state) return true;
    try {
        auto st = std::make_unique<ModuleState>();
        st->bundle_dir   = find_bundle_contents_();
        if (st->bundle_dir.empty()) return false;
        st->resources_dir = st->bundle_dir + "/Resources";

        st->tone_meta = load_composite_meta(st->resources_dir + "/tone_meta.json");
        st->autoeq_meta = load_meta(st->resources_dir + "/" +
                                    st->tone_meta.sub_bundles.at("auto_eq")
                                    + "/plugin_meta.json");
        st->sat_meta    = load_meta(st->resources_dir + "/" +
                                    st->tone_meta.sub_bundles.at("saturator")
                                    + "/plugin_meta.json");
        st->la2a_meta   = load_meta(st->resources_dir + "/" +
                                    st->tone_meta.sub_bundles.at("la2a")
                                    + "/plugin_meta.json");

        // Pull the DSP block payloads we need at chain construction time.
        if (st->sat_meta.dsp_blocks.empty()) {
            throw std::runtime_error("saturator sub-bundle has no dsp_blocks");
        }
        st->sat_rational = std::get<RationalAParams>(st->sat_meta.dsp_blocks[0].params);
        if (st->autoeq_meta.dsp_blocks.empty()) {
            throw std::runtime_error("auto_eq sub-bundle has no dsp_blocks");
        }
        st->autoeq_eq = std::get<ParametricEq5BandParams>(st->autoeq_meta.dsp_blocks[0].params);

        st->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nablafx-tone");
        populate_descriptor_(*st);
        g_state = st.release();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static void entry_deinit() { delete g_state; g_state = nullptr; }

static const void* entry_get_factory(const char* factory_id) {
    if (std::strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) return &s_factory;
    return nullptr;
}

}  // namespace nablafx_tone

extern "C" {
CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT,
    nablafx_tone::entry_init,
    nablafx_tone::entry_deinit,
    nablafx_tone::entry_get_factory,
};
}
