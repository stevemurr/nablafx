#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nablafx {

struct ControlSpec {
    std::string id;
    std::string name;
    float       min;
    float       max;
    float       def;
    float       skew;
    std::string unit;
};

struct StateSpec {
    std::string          name;   // ONNX name stem (e.g. "processor_lstm_h")
    std::vector<int64_t> shape;  // (num_layers, 1, hidden_size)
    std::string          dtype;  // always "float32" in v1
};

struct PluginMeta {
    int                       schema_version{};
    std::string               effect_name;
    std::string               model_id;
    std::string               architecture;   // "tcn" | "lstm" | "gcn"
    int                       sample_rate{};
    int                       channels{};
    bool                      causal{};
    int                       receptive_field{};
    int                       latency_samples{};
    int                       num_controls{};
    std::vector<ControlSpec>  controls;
    std::vector<StateSpec>    state_tensors;
    std::vector<std::string>  input_names;
    std::vector<std::string>  output_names;
};

// Parse a plugin_meta.json file from disk. Throws std::runtime_error on any
// problem (missing file, malformed JSON, unknown schema_version).
PluginMeta load_meta(const std::string& path);

}  // namespace nablafx
