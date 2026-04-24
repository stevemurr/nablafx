#include "meta.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nablafx {

using nlohmann::json;

PluginMeta load_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }
    json j;
    f >> j;

    PluginMeta m;
    m.schema_version = j.value("schema_version", 0);
    if (m.schema_version != 1) {
        std::ostringstream oss;
        oss << "unsupported plugin_meta schema_version " << m.schema_version
            << " (this build understands version 1)";
        throw std::runtime_error(oss.str());
    }

    m.effect_name     = j.at("effect_name").get<std::string>();
    m.model_id        = j.at("model_id").get<std::string>();
    m.architecture    = j.at("architecture").get<std::string>();
    m.sample_rate     = j.at("sample_rate").get<int>();
    m.channels        = j.at("channels").get<int>();
    m.causal          = j.at("causal").get<bool>();
    m.receptive_field = j.at("receptive_field").get<int>();
    m.latency_samples = j.at("latency_samples").get<int>();
    m.num_controls    = j.at("num_controls").get<int>();

    for (const auto& c : j.at("controls")) {
        m.controls.push_back(ControlSpec{
            c.at("id").get<std::string>(),
            c.at("name").get<std::string>(),
            c.at("min").get<float>(),
            c.at("max").get<float>(),
            c.at("default").get<float>(),
            c.value("skew", 1.0f),
            c.value("unit", std::string{}),
        });
    }

    for (const auto& s : j.at("state_tensors")) {
        StateSpec spec;
        spec.name  = s.at("name").get<std::string>();
        spec.shape = s.at("shape").get<std::vector<int64_t>>();
        spec.dtype = s.at("dtype").get<std::string>();
        if (spec.dtype != "float32") {
            throw std::runtime_error("state tensor " + spec.name + ": only float32 is supported");
        }
        m.state_tensors.push_back(std::move(spec));
    }

    m.input_names  = j.at("input_names").get<std::vector<std::string>>();
    m.output_names = j.at("output_names").get<std::vector<std::string>>();
    return m;
}

}  // namespace nablafx
