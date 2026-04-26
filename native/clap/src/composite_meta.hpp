// tone_meta.json — top-level meta for the composite TONE plugin.
//
// Written on the Python side by ``nablafx.export.composite``; read here on
// module load to wire the host-exposed AMT/TRM knobs to per-stage
// parameters and to locate the three sub-bundles.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "meta.hpp"

namespace nablafx {

struct CompositeAmountSat {
    float pre_gain_db_max  = 12.0f;
    float post_gain_db_max = -12.0f;
    float wet_mix_max      = 1.0f;
};

struct CompositeAmountLa2a {
    float peak_reduction_min = 20.0f;
    float peak_reduction_max = 70.0f;
    float comp_or_limit      = 1.0f;
};

struct CompositeAmountAutoEq {
    float wet_mix_max = 1.0f;
};

struct CompositeLevelerCfg {
    float target_lufs = -14.0f;
};

struct CompositeCeilingCfg {
    float ceiling_dbtp = -1.0f;
    float lookahead_ms = 1.5f;
    float attack_ms    = 0.5f;
    float release_ms   = 50.0f;
};

struct CompositeMeta {
    int                                            schema_version{};
    std::string                                    effect_name;
    std::string                                    model_id;
    int                                            sample_rate{};
    int                                            channels{};
    // Role name → directory name (relative to .clap/Contents/Resources/).
    std::unordered_map<std::string, std::string>   sub_bundles;
    // Host-exposed knobs, keyed by id ("AMT", "TRM"). Stored as the same
    // ControlSpec used by the per-stage plugins so existing param-id helpers
    // apply unchanged.
    std::vector<ControlSpec>                       controls;

    CompositeAmountSat                             amt_sat;
    CompositeAmountLa2a                            amt_la2a;
    CompositeAmountAutoEq                          amt_autoeq;
    CompositeLevelerCfg                            leveler;
    CompositeCeilingCfg                            ceiling;
};

// Throws std::runtime_error on malformed JSON or unknown schema_version.
CompositeMeta load_composite_meta(const std::string& path);

}  // namespace nablafx
