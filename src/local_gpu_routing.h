// SPDX-License-Identifier: Apache-2.0
// Tensor ownership rules for the opt-in MOSS Local GPU decoder.

#pragma once

#include <string_view>

namespace openmoss {

inline bool local_gpu_weight_name(std::string_view name) {
    return name.substr(0, 11) == "moss.local."
        || name == "moss.local_text_head.weight";
}

inline bool local_gpu_tensor_name(std::string_view name) {
    return local_gpu_weight_name(name)
        || name.substr(0, 17) == "moss.audio_embed.";
}

} // namespace openmoss
