// SPDX-License-Identifier: Apache-2.0
// No-model check: the build option must be explicit and queryable.

#include "local_gpu_routing.h"
#include "vntts_runtime_options.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace {

struct Tensor {
    const char * name;
    std::array<unsigned char, 3> bytes;
};

bool route_tensor_groups() {
    const Tensor tensors[] = {
        {"moss.local.h.0.attn.c_attn.weight", {1, 2, 3}},
        {"moss.local_text_head.weight", {4, 5, 6}},
        {"moss.audio_embed.0.weight", {7, 8, 9}},
        {"moss.codec.dec.0.weight", {10, 11, 12}},
    };
    std::array<std::array<unsigned char, 3>, 4> cpu{};
    std::array<std::array<unsigned char, 3>, 4> gpu{};
    size_t n_cpu = 0, n_gpu = 0;
    for (const Tensor & tensor : tensors) {
        if (!openmoss::local_gpu_weight_name(tensor.name)) cpu[n_cpu++] = tensor.bytes;
        if (openmoss::local_gpu_tensor_name(tensor.name)) gpu[n_gpu++] = tensor.bytes;
    }
    return n_cpu == 2 && n_gpu == 3
        && std::memcmp(cpu[0].data(), tensors[2].bytes.data(), tensors[2].bytes.size()) == 0
        && std::memcmp(cpu[1].data(), tensors[3].bytes.data(), tensors[3].bytes.size()) == 0
        && std::memcmp(gpu[0].data(), tensors[0].bytes.data(), tensors[0].bytes.size()) == 0
        && std::memcmp(gpu[1].data(), tensors[1].bytes.data(), tensors[1].bytes.size()) == 0
        && std::memcmp(gpu[2].data(), tensors[2].bytes.data(), tensors[2].bytes.size()) == 0;
}

bool runtime_options_valid() {
    return openmoss::vntts_valid_aux_cpu_threads(1)
        && openmoss::vntts_valid_aux_cpu_threads(4)
        && openmoss::vntts_valid_aux_cpu_threads(8)
        && !openmoss::vntts_valid_aux_cpu_threads(0)
        && !openmoss::vntts_valid_aux_cpu_threads(17)
        && openmoss::vntts_valid_local_gpu_request(false, false)
        && openmoss::vntts_valid_local_gpu_request(true, true)
        && !openmoss::vntts_valid_local_gpu_request(true, false);
}

} // namespace

int main() {
    if (!route_tensor_groups()) return 1;
    if (!runtime_options_valid()) return 1;
    std::printf("{\"schema\":\"vntts.local-gpu-check\",\"runtime_controls\":true,\"routing\":\"ok\"}\n");
    return 0;
}
