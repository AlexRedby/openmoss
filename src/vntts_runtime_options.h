// SPDX-License-Identifier: Apache-2.0
// Small shared validation for VNTTS's process-start runtime controls.

#pragma once

namespace openmoss {

constexpr int VNTTS_AUX_CPU_THREADS_DEFAULT = 4;
constexpr int VNTTS_AUX_CPU_THREADS_MIN = 1;
constexpr int VNTTS_AUX_CPU_THREADS_MAX = 16;

inline bool vntts_valid_aux_cpu_threads(int value) {
    return value >= VNTTS_AUX_CPU_THREADS_MIN && value <= VNTTS_AUX_CPU_THREADS_MAX;
}

inline bool vntts_valid_local_gpu_request(bool local_gpu, bool aux_cpu) {
    return !local_gpu || aux_cpu;
}

} // namespace openmoss
