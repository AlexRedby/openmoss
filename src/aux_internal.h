// SPDX-License-Identifier: Apache-2.0
//
// Internal definition of Model::Aux. Shared between model.cpp (which owns it)
// and codec.cpp (which reaches in for the backend handle and tensor map).

#pragma once

#include "openmoss/model.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace openmoss {

struct Model::Aux {
    // VNTTS modification: sole ownership of the optional auxiliary CPU pool.
    Aux() = default;
    Aux(const Aux &) = delete;
    Aux & operator=(const Aux &) = delete;

    ggml_backend_t        backend = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;

    // Tensors owned by us:
    //   - moss.audio_embed.{i}.weight     i in [0, n_vq)
    //   - moss.audio_head.{i}.weight      i in [0, n_vq)
    //   - moss.codec.*                    (lazy/optional)
    //   - "_text_embed"                   the Qwen3 input embedding table
    std::unordered_map<std::string, ggml_tensor *> tensors;
    ggml_tensor * text_embed = nullptr;

    bool codec_present = false;
    int32_t hidden_size      = 0;
    int32_t n_vq             = 0;
    int32_t audio_vocab_full = 0;
    int32_t text_vocab_size  = 0;

    ggml_gallocr_t galloc = nullptr;

    // This experiment is deliberately separate from libllama's backbone pool.
    // It is only installed on this direct aux CPU backend, before requests can
    // use it. ggml_backend_free() does not own an attached threadpool.
    ggml_threadpool_t cpu_pool = nullptr;

    // Returns true only when the opt-in build attached the fixed four-thread
    // pool. A false return leaves the backend on GGML's disposable-pool path.
    bool init_cpu_pool() {
#if defined(OPENMOSS_PERSISTENT_AUX_CPU_POOL) && OPENMOSS_PERSISTENT_AUX_CPU_POOL
        if (!backend || !ggml_backend_is_cpu(backend)) return false;
        if (cpu_pool) return true;

        constexpr int n_threads = GGML_DEFAULT_N_THREADS;
        struct ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
        params.poll = 0; // persistent idle workers sleep instead of busy-polling
        ggml_threadpool_t pool = ggml_threadpool_new(&params);
        if (!pool) return false;

        ggml_backend_cpu_set_n_threads(backend, n_threads);
        ggml_backend_cpu_set_threadpool(backend, pool);
        cpu_pool = pool;
        return true;
#else
        return false;
#endif
    }

    ~Aux() {
        if (galloc)  ggml_gallocr_free(galloc);
        if (buffer)  ggml_backend_buffer_free(buffer);
        if (ctx)     ggml_free(ctx);
#if defined(OPENMOSS_PERSISTENT_AUX_CPU_POOL) && OPENMOSS_PERSISTENT_AUX_CPU_POOL
        if (cpu_pool) {
            // The backend merely borrows this pool. Detach it before freeing
            // the owned workers; destruction happens after requests quiesce.
            if (backend && ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_threadpool(backend, nullptr);
            }
            ggml_threadpool_free(cpu_pool);
            cpu_pool = nullptr;
        }
#endif
        if (backend) ggml_backend_free(backend);
    }
};

} // namespace openmoss
