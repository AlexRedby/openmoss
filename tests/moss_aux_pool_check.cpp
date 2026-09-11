// SPDX-License-Identifier: Apache-2.0
// VNTTS modification: exercise the real Aux owner without loading model weights.
#include "aux_internal.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Graph {
    openmoss::Model::Aux aux;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;

    explicit Graph(int n_threads, bool initialize_pool) {
        aux.backend = ggml_backend_cpu_init();
        require(aux.backend != nullptr, "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(aux.backend, n_threads);
        if (initialize_pool) require(aux.init_cpu_pool(n_threads), "aux CPU configuration failed");
        ggml_init_params params{};
        params.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
        params.no_alloc = true;
        aux.ctx = ggml_init(params);
        require(aux.ctx != nullptr, "context allocation failed");
        graph = ggml_new_graph(aux.ctx);
        auto * a = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, 64, 64);
        auto * b = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, 64, 32);
        ggml_set_input(a);
        ggml_set_input(b);
        output = ggml_soft_max(aux.ctx, ggml_mul_mat(aux.ctx, a, b));
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        aux.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux.backend));
        require(aux.galloc && ggml_gallocr_alloc_graph(aux.galloc, graph), "graph allocation failed");
        for (auto * input : {a, b}) {
            std::vector<float> data(size_t(ggml_nelements(input)));
            for (size_t i = 0; i < data.size(); ++i) data[i] = float(int(i % 13) - 6) * 0.125f;
            ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
        }
    }

    std::vector<float> run() {
        require(ggml_backend_graph_compute(aux.backend, graph) == GGML_STATUS_SUCCESS, "graph failed");
        std::vector<float> data(size_t(ggml_nelements(output)));
        ggml_backend_tensor_get(output, data.data(), 0, data.size() * sizeof(float));
        return data;
    }
};

int main() {
    try {
        double baseline_s = 0, candidate_s = 0;
        // Correctness smoke, not a benchmark: exercise each order twice per owner.
        // Repeated identical-input runs are costly on constrained CI runners.
        constexpr int cycles = 4, runs = 4;
        constexpr int baseline_threads = 4;
        const int worker_counts[] = {1, 4, 8};
        for (const int candidate_threads : worker_counts) {
        for (int cycle = 0; cycle < cycles; ++cycle) {
            std::fprintf(stderr, "aux CPU check: cycle %d/%d, workers %d vs %d\n",
                         cycle + 1, cycles, baseline_threads, candidate_threads);
            std::fflush(stderr);
            Graph baseline(baseline_threads, false), candidate(candidate_threads, true);
            const auto expected = baseline.run();
            std::fprintf(stderr, "aux CPU check: baseline graph complete\n");
            std::fflush(stderr);
            // Initialization is idempotent; repeated use must retain one owner.
            const auto pool = candidate.aux.cpu_pool;
            require(candidate.aux.init_cpu_pool(candidate_threads), "aux CPU reconfiguration failed");
            require(pool == candidate.aux.cpu_pool, "pool replaced during reinitialization");
#ifdef OPENMOSS_PERSISTENT_AUX_CPU_POOL
            require(pool != nullptr, "expected persistent workers");
#else
            require(pool == nullptr, "persistent pool unexpectedly attached");
#endif
            for (int i = 0; i < runs; ++i) {
                std::fprintf(stderr, "aux CPU check: alternating run %d/%d\n", i + 1, runs);
                std::fflush(stderr);
                // Alternate order; these are synthetic timings, not a speech gate.
                for (bool use_candidate : {bool(i % 2), !bool(i % 2)}) {
                    const auto start = std::chrono::steady_clock::now();
                    const auto actual = (use_candidate ? candidate : baseline).run();
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    (use_candidate ? candidate_s : baseline_s) += elapsed;
                    require(actual.size() == expected.size() &&
                            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                            "CPU graph output changed");
                }
            }
            std::fprintf(stderr, "aux CPU check: abort and recovery\n");
            std::fflush(stderr);
            ggml_backend_cpu_set_abort_callback(candidate.aux.backend, [](void *) { return true; }, nullptr);
            require(ggml_backend_graph_compute(candidate.aux.backend, candidate.graph) == GGML_STATUS_ABORTED,
                    "abort callback did not stop graph");
            ggml_backend_cpu_set_abort_callback(candidate.aux.backend, nullptr, nullptr);
            require(candidate.run() == expected, "graph failed to recover after abort");
            // Aux destructors join/free pools after completed or aborted graphs.
        }
        }
        std::printf("{\"schema\":\"vntts.native-aux-pool-check\",\"version\":\"%s\","
                    "\"baseline_threads\":4,\"worker_counts\":[1,4,8],\"cycles\":%d,\"runs_per_cycle\":%d,"
                    "\"output_identical\":true,\"abort_recovery\":true,"
                    "\"baseline_s\":%.6f,\"candidate_s\":%.6f,"
                    "\"scope\":\"synthetic CPU graphs, not speech performance\"}\n",
                    OPENMOSS_VERSION, cycles, runs,
                    baseline_s, candidate_s);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "aux pool check failed: %s\n", error.what());
        return 1;
    }
}
