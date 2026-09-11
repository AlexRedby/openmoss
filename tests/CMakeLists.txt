# Smoke tests — meant to validate plumbing, not correctness of generation.
# Build with: cmake --build build --target moss-tts-info

# VNTTS modification: no weights, audio device or GPU needed.
add_executable(moss-aux-pool-check moss_aux_pool_check.cpp)
target_include_directories(moss-aux-pool-check PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(moss-aux-pool-check PRIVATE openmoss_core ggml-cpu)
add_test(NAME moss-aux-pool-check COMMAND moss-aux-pool-check)
set_tests_properties(moss-aux-pool-check PROPERTIES TIMEOUT 30)

# VNTTS modification: verifies the diagnostic build selection without a model.
add_executable(moss-local-gpu-check moss_local_gpu_check.cpp)
target_include_directories(moss-local-gpu-check PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(moss-local-gpu-check PRIVATE openmoss_core)
add_test(NAME moss-local-gpu-check COMMAND moss-local-gpu-check)
set_tests_properties(moss-local-gpu-check PROPERTIES TIMEOUT 30)

add_executable(moss-tts-info ${CMAKE_CURRENT_SOURCE_DIR}/moss_tts_info.cpp)
target_link_libraries(moss-tts-info PRIVATE openmoss_core)

add_executable(moss-tts-compute-test ${CMAKE_CURRENT_SOURCE_DIR}/moss_tts_compute.cpp)
target_link_libraries(moss-tts-compute-test PRIVATE openmoss_core)

add_executable(moss-codec-roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/moss_codec_roundtrip.cpp)
target_link_libraries(moss-codec-roundtrip PRIVATE openmoss_core)

add_executable(moss-local-probe ${CMAKE_CURRENT_SOURCE_DIR}/moss_local_probe.cpp)
target_link_libraries(moss-local-probe PRIVATE openmoss_core)

add_executable(moss-sfx-probe ${CMAKE_CURRENT_SOURCE_DIR}/moss_sfx_probe.cpp)
target_link_libraries(moss-sfx-probe PRIVATE openmoss_core)

add_executable(moss-sfx-vae-probe ${CMAKE_CURRENT_SOURCE_DIR}/moss_sfx_vae_probe.cpp)
target_link_libraries(moss-sfx-vae-probe PRIVATE openmoss_core)

add_executable(moss-prompt-probe ${CMAKE_CURRENT_SOURCE_DIR}/moss_prompt_probe.cpp)
target_link_libraries(moss-prompt-probe PRIVATE openmoss_core)

add_executable(moss-codec-causality ${CMAKE_CURRENT_SOURCE_DIR}/moss_codec_causality.cpp)
target_link_libraries(moss-codec-causality PRIVATE openmoss_core)

add_executable(moss-codec-stream ${CMAKE_CURRENT_SOURCE_DIR}/moss_codec_stream.cpp)
target_link_libraries(moss-codec-stream PRIVATE openmoss_core)
