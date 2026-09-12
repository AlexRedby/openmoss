// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/voice_code_cache.h"

int main() {
    namespace fs = std::filesystem;
    using openmoss::EncodedReference;
    using openmoss::server::load_voice_codes;
    using openmoss::server::save_voice_codes;
    using openmoss::server::voice_wav_checksum;

    const auto path = fs::temp_directory_path() / "openmoss-voice-code-cache-test.json";
    const std::vector<uint8_t> wav_a = {1, 2, 3, 4};
    const std::vector<uint8_t> wav_b = {1, 2, 3, 5};
    const EncodedReference expected{{0, 1, 2, 3, 4, 5}, 2};
    fs::remove(path);
    if (!save_voice_codes(path, "model-a", voice_wav_checksum(wav_a), 3, 16, expected)) return 1;

    EncodedReference loaded;
    if (!load_voice_codes(path, "model-a", voice_wav_checksum(wav_a), 3, 16, loaded) ||
        loaded.n_frames != expected.n_frames || loaded.codes != expected.codes) return 2;
    if (load_voice_codes(path, "model-a", voice_wav_checksum(wav_b), 3, 16, loaded)) return 3;
    if (load_voice_codes(path, "model-b", voice_wav_checksum(wav_a), 3, 16, loaded)) return 4;

    nlohmann::json changed;
    {
        std::ifstream source(path);
        changed = nlohmann::json::parse(source);
    }
    changed["codes"][0] = 6;
    std::ofstream(path, std::ios::trunc) << changed.dump();
    if (load_voice_codes(path, "model-a", voice_wav_checksum(wav_a), 3, 16, loaded)) return 5;

    const EncodedReference invalid{{0, 1, 2, 3, 4, 16}, 2};
    if (save_voice_codes(path, "model-a", voice_wav_checksum(wav_a), 3, 16, invalid)) return 6;
    std::ofstream(path, std::ios::trunc) << "{broken";
    if (load_voice_codes(path, "model-a", voice_wav_checksum(wav_a), 3, 16, loaded)) return 7;
    fs::remove(path);
    return 0;
}
