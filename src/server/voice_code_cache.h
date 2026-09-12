// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "openmoss/pipeline.h"

namespace openmoss::server {

inline uint64_t fnv1a64(const uint8_t * data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::string checksum_hex(uint64_t hash) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

inline std::string voice_wav_checksum(const std::vector<uint8_t> & bytes) {
    return checksum_hex(fnv1a64(bytes.data(), bytes.size()));
}

inline std::string voice_codes_checksum(const std::vector<int32_t> & codes) {
    uint64_t hash = 14695981039346656037ULL;
    for (int32_t code : codes) {
        const uint32_t value = static_cast<uint32_t>(code);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= uint8_t(value >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return checksum_hex(hash);
}

inline bool load_voice_codes(const std::filesystem::path & path,
                             const std::string & model_key,
                             const std::string & wav_checksum,
                             int32_t n_vq,
                             int32_t audio_vocab_size,
                             EncodedReference & out) {
    try {
        std::ifstream source(path, std::ios::binary | std::ios::ate);
        if (!source) return false;
        const auto size = source.tellg();
        if (size <= 0 || size > 8 * 1024 * 1024) return false;
        source.seekg(0);
        const auto document = nlohmann::json::parse(source);
        if (document.value("schema", "") != "openmoss.voice-codes" ||
            document.value("version", 0) != 1 ||
            document.value("model_key", "") != model_key ||
            document.value("wav_checksum", "") != wav_checksum ||
            document.value("n_vq", 0) != n_vq ||
            document.value("audio_vocab_size", 0) != audio_vocab_size ||
            !document.contains("codes_checksum")) {
            return false;
        }
        const int32_t n_frames = document.at("n_frames").get<int32_t>();
        auto codes = document.at("codes").get<std::vector<int32_t>>();
        if (n_frames <= 0 || n_vq <= 0 || audio_vocab_size <= 0 ||
            codes.size() != size_t(n_frames) * size_t(n_vq)) {
            return false;
        }
        for (int32_t code : codes) {
            if (code < 0 || code >= audio_vocab_size) return false;
        }
        if (document.at("codes_checksum").get<std::string>() !=
            voice_codes_checksum(codes)) return false;
        out.codes = std::move(codes);
        out.n_frames = n_frames;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

inline bool save_voice_codes(const std::filesystem::path & path,
                             const std::string & model_key,
                             const std::string & wav_checksum,
                             int32_t n_vq,
                             int32_t audio_vocab_size,
                             const EncodedReference & reference) {
    if (reference.n_frames <= 0 || n_vq <= 0 || audio_vocab_size <= 0 ||
        reference.codes.size() != size_t(reference.n_frames) * size_t(n_vq)) {
        return false;
    }
    for (int32_t code : reference.codes) {
        if (code < 0 || code >= audio_vocab_size) return false;
    }
    const nlohmann::json document = {
        {"schema", "openmoss.voice-codes"},
        {"version", 1},
        {"model_key", model_key},
        {"wav_checksum", wav_checksum},
        {"n_vq", n_vq},
        {"audio_vocab_size", audio_vocab_size},
        {"n_frames", reference.n_frames},
        {"codes", reference.codes},
        {"codes_checksum", voice_codes_checksum(reference.codes)},
    };
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream target(temporary, std::ios::trunc);
        if (!target || !(target << document.dump())) return false;
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (!error) return true;
    std::filesystem::remove(temporary, error);
    return false;
}

} // namespace openmoss::server
