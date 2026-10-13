#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <algorithm>

class BPEDetokenizer {
public:
    std::vector<uint32_t> offsets;
    std::vector<char> str_arena;
    uint32_t num_tokens = 0;

    std::unordered_map<std::string, int> token_to_id;
    size_t max_token_len = 0;

    bool load(const std::string& vocab_bin_path) {
        std::ifstream f(vocab_bin_path, std::ios::binary);
        if (!f) {
            std::cerr << "BPEDetokenizer: failed to open " << vocab_bin_path << "\n";
            return false;
        }

        f.read(reinterpret_cast<char*>(&num_tokens), sizeof(uint32_t));
        if (num_tokens == 0 || num_tokens > 500000) {
            std::cerr << "BPEDetokenizer: invalid num_tokens = " << num_tokens << "\n";
            return false;
        }

        offsets.resize(num_tokens);
        f.read(reinterpret_cast<char*>(offsets.data()), num_tokens * sizeof(uint32_t));

        // Remaining bytes are the string arena
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t arena_size = file_size - (sizeof(uint32_t) + num_tokens * sizeof(uint32_t));

        f.seekg(sizeof(uint32_t) + num_tokens * sizeof(uint32_t), std::ios::beg);
        str_arena.resize(arena_size);
        f.read(str_arena.data(), arena_size);

        token_to_id.reserve(num_tokens);
        for (uint32_t i = 0; i < num_tokens; ++i) {
            uint32_t off = offsets[i];
            if (off < str_arena.size()) {
                const char* raw = str_arena.data() + off;
                std::string s(raw);
                token_to_id[s] = static_cast<int>(i);
                if (s.size() > max_token_len) max_token_len = s.size();
            }
        }

        return true;
    }

    std::vector<int> encode(const std::string& text) const {
        std::string bpe_input;
        bpe_input.reserve(text.size() * 2);
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == ' ') {
                bpe_input += "\xC4\xA0"; // 'Ġ'
            } else if (c == '\n') {
                bpe_input += "\xC4\x8A"; // 'Ċ'
            } else if (c == '\t') {
                bpe_input += "\xC4\x89"; // 'ĉ'
            } else if (c == '\r') {
                bpe_input += "\xC4\x8D"; // 'č'
            } else {
                bpe_input += text[i];
            }
        }

        std::vector<int> tokens;
        size_t pos = 0;
        size_t limit = max_token_len > 0 ? max_token_len : 64;

        while (pos < bpe_input.size()) {
            bool matched = false;
            size_t max_l = std::min(limit, bpe_input.size() - pos);
            for (size_t len = max_l; len >= 1; --len) {
                std::string sub = bpe_input.substr(pos, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                pos += 1;
            }
        }
        return tokens;
    }

    std::string decode(int token_id) const {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= num_tokens) {
            return "[" + std::to_string(token_id) + "]";
        }

        uint32_t off = offsets[token_id];
        if (off >= str_arena.size()) return "";

        const char* raw = str_arena.data() + off;
        std::string s(raw);

        // BPE conversion:
        // Replace 'Ġ' (0xC4 0xA0) with ' '
        // Replace 'Ċ' (0xC4 0x8A) with '\n'
        std::string res;
        res.reserve(s.size());

        for (size_t i = 0; i < s.size(); ) {
            unsigned char c1 = static_cast<unsigned char>(s[i]);
            if (c1 == 0xC4 && i + 1 < s.size()) {
                unsigned char c2 = static_cast<unsigned char>(s[i + 1]);
                if (c2 == 0xA0) { // 'Ġ' -> ' '
                    res += ' ';
                    i += 2;
                    continue;
                } else if (c2 == 0x8A) { // 'Ċ' -> '\n'
                    res += '\n';
                    i += 2;
                    continue;
                }
            }
            res += s[i];
            i += 1;
        }

        return res;
    }
};
