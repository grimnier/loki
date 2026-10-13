#pragma once

// Host record of MoE routing for one conversation.
// DeepSeek-R1 has 61 transformer layers. Layers 3..60 are MoE: 58 layers,
// 256 routed experts, top-8. Slot count matches the compact KV ring.
// Expert id 7 in MoE-layer 0 is a different tensor from expert id 7 in MoE-layer 1.
// Scores are stored as the router emitted them. This header does not renormalize.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wag {

inline constexpr int kMaxTokens = 4096;
inline constexpr int kMoeLayers = 58;
inline constexpr int kTopExperts = 8;
inline constexpr int kExpertCount = 256;
inline constexpr int kFirstMoeLayer = 3;
inline constexpr uint16_t kNoExpert = 0xFFFFu;
inline constexpr int kRouteWidth = kMoeLayers * kTopExperts;

struct WeightRoutingTrace {
    uint16_t active_experts[kMoeLayers][kTopExperts];
    float routing_weights[kMoeLayers][kTopExperts];
    uint16_t mode = 0;
    uint64_t layers_present = 0;

    WeightRoutingTrace() {
        for (int layer = 0; layer < kMoeLayers; ++layer) {
            for (int k = 0; k < kTopExperts; ++k) {
                active_experts[layer][k] = kNoExpert;
                routing_weights[layer][k] = 0.f;
            }
        }
    }

    bool has_layer(uint32_t moe_layer) const {
        return moe_layer < static_cast<uint32_t>(kMoeLayers) &&
               ((layers_present >> moe_layer) & 1ull) != 0;
    }
};

struct TrajectoryMatch {
    uint32_t context_begin = 0;
    uint32_t context_end = 0;
    float cosine = 0.f;
    float jaccard = 0.f;
};

// CopyFn is void(void* dst, uint64_t src_device, size_t bytes).
// Pass the device pointer as an integer so this header does not include CUDA.
class WeightTraceCache {
public:
    WeightTraceCache() : steps_(static_cast<std::size_t>(kMaxTokens)) {}

    void record(uint32_t step, std::span<const uint16_t> expert_ids,
                std::span<const float> weights, uint16_t mode = 0) {
        check_step(step);
        if (expert_ids.size() != static_cast<std::size_t>(kRouteWidth) ||
            weights.size() != static_cast<std::size_t>(kRouteWidth)) {
            throw std::invalid_argument("router output must be 58*8 ids and 58*8 scores, layer-major");
        }
        for (int i = 0; i < kRouteWidth; ++i) {
            if (expert_ids[static_cast<std::size_t>(i)] >= kExpertCount) {
                throw std::invalid_argument("expert id out of range");
            }
            if (!std::isfinite(weights[static_cast<std::size_t>(i)])) {
                throw std::invalid_argument("routing score must be finite");
            }
        }
        WeightRoutingTrace& slot = steps_[step];
        const bool was_empty = slot.layers_present == 0;
        for (int layer = 0; layer < kMoeLayers; ++layer) {
            write_layer(slot, static_cast<uint32_t>(layer),
                        expert_ids.subspan(static_cast<std::size_t>(layer * kTopExperts), kTopExperts),
                        weights.subspan(static_cast<std::size_t>(layer * kTopExperts), kTopExperts));
        }
        slot.mode = mode;
        slot.layers_present = (1ull << kMoeLayers) - 1ull;
        note(step, was_empty);
    }

    void record_u32(uint32_t step, std::span<const uint32_t> expert_ids,
                    std::span<const float> weights, uint16_t mode = 0) {
        if (expert_ids.size() != static_cast<std::size_t>(kRouteWidth)) {
            throw std::invalid_argument("router output must be 58*8 ids and 58*8 scores, layer-major");
        }
        uint16_t narrow[kRouteWidth];
        for (int i = 0; i < kRouteWidth; ++i) narrow[i] = narrow_id(expert_ids[static_cast<std::size_t>(i)]);
        record(step, std::span<const uint16_t>(narrow, kRouteWidth), weights, mode);
    }

    void record_layer(uint32_t step, uint32_t moe_layer,
                      std::span<const uint16_t> expert_ids,
                      std::span<const float> weights, uint16_t mode = 0) {
        check_step(step);
        check_layer(moe_layer);
        WeightRoutingTrace& slot = steps_[step];
        const bool was_empty = slot.layers_present == 0;
        write_layer(slot, moe_layer, expert_ids, weights);
        slot.mode = mode;
        slot.layers_present |= 1ull << moe_layer;
        note(step, was_empty);
    }

    void record_layer_u32(uint32_t step, uint32_t moe_layer,
                          std::span<const uint32_t> expert_ids,
                          std::span<const float> weights, uint16_t mode = 0) {
        if (expert_ids.size() != static_cast<std::size_t>(kTopExperts)) {
            throw std::invalid_argument("layer router output must be 8 ids and 8 scores");
        }
        uint16_t narrow[kTopExperts];
        for (int k = 0; k < kTopExperts; ++k) narrow[k] = narrow_id(expert_ids[static_cast<std::size_t>(k)]);
        record_layer(step, moe_layer, std::span<const uint16_t>(narrow, kTopExperts), weights, mode);
    }

    template <class CopyFn>
    void record_from_device(uint32_t step, std::uint64_t d_expert_ids, std::uint64_t d_weights,
                            CopyFn&& copy, uint16_t mode = 0) {
        uint16_t ids[kRouteWidth];
        float scores[kRouteWidth];
        copy(ids, d_expert_ids, sizeof ids);
        copy(scores, d_weights, sizeof scores);
        record(step, std::span<const uint16_t>(ids, kRouteWidth),
               std::span<const float>(scores, kRouteWidth), mode);
    }

    template <class CopyFn>
    void record_layer_from_device(uint32_t step, uint32_t moe_layer,
                                  std::uint64_t d_expert_ids, std::uint64_t d_weights,
                                  CopyFn&& copy, uint16_t mode = 0) {
        uint16_t ids[kTopExperts];
        float scores[kTopExperts];
        copy(ids, d_expert_ids, sizeof ids);
        copy(scores, d_weights, sizeof scores);
        record_layer(step, moe_layer, std::span<const uint16_t>(ids, kTopExperts),
                     std::span<const float>(scores, kTopExperts), mode);
    }

    // Existing host router writes uint32_t expert ids (route_top8).
    template <class CopyFn>
    void record_layer_from_device_u32(uint32_t step, uint32_t moe_layer,
                                      std::uint64_t d_expert_ids, std::uint64_t d_weights,
                                      CopyFn&& copy, uint16_t mode = 0) {
        uint32_t ids[kTopExperts];
        float scores[kTopExperts];
        copy(ids, d_expert_ids, sizeof ids);
        copy(scores, d_weights, sizeof scores);
        record_layer_u32(step, moe_layer, std::span<const uint32_t>(ids, kTopExperts),
                         std::span<const float>(scores, kTopExperts), mode);
    }

    // matrix is 256*256, row-major. Diagonal counts selections of that expert.
    // Off-diagonal (i, j) counts token steps in [begin, end) where both experts
    // sat in this layer's top-8. A layer that was not recorded at that step is skipped.
    void affinity(uint32_t moe_layer, uint32_t begin, uint32_t end, std::span<uint32_t> matrix) const {
        check_layer(moe_layer);
        check_range(begin, end);
        zero_matrix(matrix);
        for (uint32_t step = begin; step < end; ++step) {
            if (steps_[step].has_layer(moe_layer)) accumulate_pairs(steps_[step], moe_layer, matrix);
        }
    }

    void affinity_mode(uint32_t moe_layer, uint16_t mode, std::span<uint32_t> matrix) const {
        check_layer(moe_layer);
        zero_matrix(matrix);
        for (uint32_t step = 0; step < high_water_; ++step) {
            if (steps_[step].mode == mode && steps_[step].has_layer(moe_layer)) {
                accumulate_pairs(steps_[step], moe_layer, matrix);
            }
        }
    }

    // Compare the prompt route [prompt_begin, prompt_end) with every same-length
    // window of past context. Windows that overlap the prompt are omitted.
    std::vector<TrajectoryMatch> find_similar(uint32_t prompt_begin, uint32_t prompt_end,
                                              uint32_t context_begin, uint32_t context_end,
                                              std::size_t top_n, uint32_t stride = 1) const {
        check_range(prompt_begin, prompt_end);
        check_range(context_begin, context_end);
        if (stride == 0) throw std::invalid_argument("stride must be nonzero");
        if (top_n == 0) return {};

        const uint32_t width = prompt_end - prompt_begin;
        if (context_end - context_begin < width) return {};

        std::vector<float> query(static_cast<std::size_t>(hist_len()), 0.f);
        std::vector<uint8_t> query_seen(static_cast<std::size_t>(hist_len()), 0);
        accumulate_hist(prompt_begin, prompt_end, query, query_seen);

        std::vector<float> hist(static_cast<std::size_t>(hist_len()), 0.f);
        std::vector<uint8_t> seen(static_cast<std::size_t>(hist_len()), 0);
        std::vector<TrajectoryMatch> hits;
        for (uint32_t begin = context_begin; begin + width <= context_end; begin += stride) {
            const uint32_t end = begin + width;
            if (begin < prompt_end && prompt_begin < end) continue;
            std::fill(hist.begin(), hist.end(), 0.f);
            std::fill(seen.begin(), seen.end(), 0);
            accumulate_hist(begin, end, hist, seen);
            TrajectoryMatch hit;
            hit.context_begin = begin;
            hit.context_end = end;
            hit.cosine = cosine(query, hist);
            hit.jaccard = jaccard(query_seen, seen);
            hits.push_back(hit);
        }

        const std::size_t keep = std::min(top_n, hits.size());
        std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(keep), hits.end(),
                          [](const TrajectoryMatch& a, const TrajectoryMatch& b) {
                              if (a.cosine != b.cosine) return a.cosine > b.cosine;
                              return a.context_begin < b.context_begin;
                          });
        hits.resize(keep);
        return hits;
    }

    const WeightRoutingTrace& at(uint32_t step) const {
        check_step(step);
        return steps_[step];
    }

    uint32_t recorded_steps() const { return recorded_; }
    uint32_t high_water() const { return high_water_; }

    void clear() {
        steps_.assign(static_cast<std::size_t>(kMaxTokens), WeightRoutingTrace{});
        recorded_ = 0;
        high_water_ = 0;
    }

private:
    static constexpr int hist_len() { return kMoeLayers * kExpertCount; }

    static void check_step(uint32_t step) {
        if (step >= static_cast<uint32_t>(kMaxTokens)) {
            throw std::invalid_argument("token step is outside the 4096-slot trace");
        }
    }

    static void check_layer(uint32_t moe_layer) {
        if (moe_layer >= static_cast<uint32_t>(kMoeLayers)) {
            throw std::invalid_argument("moe layer must be in 0..57 (transformer layers 3..60)");
        }
    }

    static void check_range(uint32_t begin, uint32_t end) {
        if (begin > end || end > static_cast<uint32_t>(kMaxTokens)) {
            throw std::invalid_argument("token range is outside the trace");
        }
    }

    static uint16_t narrow_id(uint32_t id) {
        if (id >= static_cast<uint32_t>(kExpertCount)) {
            throw std::invalid_argument("expert id out of range");
        }
        return static_cast<uint16_t>(id);
    }

    static void write_layer(WeightRoutingTrace& slot, uint32_t moe_layer,
                            std::span<const uint16_t> expert_ids, std::span<const float> weights) {
        if (expert_ids.size() != static_cast<std::size_t>(kTopExperts) ||
            weights.size() != static_cast<std::size_t>(kTopExperts)) {
            throw std::invalid_argument("layer router output must be 8 ids and 8 scores");
        }
        uint16_t ids[kTopExperts];
        float scores[kTopExperts];
        for (int k = 0; k < kTopExperts; ++k) {
            const uint16_t id = expert_ids[static_cast<std::size_t>(k)];
            const float score = weights[static_cast<std::size_t>(k)];
            if (id >= kExpertCount) throw std::invalid_argument("expert id out of range");
            if (!std::isfinite(score)) throw std::invalid_argument("routing score must be finite");
            ids[k] = id;
            scores[k] = score;
        }
        for (int k = 0; k < kTopExperts; ++k) {
            slot.active_experts[moe_layer][k] = ids[k];
            slot.routing_weights[moe_layer][k] = scores[k];
        }
    }

    void note(uint32_t step, bool was_empty) {
        if (was_empty) ++recorded_;
        if (step >= high_water_) high_water_ = step + 1;
    }

    static void zero_matrix(std::span<uint32_t> matrix) {
        if (matrix.size() != static_cast<std::size_t>(kExpertCount * kExpertCount)) {
            throw std::invalid_argument("affinity matrix must be 256*256");
        }
        std::fill(matrix.begin(), matrix.end(), 0u);
    }

    static void accumulate_pairs(const WeightRoutingTrace& slot, uint32_t moe_layer,
                                 std::span<uint32_t> matrix) {
        uint16_t uniq[kTopExperts];
        int n = 0;
        for (int k = 0; k < kTopExperts; ++k) {
            const uint16_t id = slot.active_experts[moe_layer][k];
            if (id == kNoExpert) continue;
            bool seen = false;
            for (int u = 0; u < n; ++u) seen = seen || uniq[u] == id;
            if (seen) continue;
            uniq[n++] = id;
        }
        for (int i = 0; i < n; ++i) {
            const uint32_t a = uniq[i];
            matrix[static_cast<std::size_t>(a) * kExpertCount + a] += 1u;
            for (int j = i + 1; j < n; ++j) {
                const uint32_t b = uniq[j];
                matrix[static_cast<std::size_t>(a) * kExpertCount + b] += 1u;
                matrix[static_cast<std::size_t>(b) * kExpertCount + a] += 1u;
            }
        }
    }

    void accumulate_hist(uint32_t begin, uint32_t end, std::vector<float>& hist,
                         std::vector<uint8_t>& seen) const {
        for (uint32_t step = begin; step < end; ++step) {
            const WeightRoutingTrace& slot = steps_[step];
            for (int layer = 0; layer < kMoeLayers; ++layer) {
                if (!slot.has_layer(static_cast<uint32_t>(layer))) continue;
                for (int k = 0; k < kTopExperts; ++k) {
                    const uint16_t id = slot.active_experts[layer][k];
                    if (id == kNoExpert) continue;
                    const std::size_t index = static_cast<std::size_t>(layer * kExpertCount + id);
                    hist[index] += slot.routing_weights[layer][k];
                    seen[index] = 1;
                }
            }
        }
    }

    static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (na <= 0.0 || nb <= 0.0) return 0.f;
        return static_cast<float>(dot / std::sqrt(na * nb));
    }

    static float jaccard(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int used_layers = 0;
        double sum = 0.0;
        for (int layer = 0; layer < kMoeLayers; ++layer) {
            int inter = 0;
            int uni = 0;
            const std::size_t base = static_cast<std::size_t>(layer * kExpertCount);
            for (int e = 0; e < kExpertCount; ++e) {
                const bool pa = a[base + static_cast<std::size_t>(e)] != 0;
                const bool pb = b[base + static_cast<std::size_t>(e)] != 0;
                if (pa && pb) ++inter;
                if (pa || pb) ++uni;
            }
            if (uni == 0) continue;
            ++used_layers;
            sum += static_cast<double>(inter) / static_cast<double>(uni);
        }
        if (used_layers == 0) return 0.f;
        return static_cast<float>(sum / used_layers);
    }

    std::vector<WeightRoutingTrace> steps_;
    uint32_t recorded_ = 0;
    uint32_t high_water_ = 0;
};

}  // namespace wag
