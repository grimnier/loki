#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe {

// Two VRAM slots for sequential MoE layers. The host expert pool must already
// be page-locked (cudaHostAlloc or cudaHostRegister) and laid out as
// [layer][expert][expert_bytes].
//
// To overlap work, call sync_layer(L), prefetch_layer(L+1, ids), then enqueue
// layer L's kernels on compute_stream(). The next layer's IDs must already be
// available to the host. sync_layer only queues a device-side event wait; it
// does not wait for the transfer on the CPU.
class MoeStreamBuffer {
public:
    MoeStreamBuffer(const void* pinned_host_pool,
                    std::size_t host_pool_bytes,
                    int num_layers,
                    int experts_per_layer,
                    std::size_t expert_bytes,
                    int top_k)
        : host_pool_(static_cast<const std::byte*>(pinned_host_pool)),
          host_pool_bytes_(host_pool_bytes),
          num_layers_(num_layers),
          experts_per_layer_(experts_per_layer),
          expert_bytes_(expert_bytes),
          top_k_(top_k),
          slot_bytes_(checked_slot_bytes(top_k, expert_bytes)) {
        if (!host_pool_ || num_layers_ <= 0 || experts_per_layer_ <= 0 ||
            expert_bytes_ == 0 || top_k_ <= 0 || top_k_ > experts_per_layer_) {
            throw std::invalid_argument("invalid MoE stream buffer dimensions");
        }

        const auto expert_count = checked_mul(
            static_cast<std::size_t>(num_layers_),
            static_cast<std::size_t>(experts_per_layer_),
            "expert count overflow");
        const auto required_bytes = checked_mul(expert_count, expert_bytes_,
                                                "host pool size overflow");
        if (host_pool_bytes_ < required_bytes) {
            throw std::invalid_argument("host expert pool is too small");
        }
        require_pinned_host_memory(host_pool_);

        try {
            check(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
            for (auto& slot : slots_) {
                check(cudaMalloc(&slot.device, slot_bytes_), "cudaMalloc expert slot");
                check(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming),
                      "cudaEventCreate ready");
                check(cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming),
                      "cudaEventCreate reusable");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~MoeStreamBuffer() { release(); }

    MoeStreamBuffer(const MoeStreamBuffer&) = delete;
    MoeStreamBuffer& operator=(const MoeStreamBuffer&) = delete;
    MoeStreamBuffer(MoeStreamBuffer&&) = delete;
    MoeStreamBuffer& operator=(MoeStreamBuffer&&) = delete;

    // Queue the selected experts for `layer` into its ping-pong slot.
    // `topk_expert_ids` is read and validated before this call returns.
    void prefetch_layer(int layer, const int* topk_expert_ids) {
        check_layer(layer);
        if (!topk_expert_ids) {
            throw std::invalid_argument("topk_expert_ids is null");
        }
        if (last_prefetched_layer_ >= 0 && layer != last_prefetched_layer_ + 1) {
            throw std::logic_error("layers must be prefetched in increasing order");
        }
        if (last_prefetched_layer_ >= 0 && last_synced_layer_ != layer - 1) {
            throw std::logic_error("sync the previous layer before prefetching the next");
        }

        Slot& slot = slots_[slot_index(layer)];
        if (slot.layer >= 0 && slot.layer != layer - 2) {
            throw std::logic_error("attempt to overwrite a slot still assigned to another layer");
        }

        // Snapshot and validate before submitting any CUDA work, so a bad ID
        // cannot leave a partially queued transfer for this layer.
        std::vector<std::size_t> source_offsets;
        source_offsets.reserve(static_cast<std::size_t>(top_k_));
        for (int i = 0; i < top_k_; ++i) {
            const int expert_id = topk_expert_ids[i];
            if (expert_id < 0 || expert_id >= experts_per_layer_) {
                throw std::out_of_range("expert id is outside the layer");
            }

            const std::size_t expert_index =
                static_cast<std::size_t>(layer) *
                    static_cast<std::size_t>(experts_per_layer_) +
                static_cast<std::size_t>(expert_id);
            source_offsets.push_back(
                checked_mul(expert_index, expert_bytes_, "expert offset overflow"));
        }

        // This event is inserted before the caller launches the current layer.
        // Thus the old occupant has finished using this slot, while the next
        // transfer can run alongside the current layer's kernels.
        check(cudaEventRecord(slot.reusable, compute_stream_),
              "cudaEventRecord reusable");
        check(cudaStreamWaitEvent(transfer_stream_, slot.reusable, 0),
              "cudaStreamWaitEvent reusable");

        for (int i = 0; i < top_k_; ++i) {
            const std::size_t source_offset = source_offsets[static_cast<std::size_t>(i)];
            if (source_offset > host_pool_bytes_ ||
                expert_bytes_ > host_pool_bytes_ - source_offset) {
                throw std::out_of_range("expert extends beyond the host pool");
            }

            const auto* source = host_pool_ + source_offset;
            auto* destination = static_cast<std::byte*>(slot.device) +
                                static_cast<std::size_t>(i) * expert_bytes_;
            check(cudaMemcpyAsync(destination, source, expert_bytes_,
                                  cudaMemcpyHostToDevice, transfer_stream_),
                  "cudaMemcpyAsync expert");
        }

        check(cudaEventRecord(slot.ready, transfer_stream_), "cudaEventRecord ready");
        slot.layer = layer;
        slot.wait_enqueued = false;
        last_prefetched_layer_ = layer;
    }

    // Make subsequent work submitted to compute_stream() wait for this layer's
    // weights. This is a GPU-side dependency and returns without host sync.
    void sync_layer(int layer) {
        check_layer(layer);
        if (last_synced_layer_ >= 0 && layer != last_synced_layer_ + 1) {
            throw std::logic_error("layers must be synchronized in increasing order");
        }

        Slot& slot = slots_[slot_index(layer)];
        if (slot.layer != layer) {
            throw std::logic_error("layer has not been prefetched");
        }
        check(cudaStreamWaitEvent(compute_stream_, slot.ready, 0),
              "cudaStreamWaitEvent ready");
        slot.wait_enqueued = true;
        last_synced_layer_ = layer;
    }

    // The returned slot is safe for kernels queued after sync_layer(layer) on
    // compute_stream(). The caller must not use it from a different stream
    // without adding an equivalent event dependency.
    void* buffer_for_layer(int layer) {
        check_layer(layer);
        Slot& slot = slots_[slot_index(layer)];
        if (slot.layer != layer || !slot.wait_enqueued) {
            throw std::logic_error("call sync_layer(layer) before requesting its buffer");
        }
        return slot.device;
    }

    // CUDA's legacy default stream, stream 0, is the compute queue.
    cudaStream_t compute_stream() const noexcept { return compute_stream_; }
    cudaStream_t transfer_stream() const noexcept { return transfer_stream_; }
    std::size_t slot_bytes() const noexcept { return slot_bytes_; }

private:
    struct Slot {
        void* device = nullptr;
        cudaEvent_t ready = nullptr;
        cudaEvent_t reusable = nullptr;
        int layer = -1;
        bool wait_enqueued = false;
    };

    static constexpr std::size_t slot_index(int layer) noexcept {
        return static_cast<std::size_t>(layer) & 1u;
    }

    static std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
            throw std::overflow_error(what);
        }
        return a * b;
    }

    static std::size_t checked_slot_bytes(int top_k, std::size_t expert_bytes) {
        if (top_k <= 0 || expert_bytes == 0) {
            throw std::invalid_argument("invalid MoE stream buffer dimensions");
        }
        return checked_mul(static_cast<std::size_t>(top_k), expert_bytes,
                           "slot size overflow");
    }

    static void check(cudaError_t error, const char* call) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(call) + ": " +
                                     cudaGetErrorString(error));
        }
    }

    static void require_pinned_host_memory(const void* ptr) {
        cudaPointerAttributes attributes{};
        const cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
        if (error != cudaSuccess) {
            (void)cudaGetLastError();
            throw std::invalid_argument(
                "host expert pool must be pinned with cudaHostAlloc or cudaHostRegister");
        }
        if (attributes.type != cudaMemoryTypeHost) {
            throw std::invalid_argument("host expert pool is not pinned host memory");
        }
    }

    void check_layer(int layer) const {
        if (layer < 0 || layer >= num_layers_) {
            throw std::out_of_range("layer is outside the configured model");
        }
    }

    void release() noexcept {
        // Synchronization is intentionally confined to teardown, never the
        // per-layer enqueue path.
        if (transfer_stream_) (void)cudaStreamSynchronize(transfer_stream_);
        (void)cudaStreamSynchronize(compute_stream_);
        for (auto& slot : slots_) {
            if (slot.ready) (void)cudaEventDestroy(slot.ready);
            if (slot.reusable) (void)cudaEventDestroy(slot.reusable);
            if (slot.device) (void)cudaFree(slot.device);
            slot = Slot{};
        }
        if (transfer_stream_) (void)cudaStreamDestroy(transfer_stream_);
        transfer_stream_ = nullptr;
    }

    const std::byte* host_pool_;
    std::size_t host_pool_bytes_;
    int num_layers_;
    int experts_per_layer_;
    std::size_t expert_bytes_;
    int top_k_;
    std::size_t slot_bytes_;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    Slot slots_[2]{};
    int last_prefetched_layer_ = -1;
    int last_synced_layer_ = -1;
};

} // namespace moe
