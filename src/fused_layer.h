// fused_layer.h -- one DeepSeek-R1 layer as direct CUDA launches (see fused_layer.cu), called from server_core_fused.cpp.
//   The engine hands device pointers of the layer's weights (ggml tensors' ->data), the KV cache slabs and the expert arena;
//   the context owns the activation buffers (8 rows), its own stream and the pinned host mirrors the engine reads back.
//   Host<->device traffic of the layer path goes through kernels on mapped pinned memory, never the copy engine, so the
//   loader's weight uploads can stream continuously beside it; the stream waits on the loader's events (fl_wait_event).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
struct FLCtx;
enum { FL_Q4K = 0, FL_Q6K = 1, FL_Q80 = 2, FL_Q5K = 3, FL_IQ2 = 4, FL_Q2K = 5, FL_Q3K = 6 };
struct FLW {   // device pointers of one layer's always-used weights (null where the layer has no such tensor)
    const void * attn_norm, * q_a, * q_a_norm, * q_b, * kv_a, * kv_a_norm, * kv_b, * k_bT, * attn_out, * ffn_norm, * gate_inp, * probs_b;
    const void * sh_gate, * sh_up, * sh_down;   // MoE layers: shared expert (Q5_K, Q5_K, Q6_K)
    const void * d_gate, * d_up, * d_down;      // dense layers 0-2 (Q4_K, Q4_K, Q6_K)
    int kv_b_v;    // 1: kv_b is the trimmed v half (attn_kv_b_v, [128 heads x 128 rows] of 420 B); 0: the full slab (256 rows per head, v at rows 128..255)
    int gate_f16;  // 1: gate_inp is F16 (ffn_gate_inp_f16); 0: F32
};
struct FLOut {   // host (pinned) mirrors filled by fl_fetch / fl_moe / fl_skip; null where not requested
    const int * sel; const float * wts; const float * ncur; const float * fi; const float * skt; const int * sel_s; const float * wts_s; const float * sc; const float * ccat; const float * h; const float * moe;
};
extern "C" {
FLCtx * fl_init(int kvp, int sc_cap);
// attention + router of one layer on N rows (<= 8) at positions pos[0..N) (pos0 = pos[0]); hin = N x 7168 host floats, or null to take the
// previous layer's output still on the device. dense: the layer's dense FFN runs too and h is ready for fetch (what 32). last_row >= 0: only
// that row goes on (moved to row 0; n = 1 afterwards). MoE layers: the shared expert runs here as well (its partials feed the spec pass and the combine).
void fl_attn(FLCtx *, const FLW *, int N, int pos0, const int * pos, void * KC, void * VC, const float * hin, int dense, int last_row, int wrow);   // wrow >= 0: this 8-row group of a wide pass starts at row wrow
// wide pass (cold prefill): buffers for wmax rows; per arena batch of experts, slots of up to 8 (row, k) pairs; then h/moe of all rows
int fl_wide_init(FLCtx *, int wmax, int KE);
void fl_wide_moe(FLCtx *, const void * gate, const void * up, const void * down, int down_fmt, int nexp, const int * midx, const int * rows, const int * kidx, const float * wk, int nslots, int KE, int par);   // par = batch parity (slot tables double-buffered)
void fl_wide_mark(FLCtx *, int par);   // record: this parity's staging copies + kernels are queued
void fl_wide_wait(FLCtx *, int par);   // host wait on that record before rewriting the parity's staging half / tables
void fl_wide_finish(FLCtx *, int n, int KE, FLOut * out);
// attribution: the routed experts' weighted pushes (KE x 7168) for one row, as added by the combine: narrow MoE (ids = slot indices, w = weights) / wide pass (row)
void fl_moe_outs(FLCtx *, int b, const int * ids, const float * w, int KE, float * out);
void fl_wide_outs(FLCtx *, int row, int KE, float * out);
void fl_wide_skip(FLCtx *, const float * mo, int n, FLOut * out);
// speculative attention + router of layer L+1 on (ffn_inp + shared expert output), writing KC1/VC1 at the same positions
void fl_spec(FLCtx *, const FLW * W1, void * KC1, void * VC1);
// geometry sketch of ncur (TPROJ F32 [256][7168]) and, with a table, the scan against GSKT (F16 [K][256])
void fl_sketch(FLCtx *, const float * TPROJ, const void * GSKT, int K);
// copy n rows of the requested outputs to the host and wait: what = 1 sel/wts/ncur/fi | 2 skt | 4 spec sel/wts | 8 scan | 16 ccat (N rows) | 32 h (dense)
void fl_fetch(FLCtx *, int n, int what, FLOut * out);
// routed experts (gate/up IQ2_XXS, down Q2_K or Q3_K, nexp slots packed as in the arena) for n rows with ids/w [n][KE] -> h (routed + shared + ffn_inp) and moe (routed only), waited
void fl_moe(FLCtx *, const void * gate, const void * up, const void * down, int nexp, int down_fmt, const int * ids, const float * w, int KE, int n, FLOut * out);
// every row served from memory: h = mo + shared + ffn_inp, waited
void fl_skip(FLCtx *, const float * mo, int n, FLOut * out);
// pinned host -> device on the layer stream by a kernel reading the mapped memory (ordered before the next launches; no copy engine)
void fl_h2d(FLCtx *, void * dst, const void * src, size_t bytes);
// the layer stream waits for an event (the loader's upload of a layer) before anything launched after this
void fl_wait_event(FLCtx *, cudaEvent_t ev);
double fl_gpu_ms(FLCtx *, int which);   // accumulated GPU time: 0 attention phase (incl. idle while the host waits), 1 experts, 2 attention+router+shared, 3 spec+scan
}
