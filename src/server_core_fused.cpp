// server_core_fused.cpp -- generated from server_core_async.cpp by patch_fused.py: --fused 1 runs the layers as direct launches (fused_layer.cu)
// server_core_async.cpp -- INGOT CORE: asynchronous two-stream DeepSeek-R1 runtime (lives next to server_core.cpp, which is untouched)
// compute stream: the exact R1 layer graphs (ggml CUDA kernels, bit-exact against llama.cpp in dsr1_chain tests)
// copy stream:    a loader thread stages the next layer's always-used weights (page-locked host zones -> ping/pong device slots)
// routed experts: the router's picks are read by file address with parallel unbuffered reads, every token
// step-A hook:    --base-gguf FILE loads the always-used (non-expert) weights from a separate GGUF (e.g. a compact one) instead of the shards
// usage: server_core_async [--prompt "text" | --prompt-ids FILE] [--tokens N] [--threads N] [--pinned-gb G] [--base-gguf FILE]
//                          [--shards-dir DIR] [--vocab FILE] [--results DIR] [--siblings]
// dsr1_decode: token-by-token DeepSeek-R1 decode with the exact ggml operations (as verified in dsr1_chain/dsr1_chain2).
// - K/V cache for all 61 layers lives on the GPU (cache length 256)
// - each layer's always-used weights get a home at startup: VRAM, else pinned RAM, else read from disk per token
// - routed experts: only the router's picks are fetched by address (parallel unbuffered reads) every token
// usage: dsr1_decode prompt_ids.bin n_new out_ids.bin shard1 shard2 shard3 shard4 [threads] [pinned_GB]
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <ctime>
#include <array>
#include <set>
#include <deque>
#include <cstring>
#include <unordered_map>
#include <map>
#include <cmath>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <iostream>
#include "../include/bpe_detokenizer.hpp"
#include "fused_layer.h"
#include <random>
static int FUSED = 0, FSLOTS = 4; static FLCtx * FL = nullptr; static bool TRIM = false, WIDE_SAID = false; static float SAMP_T = 0.f, SAMP_P = 1.f; static int SAMP_K = 0; static std::mt19937 SAMP_RNG(12345);
static int pick_tok(const float * lg, size_t n) {   // the head's decision: SAMP_T <= 0 -> the first argmax (unchanged); else top-k / top-p sampling at temperature SAMP_T
    if (SAMP_T <= 0.f) { return (int) (std::max_element(lg, lg + n) - lg); }
    const int K = SAMP_K > 0 ? std::min<int>(SAMP_K, (int) n) : (int) n; std::vector<int> ix(n); for (size_t i = 0; i < n; ++i) { ix[i] = (int) i; }
    std::partial_sort(ix.begin(), ix.begin() + K, ix.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    const double mx = lg[ix[0]]; std::vector<double> p(K); double Z = 0; for (int k = 0; k < K; ++k) { p[k] = std::exp(((double) lg[ix[k]] - mx) / SAMP_T); Z += p[k]; }
    int kk = K; if (SAMP_P < 1.f) { double cum = 0; for (int k = 0; k < K; ++k) { cum += p[k] / Z; if (cum >= SAMP_P) { kk = k + 1; break; } } }
    double tot = 0; for (int k = 0; k < kk; ++k) { tot += p[k]; } std::uniform_real_distribution<double> U(0.0, 1.0); const double r = U(SAMP_RNG) * tot; double acc = 0;
    for (int k = 0; k < kk; ++k) { acc += p[k]; if (r <= acc) { return ix[k]; } } return ix[kk - 1];
} static cudaEvent_t LD_EV[4] = { 0 }; static cudaEvent_t LT0[4] = { 0 }, LT1[4] = { 0 }; static double LD_BUSY = 0, LD_WAIT = 0, LD_BYTES = 0; static int LT_ON[4] = { 0, 0, 0, 0 }; static bool H_DEV = false;   /* fused layer path (patch_fused.py) */
#include <io.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
struct TI { int file; size_t off; ggml_type type; int64_t ne[4]; };
static std::string ATTRIB; static int ATTRIB_STEP = 0, ATTR_PNO = 0; static bool ATTR_ON = false; static std::vector<float> ATTR_REC, ATTR_X, ATTR_FI, ATTR_MOE, ATTR_OUTS; static int ATTR_IDS[8] = { 0 }; static float ATTR_W[8] = { 0 }; static int ATTR_KE = 0;
static void attr_moe(const std::vector<int32_t> & selh, const std::vector<float> & wh, const std::vector<int32_t> & idsc, const std::vector<float> & fih, const FLOut & FO, int KE, int64_t ar, bool wide) {
    const int E_ = 7168; ATTR_KE = KE; for (int k = 0; k < 8; ++k) { ATTR_IDS[k] = k < KE ? selh[ar * KE + k] : 0; ATTR_W[k] = k < KE ? wh[ar * KE + k] : 0.f; }
    ATTR_FI.assign(fih.begin() + ar * E_, fih.begin() + (ar + 1) * E_); ATTR_MOE.assign(E_, 0.f); if (FO.moe) { std::memcpy(ATTR_MOE.data(), FO.moe + ar * E_, E_ * 4); }
    ATTR_OUTS.assign((size_t) KE * E_, 0.f); if (wide) { fl_wide_outs(FL, (int) ar, KE, ATTR_OUTS.data()); } else { fl_moe_outs(FL, (int) ar, idsc.data() + ar * KE, wh.data() + ar * KE, KE, ATTR_OUTS.data()); }
}
static void attr_layer(int L, bool dense, const std::vector<float> & h, int64_t ao) {
    const int E_ = 7168; ATTR_REC.push_back((float) L); ATTR_REC.push_back(dense ? 1.f : 0.f); ATTR_REC.push_back((float) (dense ? 0 : ATTR_KE));
    for (int k = 0; k < 8; ++k) { ATTR_REC.push_back(dense ? 0.f : (float) ATTR_IDS[k]); } for (int k = 0; k < 8; ++k) { ATTR_REC.push_back(dense ? 0.f : ATTR_W[k]); }
    ATTR_REC.insert(ATTR_REC.end(), ATTR_X.begin(), ATTR_X.end());
    if (dense || ATTR_FI.empty()) { ATTR_REC.insert(ATTR_REC.end(), (size_t) 2 * E_, 0.f); } else { ATTR_REC.insert(ATTR_REC.end(), ATTR_FI.begin(), ATTR_FI.end()); ATTR_REC.insert(ATTR_REC.end(), ATTR_MOE.begin(), ATTR_MOE.end()); }
    for (int k = 0; k < 8; ++k) { if (!dense && k < ATTR_KE && ATTR_OUTS.size() >= (size_t) (k + 1) * E_) { ATTR_REC.insert(ATTR_REC.end(), ATTR_OUTS.begin() + (size_t) k * E_, ATTR_OUTS.begin() + (size_t) (k + 1) * E_); } else { ATTR_REC.insert(ATTR_REC.end(), (size_t) E_, 0.f); } }
    ATTR_REC.insert(ATTR_REC.end(), h.begin() + ao * E_, h.begin() + (ao + 1) * E_); ATTR_FI.clear(); ATTR_MOE.clear(); ATTR_OUTS.clear(); ATTR_KE = 0;
}
static void attr_finish(const std::vector<float> & logits, const std::vector<float> & h, std::map<std::string, ggml_tensor *> & HT, const std::map<std::string, TI> & TIDX, const std::vector<std::string> & PATHS, float eps) {
    const int E_ = 7168; FILE * f = std::fopen(ATTRIB.c_str(), "wb"); if (!f) { std::fprintf(stderr, "[attrib] cannot write %s\n", ATTRIB.c_str()); ATTR_ON = false; return; }
    const int32_t nrec = (int32_t) (ATTR_REC.size() / (19 + 12 * (size_t) E_)); int32_t hdr[4] = { 0x31525441, E_, nrec, 8 }; std::fwrite(hdr, 4, 4, f); std::fwrite(ATTR_REC.data(), 4, ATTR_REC.size(), f);
    const size_t V = logits.size(); int k1 = 0, k2 = -1; for (size_t i = 1; i < V; ++i) { if (logits[i] > logits[k1]) { k1 = (int) i; } } for (size_t i = 0; i < V; ++i) { if ((int) i != k1 && (k2 < 0 || logits[i] > logits[k2])) { k2 = (int) i; } }
    int32_t ids[2] = { k1, k2 }; float lg[2] = { logits[k1], logits[k2] }; std::fwrite(ids, 4, 2, f); std::fwrite(lg, 4, 2, f); std::fwrite(&eps, 4, 1, f);
    std::vector<float> g(E_); ggml_backend_tensor_get(HT["output_norm.weight"], g.data(), 0, (size_t) E_ * 4); std::fwrite(g.data(), 4, E_, f);
    const TI & th = TIDX.at("output.weight"); const size_t rb = ggml_row_size(th.type, E_); std::vector<uint8_t> row(rb); std::vector<float> hr(E_); FILE * mf = std::fopen(PATHS[th.file].c_str(), "rb");
    for (int j = 0; j < 2; ++j) { _fseeki64(mf, (long long) (th.off + (size_t) ids[j] * rb), SEEK_SET); std::fread(row.data(), 1, rb, mf); ggml_get_type_traits(th.type)->to_float(row.data(), hr.data(), E_); std::fwrite(hr.data(), 4, E_, f); } std::fclose(mf);
    std::fwrite(h.data(), 4, E_, f); std::fclose(f); std::fprintf(stderr, "[attrib] %d layers recorded for token %d (logit %.3f, runner-up %d at %.3f) -> %s\n", nrec, k1, lg[0], k2, lg[1], ATTRIB.c_str()); ATTR_ON = false;
}
static std::map<std::string, TI> TIDX;
static std::vector<std::string> PATHS;
static int NTHREADS = 4;
static float fbits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static void load_index() {
    for (int s = 0; s < (int) PATHS.size(); ++s) {
        ggml_context * meta = nullptr; gguf_init_params p = { true, &meta };
        gguf_context * g = gguf_init_from_file(PATHS[s].c_str(), p);
        if (!g) { std::fprintf(stderr, "cannot open %s\n", PATHS[s].c_str()); std::exit(2); }
        const size_t base = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * nm = gguf_get_tensor_name(g, i); ggml_tensor * t = ggml_get_tensor(meta, nm);
            TI ti; ti.file = s; ti.off = base + gguf_get_tensor_offset(g, i); ti.type = t->type; for (int d = 0; d < 4; ++d) { ti.ne[d] = t->ne[d]; }
            TIDX[nm] = ti;
        }
        gguf_free(g); ggml_free(meta);
    }
}
// ---- parallel unbuffered reads by address
static int RJ_SPLIT = 0; struct Job { int file; uint64_t off; size_t size; uint8_t * dst; };
static void read_jobs(std::vector<Job> & jobs, uint8_t * const * dev = nullptr, cudaStream_t st = nullptr) {
    thread_local HANDLE hs[8] = { 0 }; thread_local std::vector<uint8_t *> bufs; thread_local std::vector<size_t> caps; thread_local std::vector<HANDLE> evs;   // per calling thread, kept across calls
    std::vector<uint8_t *> sdev; uint8_t * const * devp = dev; if (RJ_SPLIT) { std::vector<Job> sj; const size_t PC = (size_t) 1 << 20; for (size_t ji = 0; ji < jobs.size(); ++ji) { const Job & j = jobs[ji]; for (size_t d = 0; d < j.size; d += PC) { sj.push_back({ j.file, j.off + d, std::min(PC, j.size - d), j.dst + d }); if (dev) { sdev.push_back(dev[ji] ? dev[ji] + d : nullptr); } } } jobs.swap(sj); if (dev) { devp = sdev.data(); } } const uint64_t A = 4096; const size_t n = jobs.size(); int fail = 0;
    while (bufs.size() < n) { bufs.push_back(nullptr); caps.push_back(0); evs.push_back(CreateEventA(nullptr, TRUE, FALSE, nullptr)); }
    std::vector<OVERLAPPED> ov(n); std::vector<uint64_t> a0s(n);
    for (size_t i = 0; i < n; ++i) {   // issue everything now: the drive sees the whole batch at once
        const Job & j = jobs[i];
        if (!hs[j.file]) { hs[j.file] = CreateFileA(PATHS[j.file].c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr); }
        const uint64_t a0 = j.off / A * A, a1 = (j.off + j.size + A - 1) / A * A; const size_t need = (size_t) (a1 - a0); a0s[i] = a0;
        if (need > caps[i]) { if (bufs[i]) { _aligned_free(bufs[i]); } caps[i] = need; bufs[i] = (uint8_t *) _aligned_malloc(need, (size_t) A); }
        std::memset(&ov[i], 0, sizeof(OVERLAPPED)); ov[i].Offset = (DWORD) (a0 & 0xffffffffu); ov[i].OffsetHigh = (DWORD) (a0 >> 32); ov[i].hEvent = evs[i]; ResetEvent(evs[i]);
        if (!ReadFile(hs[j.file], bufs[i], (DWORD) need, nullptr, &ov[i]) && GetLastError() != ERROR_IO_PENDING) { fail = 1; }
    }
    for (size_t i = 0; i < n; ++i) {   // collect in order; each piece goes to the GPU the moment it is in
        const Job & j = jobs[i]; DWORD got = 0;
        if (!GetOverlappedResult(hs[j.file], &ov[i], &got, TRUE) || (uint64_t) got < (j.off - a0s[i]) + j.size) { fail = 1; continue; }
        std::memcpy(j.dst, bufs[i] + (j.off - a0s[i]), j.size);
        if (devp && devp[i]) { if (FUSED && FL) { fl_h2d(FL, devp[i], j.dst, j.size); } else { cudaMemcpyAsync(devp[i], j.dst, j.size, cudaMemcpyHostToDevice, st); } }
    }
    if (fail) { std::fprintf(stderr, "direct read failed\n"); std::exit(5); }
}
static void add_jobs(std::vector<Job> & jobs, const TI & ti, uint64_t rel, size_t size, uint8_t * dst) {
    const size_t P = (size_t) 8 << 20; for (size_t d = 0; d < size; d += P) { jobs.push_back({ ti.file, ti.off + rel + d, std::min(P, size - d), dst + d }); }
}
// ---- per-layer always-used weights with a tier: 0 = VRAM, 1 = pinned RAM, 2 = disk
struct LayerW {
    int tier = 0; ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; ggml_backend_buffer_t hostbuf = nullptr; uint8_t * host = nullptr;
    std::vector<std::string> names; std::vector<size_t> offs; size_t total = 0; std::map<std::string, ggml_tensor *> t;
};
static const size_t UPCHUNK = (size_t) 4 << 20;   // loader copy granularity: a main-thread copy waits at most one chunk
static int NSLOT = 3; static ggml_backend_buffer_t SCR[4] = { nullptr, nullptr, nullptr, nullptr }; static uint8_t * SCRB[4] = { nullptr, nullptr, nullptr, nullptr };
static ggml_backend_buffer_t WST[4] = { nullptr, nullptr, nullptr, nullptr }; static uint8_t * WSTB[4] = { nullptr, nullptr, nullptr, nullptr }; static size_t WST_MAX = 0;
static std::mutex MU; static std::condition_variable CV; static int UPL = -1, CONS = -1, UPWIN = 0;   // UPWIN: main thread is inside an expert read -> the loader may upload now
static ggml_gallocr_t GA_ATTN = nullptr, GA_MOE = nullptr, GA_SKIP = nullptr, GA_SK = nullptr;   /* persistent graph allocators */ static double B_DISK_W = 0;
static ggml_backend_buffer_t PIN_STAGE = nullptr; static uint8_t * PIN_STAGE_BASE = nullptr; static size_t PIN_STAGE_SIZE = 0;
static void shrink_pin_stage(size_t n) { if (PIN_STAGE_SIZE <= n) { return; } if (PIN_STAGE) { ggml_backend_buffer_free(PIN_STAGE); } PIN_STAGE = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), n); PIN_STAGE_BASE = (uint8_t *) ggml_backend_buffer_get_base(PIN_STAGE); PIN_STAGE_SIZE = n; }
static void ensure_pin_stage(size_t n) {
    if (n <= PIN_STAGE_SIZE) { return; }
    if (PIN_STAGE) { ggml_backend_buffer_free(PIN_STAGE); }
    PIN_STAGE = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), n); PIN_STAGE_BASE = (uint8_t *) ggml_backend_buffer_get_base(PIN_STAGE); PIN_STAGE_SIZE = n;
}
static double T_W = 0, T_EX = 0, T_EXUP = 0, T_C = 0, T_HEAD = 0; static double B_DISK = 0;
static void layer_bring(LayerW & w) {             // make this layer's always-used weights present on the GPU
    if (w.tier == 0) { return; }
    const auto t0 = clk::now();
    const uint8_t * src = w.host;
    if (w.tier == 2) {
        ensure_pin_stage(w.total); std::vector<Job> jobs;
        for (size_t i = 0; i < w.names.size(); ++i) { add_jobs(jobs, TIDX.at(w.names[i]), 0, ggml_nbytes(w.t[w.names[i]]), PIN_STAGE_BASE + w.offs[i]); }
        read_jobs(jobs); src = PIN_STAGE_BASE; B_DISK += (double) w.total;
    }
    for (size_t i = 0; i < w.names.size(); ++i) { ggml_backend_tensor_set(w.t[w.names[i]], src + w.offs[i], 0, ggml_nbytes(w.t[w.names[i]])); }
    T_W += secs(t0);
}
struct Experts { ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; std::map<std::string, ggml_tensor *> t; };
static ggml_backend_buffer_t EXP_ARENA = nullptr; static size_t EXP_ARENA_SIZE = 0; static cudaStream_t EXP_STREAM = nullptr;
// ---- expert prefetch: from each layer's output, predict layers L+1 and L+2's picks and read them into a pinned host cache ahead of time
struct PFSlot { int layer = -1, expert = -1, state = 0, src = 0; uint8_t * mem = nullptr; };   // state: 0 empty, 1 reading, 2 ready
static std::vector<PFSlot> PF; static std::mutex PFM; static std::condition_variable PFCV; static int PF_CUR = 0, PF_TOP = 6, PF_D0 = 2, PF_D1 = 2, PF_ROWS = 1 << 30; static bool PF_ON = false, PF_STOP = false;
static std::vector<std::vector<float>> RW_(61), RB_(61), RN_(61); static std::deque<std::pair<int, std::vector<float>>> PFQ;
static size_t PF_HIT = 0, PF_MISS = 0, PF_LATE = 0, PF_JOBS = 0, PF_DROP = 0, PF_SKIP = 0, PF_FULL = 0; static double PF_BYTES = 0, PF_TPRED = 0, PF_TREAD = 0;
// ---- route memory: the experts each (previous token id, token id, layer) was routed to, labelled once, kept on disk, reused before any state exists
static std::unordered_map<uint64_t, std::array<int16_t, 8>> RMEM; static bool RMEM_ON = false; static int RMEM_MAXL = 12;
static std::vector<int32_t> ROW_TOK, ROW_PREV; static std::deque<std::pair<int, int>> PFX; static size_t RMEM_QUEUED = 0;
static inline uint64_t rkey(int32_t prev, int32_t tok, int L) { return ((uint64_t) (uint32_t) (prev + 1) << 32) | ((uint64_t) (uint32_t) tok << 8) | (uint64_t) L; }
struct GEntry { std::vector<float> sk; uint64_t off = 0; uint32_t used = 0, pos = 0, runid = 0, crun = 0, cpos = 0; int32_t nxt[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }; int32_t route[8] = { -1, -1, -1, -1, -1, -1, -1, -1 }; uint64_t kvoff = ~0ull; }; static uint32_t G_CLOCK = 1;
static inline uint64_t gkey(uint32_t runid, uint32_t pos) { return ((uint64_t) runid << 32) | pos; }
static std::vector<ggml_tensor *> GSKT(61, nullptr); static std::vector<ggml_backend_buffer_t> GSKB(61, nullptr); static std::vector<ggml_context *> GSKC(61, nullptr); static std::vector<uint32_t> GVER(61, 1), GSKV(61, 0), GEVICT(61, 0), GSKE(61, 0); static std::vector<size_t> GSKN(61, 0); static std::vector<std::vector<uint32_t>> GDIRTY(61); static std::vector<std::unordered_map<uint64_t, uint32_t>> GIDX(61); static int ROUTE_D = 0; static int SPEC_PROBE = 0, SPEC_L = -1, SPEC_READ = 0, LD_INFLIGHT = 3; static ggml_tensor * HDEV = nullptr; static bool HDEV_OK = false; static ggml_context * PREV_C2 = nullptr; static bool NODES_PRINTED = false; static int HCHAIN = 0; static unsigned HDEV_PASS = 0; static double T_FA = 0, T_FM = 0; static double T_AB_BUILD = 0, T_AB_ALLOC = 0, T_AB_SET = 0; static ggml_gallocr_t GA_SPEC = nullptr; static size_t SPEC_Q = 0; static int32_t SPEC_SEL[8]; static float SPEC_W[8]; static size_t SPEC_N = 0, SPEC_H2 = 0, SPEC_H4 = 0, SPEC_LAYERS = 0; static double T_SPEC = 0, T_SPECW = 0; static int GPRED_D = 0, GPRED_K = 8; static float GPRED_T = 0.3f; static size_t GPRED_STEPS = 0, GPRED_N = 0, GPRED_W = 0, GPRED_WH = 0; static float ROUTE_SIM = 0.97f; static std::vector<uint32_t> ROUTE_COVER(61, 0); static uint32_t PASS_ID = 0; static std::set<int> ROUTE_SENT; static size_t ROUTE_Q = 0; static size_t PF_HIT_SRC[4] = { 0, 0, 0 }; static int CHAIN_TRACE = 0;
static bool MLA = false, MLA_SIDE = false; static int PREFILL_CHUNK = 8; static FILE * TRACEF = nullptr; static int TEXT_CAP = 3; static int GVOTE = 5; static int TEXT_MIN = 3; static std::map<int, std::pair<int, int>> TD_STATS; static bool CHAT = false; static FILE * PROTO = nullptr; static std::atomic<bool> CHAT_STOP{false}; static std::mutex CHM; static std::condition_variable CHCV; static std::deque<std::string> CHQ; static int CHAT_THINK = 0, CHAT_ANSWER = 0;
static void chat_send(const std::string & m) { if (!PROTO) { return; } std::fputs(m.c_str(), PROTO); std::fputc('\n', PROTO); std::fflush(PROTO); }
static std::string chat_esc(const std::string & x) { std::string o; for (char c : x) { if (c == '\\') { o += "\\\\"; } else if (c == '\n') { o += "\\n"; } else if (c != '\r') { o += c; } } return o; }
static std::string chat_unesc(const std::string & x) { std::string o; for (size_t i = 0; i < x.size(); ++i) { if (x[i] == '\\' && i + 1 < x.size()) { ++i; o += x[i] == 'n' ? '\n' : x[i]; } else { o += x[i]; } } return o; }
static void chat_reader() { std::string line; while (std::getline(std::cin, line)) { if (!line.empty() && line.back() == '\r') { line.pop_back(); } if (line == "X") { CHAT_STOP = true; continue; } { std::lock_guard<std::mutex> lk(CHM); CHQ.push_back(line); } CHCV.notify_all(); } { std::lock_guard<std::mutex> lk(CHM); CHQ.push_back("Q"); } CHCV.notify_all(); }
static std::string chat_next() { std::unique_lock<std::mutex> lk(CHM); CHCV.wait(lk, [] { return !CHQ.empty(); }); std::string m = CHQ.front(); CHQ.pop_front(); return m; } static std::vector<uint8_t> ROW_DERIVED; static const uint32_t G_EXACT = 0x80000000u; static uint32_t G_EXACT_SINCE = 1791401400u;   /* 2026-10-07 14:31:40 local: first run with fixed-width passes and fp32 outputs; older states are draft-only */ static uint32_t G_POS = 0, G_RUNID = 0; static int GNEXT_L = 50; static std::vector<std::array<int32_t, 16>> GEO_NR; static std::vector<int> G_MISSROW; static FILE * MOEDF = nullptr; static int PASS_W = 8; static int MOE_G = 4; static bool RENORM_EXPERTS = true;   /* kept experts rescaled to the routed scale: 14/40 tokens match the 8-expert run vs 6/40 unscaled (fid_*.log) */ static float COLD_SHARE = 0.7f; static bool COLD_TEXT = false; static int COLD_DRAFT = 2; static int PREFILL_EXPERTS = -1;   /* -1: same as --experts */ static size_t COLD_PASSES = 0; static int64_t N_PAD_FROM = (int64_t) 1 << 40;   /* rows >= N_PAD_FROM in the current pass are padding */
struct FamM { float w; uint32_t crun; std::array<int32_t, 16> nxt; };
static std::vector<std::vector<FamM>> GEO_FAM; static std::vector<int> GEO_FAML; static std::vector<std::pair<uint32_t, uint32_t>> GEO_NRE; static std::vector<float> GEO_NRS; static std::unordered_map<uint32_t, std::vector<int32_t>> RUNP; static std::string RUNP_PATH = R"(ptx_trees\run_prompts.bin)"; static std::vector<int32_t> OURQ;
static float FAM_SIM = 0.6f, FAM_SHARE = 0.5f, GSIM_PRE = -1.0f; static int FAM_K = 8, FAM_L = 40, FAM_SOLO = 4; static size_t FAM_USED = 0, FAM_OK = 0;
static std::vector<int32_t> fam_question(const std::vector<int32_t> & P) { size_t s0 = 0; for (size_t i = 0; i < P.size(); ++i) { if (P[i] == 128803) { s0 = i + 1; } } std::vector<int32_t> q; for (size_t i = s0; i < P.size() && P[i] < 128000; ++i) { q.push_back(P[i]); } return q; }
static void runp_add(uint32_t r, const std::vector<int32_t> & P) { RUNP[r] = fam_question(P); if (RUNP_PATH.empty()) { return; } FILE * f = std::fopen(RUNP_PATH.c_str(), "ab"); if (!f) { return; } const uint32_t nn = (uint32_t) P.size(); std::fwrite(&r, 4, 1, f); std::fwrite(&nn, 4, 1, f); if (nn) { std::fwrite(P.data(), 4, nn, f); } std::fclose(f); }
static void runp_load() { if (RUNP_PATH.empty()) { return; } FILE * f = std::fopen(RUNP_PATH.c_str(), "rb"); if (!f) { return; } uint32_t r = 0, nn = 0; while (std::fread(&r, 4, 1, f) == 1 && std::fread(&nn, 4, 1, f) == 1 && nn < 100000) { std::vector<int32_t> P(nn); if (nn && std::fread(P.data(), 4, nn, f) != nn) { break; } RUNP[r] = fam_question(P); } std::fclose(f); std::fprintf(stderr, "[family] %zu run prompts known (%s)\n", RUNP.size(), RUNP_PATH.c_str()); } static int GEO_LAST = -1; static size_t GN_USED = 0, GN_OK = 0, G_ADOPT = 0;
static double T_AB = 0, T_GET = 0, T_LD = 0, T_MB = 0, T_ST = 0, T_SK = 0, T_FR = 0;
static FILE * GOUTF = nullptr; static uint64_t GOUT_END = 0; static std::mutex GOUTM;   // outputs: fp16[7168] records appended to <geo>.out, read back by offset on a hit
static FILE * GKVF = nullptr; static uint64_t GKV_END = 0; static std::vector<ggml_fp16_t> KVLAT; static int GEO_RUN = 32; static size_t GR_TOK = 0, GR_STRETCH = 0; static std::vector<int32_t> GEO_SEED; static ggml_gallocr_t GA_KV = nullptr;   /* stored attention latents: 61 layers x 576 fp16 per position, keyed from the layer-60 state */
static FILE * GOUTF32 = nullptr; static uint64_t GOUT32_END = 0; static const uint64_t GOUT_F32 = 1ull << 63;   /* offset bit 63: fp32 record in <geo>.out32 (new outputs); clear: fp16 record in <geo>.out (legacy) */
static void gout_read(uint64_t off, float * dst) { if (off & GOUT_F32) { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GOUTF32, (long long) (off & ~GOUT_F32), SEEK_SET); std::fread(dst, 4, 7168, GOUTF32); return; } std::vector<ggml_fp16_t> h(7168); { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GOUTF, (long long) off, SEEK_SET); std::fread(h.data(), 2, 7168, GOUTF); } ggml_fp16_to_fp32_row(h.data(), dst, 7168); }
static uint64_t gout_append(const float * out) { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GOUTF32, 0, SEEK_END); const uint64_t off = GOUT32_END; std::fwrite(out, 4, 7168, GOUTF32); GOUT32_END += 7168 * 4; return off | GOUT_F32; }   // geometry memory: unit state direction (fp16) + 128-d sketch -> the expert output it produced (fp16)
static std::vector<std::vector<GEntry>> GMEM(61); static bool GMEM_ON = false; static float GSIM = 0.999f, GSIM_DRAFT = 0.999f;   /* sketch cosine is a true cosine since patch_sketch_norm16: same-path states sit at 1.0000, other contexts below 0.999 */ static size_t GMAXLOAD = (size_t) 1 << 30, GCAP = 4096, G_ROWS = 0, G_HITROWS = 0, G_ROWSKIP = 0; static double G_TLOOK = 0;
static const int GSK = 256; static std::vector<float> GPROJ; static ggml_tensor * TPROJ = nullptr;
static void gproj_init() { if (!GPROJ.empty()) { return; } GPROJ.resize((size_t) GSK * 7168); uint64_t x = 0x9E3779B97F4A7C15ull; for (auto & v : GPROJ) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v = (x & 1) ? 0.0625f : -0.0625f; } }
static void gsketch(const float * x, float * o) { for (int j = 0; j < GSK; ++j) { const float * q = GPROJ.data() + (size_t) j * 7168; float a0 = 0, a1 = 0, a2 = 0, a3 = 0; for (int d = 0; d < 7168; d += 4) { a0 += q[d] * x[d]; a1 += q[d + 1] * x[d + 1]; a2 += q[d + 2] * x[d + 2]; a3 += q[d + 3] * x[d + 3]; } o[j] = a0 + a1 + a2 + a3; } double sn = 0; for (int j = 0; j < GSK; ++j) { sn += (double) o[j] * o[j]; } const float si = 1.0f / (float) std::sqrt(sn + 1e-20); for (int j = 0; j < GSK; ++j) { o[j] *= si; } }
static GEntry gmake(const float * xunit, const float * out) { GEntry ge; ge.sk.resize(GSK); gsketch(xunit, ge.sk.data()); ge.off = gout_append(out); return ge; }
static bool geo_same_question(uint32_t run) { auto it = RUNP.find(run); return it != RUNP.end() && !OURQ.empty() && it->second == OURQ; }   /* the stored walk's question is this turn's question */
static std::vector<int32_t> fam_realize(const FamM & m, const std::vector<int32_t> & ourq) {   /* a member's continuation, with any stretch it copied from its own question replaced by the same stretch of ours */
    int nn = 0; while (nn < 16 && m.nxt[nn] >= 0) { nn++; } std::vector<int32_t> out; auto it = RUNP.find(m.crun);
    if (it == RUNP.end() || it->second.size() < 2 || ourq.empty()) { for (int q = 0; q < nn && !(m.nxt[q] >= 128000 && m.nxt[q] != 128799); ++q) { out.push_back(m.nxt[q]); } return out; }
    const std::vector<int32_t> & Q = it->second; int j = 0;
    while (j < nn && (int) out.size() < 16 && !(m.nxt[j] >= 128000 && m.nxt[j] != 128799)) {
        int bo = -1, bl = 0;
        for (int o = 0; o < (int) Q.size(); ++o) { int l = 0; while (j + l < nn && o + l < (int) Q.size() && m.nxt[j + l] == Q[o + l]) { l++; } if (l > bl) { bl = l; bo = o; } }
        if (bl >= 2) { const bool to_end = (bo + bl >= (int) Q.size()) || (j + bl >= nn); for (int o = bo; o < (int) ourq.size() && (to_end || o < bo + bl) && (int) out.size() < 16; ++o) { out.push_back(ourq[o]); } j += bl; }
        else { out.push_back(m.nxt[j]); j++; }
    }
    return out; }
static std::vector<int32_t> fam_vote(const std::vector<FamM> & fam, int tok, const std::vector<int32_t> & ourq, int lim, float share) {   /* position-wise weighted vote among the members that agree with the head on the first token */
    std::vector<std::pair<float, std::vector<int32_t>>> M; for (const FamM & m : fam) { std::vector<int32_t> r = fam_realize(m, ourq); if (!r.empty() && r[0] == tok) { M.push_back({ m.w, r }); } }
    std::vector<int32_t> out; if (M.empty()) { return out; } { float mw = -1.0f; for (auto & m : M) { mw = std::max(mw, m.first); } if (mw >= GSIM) { std::vector<std::pair<float, std::vector<int32_t>>> ex; for (auto & m : M) { if (m.first >= GSIM) { ex.push_back(m); } } M.swap(ex); } }   /* same-path members (at the exact threshold) vote alone; the wider family votes only when no exact match exists */
    for (int p = 1; (int) out.size() < lim; ++p) {
        std::map<int32_t, float> vote; float tot = 0.0f; for (auto & m : M) { if ((int) m.second.size() > p) { vote[m.second[p]] += m.first; tot += m.first; } }
        if (vote.empty()) { break; } int32_t bt = -1; float bw = -1.0f; for (auto & kv : vote) { if (kv.second > bw) { bw = kv.second; bt = kv.first; } }
        if (bw < share * tot) { break; }
        std::vector<std::pair<float, std::vector<int32_t>>> keep; for (auto & m : M) { if ((int) m.second.size() > p && m.second[p] == bt) { keep.push_back(m); } }
        if (keep.size() == 1 && keep[0].first < GSIM_DRAFT && (int) out.size() >= FAM_SOLO) { break; }
        out.push_back(bt); M.swap(keep);
    }
    return out; }
static std::unordered_map<uint64_t, std::vector<float>> OMEM; static bool OMEM_ON = false; static size_t OMEM_SKIPS = 0, OMEM_LAYERS_SEEN = 0;
static std::vector<float> HMID, HFIN; static int MID_L = 30; static FILE * DUMPF = nullptr; static size_t DUMPED = 0;
static std::vector<float> DR_MX[2], DR_MY[2], DR_X[2], DR_A[2], GEO_X; static int DR_N[2] = { 0, 0 }; static bool DR_ON = false; static size_t DR_USED = 0, DR_OK = 0; static float DR_MARGIN = 0.0f;
static int pf_protect() { return std::max(PF_CUR + std::max(PF_D1, std::max(ROUTE_D, GPRED_D)) + 1, (RMEM_ON && PF_CUR <= RMEM_MAXL) ? RMEM_MAXL : 0); }
static std::atomic<int> RD_BUSY{ 0 };
static uint8_t * PIO = nullptr; static size_t PIO_CAP = (size_t) 16 << 20, PIO_OFF = 0;   // pinned ring for staged copies
static uint8_t * pio_take(size_t n) { if (!PIO) { if (cudaHostAlloc((void **) &PIO, PIO_CAP, cudaHostAllocDefault) != cudaSuccess) { PIO = (uint8_t *) std::malloc(PIO_CAP); std::fprintf(stderr, "[pinned io] ring is pageable (pinned pool exhausted)\n"); } } n = (n + 255) & ~(size_t) 255; if (PIO_OFF + n > PIO_CAP) { PIO_OFF = 0; } uint8_t * q = PIO + PIO_OFF; PIO_OFF += n; return q; }
static void pset(ggml_backend_t be, ggml_tensor * t, const void * src, size_t off, size_t n) { if (!SPEC_READ || !PIO && n > PIO_CAP || n > ((size_t) 16 << 20)) { ggml_backend_tensor_set(t, src, off, n); return; } uint8_t * q = pio_take(n); std::memcpy(q, src, n); ggml_backend_tensor_set_async(be, t, q, off, n); }
struct PGet { void * dst; uint8_t * q; size_t n; }; static std::vector<PGet> PGQ;
static void pget(ggml_backend_t be, ggml_tensor * t, void * dst, size_t off, size_t n) { if (!SPEC_READ || n > ((size_t) 16 << 20)) { ggml_backend_tensor_get(t, dst, off, n); return; } uint8_t * q = pio_take(n); ggml_backend_tensor_get_async(be, t, q, off, n); PGQ.push_back({ dst, q, n }); }
static void pget_sync(ggml_backend_t be) { if (PGQ.empty()) { return; } ggml_backend_synchronize(be); for (auto & g : PGQ) { std::memcpy(g.dst, g.q, g.n); } PGQ.clear(); }
static int pf_find(int L, int e) { for (size_t i = 0; i < PF.size(); ++i) { if (PF[i].state && PF[i].layer == L && PF[i].expert == e) { return (int) i; } } return -1; }
static void pf_fetch(int Lt, const std::set<int> & want) {        // claim cache slots for these experts of layer Lt and read them from disk
    const char * kinds[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    std::vector<int> slots, exps;
    { std::lock_guard<std::mutex> lk(PFM);
      for (int e : want) {
          if (pf_find(Lt, e & 0xFFFF) >= 0) { continue; }
          const int prot = pf_protect(); int v = -1;
          for (size_t i = 0; i < PF.size(); ++i) { if (PF[i].state == 0 || (PF[i].state == 2 && (PF[i].layer < PF_CUR || PF[i].layer > prot))) { v = (int) i; break; } }
          if (v < 0) { PF_FULL++; break; }
          PF[v].layer = Lt; PF[v].expert = e & 0xFFFF; PF[v].src = (e >> 16) & 3; PF[v].state = 1; slots.push_back(v); exps.push_back(e & 0xFFFF);
      } }
    if (slots.empty()) { return; }
    std::vector<Job> jobs; const std::string p = "blk." + std::to_string(Lt) + ".";
    for (size_t j = 0; j < slots.size(); ++j) { size_t o = 0;
        for (int k = 0; k < 3; ++k) { const TI & ti = TIDX.at(p + kinds[k]); const size_t es = ggml_row_size(ti.type, ti.ne[0]) * (size_t) ti.ne[1];
            jobs.push_back({ ti.file, ti.off + (uint64_t) exps[j] * es, es, PF[slots[j]].mem + o }); o += es; PF_BYTES += (double) es; } }
    const auto tr0 = clk::now(); while (RD_BUSY.load() > 0) { std::this_thread::sleep_for(std::chrono::microseconds(100)); } read_jobs(jobs); PF_TREAD += secs(tr0);
    { std::lock_guard<std::mutex> lk(PFM); for (int v : slots) { PF[v].state = 2; } } PFCV.notify_all();
}
static void pf_worker() {
    for (;;) {
        std::pair<int, std::vector<float>> job; std::vector<std::pair<int, int>> mx;
        { std::unique_lock<std::mutex> lk(PFM); PFCV.wait(lk, [] { return PF_STOP || !PFQ.empty() || !PFX.empty(); }); if (PF_STOP) { return; }
          if (!PFX.empty()) { mx.assign(PFX.begin(), PFX.end()); PFX.clear(); } else { job = std::move(PFQ.front()); PFQ.pop_front(); PF_JOBS++; } }
        if (!mx.empty()) {                                         // route-memory labels: these experts are read now, before any state exists
            std::map<int, std::set<int>> byL; for (auto & q : mx) { if (q.first > PF_CUR) { byL[q.first].insert(q.second); } }
            for (auto & kv : byL) { pf_fetch(kv.first, kv.second); }
            continue;
        }
        const int L0 = job.first; const std::vector<float> & h = job.second; const int n = (int) (h.size() / 7168);
        for (int Lt = std::max(3, L0 + PF_D0); Lt <= std::min(60, L0 + PF_D1); ++Lt) {
            if (Lt <= PF_CUR) { PF_SKIP++; continue; }   // that layer is already being loaded: too late
            const auto tp0 = clk::now();
            const float * W = RW_[Lt].data(); const float * B = RB_[Lt].data(); const float * G = RN_[Lt].data();
            std::set<int> want; std::vector<float> xn(7168); std::vector<std::pair<float, int>> sc(256);
            for (int r = 0; r < std::min(n, PF_ROWS); ++r) {   // --prefetch-rows 1: only the anchor row's experts are read ahead (draft rows are speculative twice over)
                const float * x = h.data() + (size_t) r * 7168; double ss = 0; for (int d = 0; d < 7168; ++d) { ss += (double) x[d] * x[d]; }
                const float k = 1.0f / std::sqrt((float) (ss / 7168) + 1e-6f); for (int d = 0; d < 7168; ++d) { xn[d] = x[d] * k * G[d]; }
                for (int e = 0; e < 256; ++e) { const float * w = W + (size_t) e * 7168; float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    for (int d = 0; d < 7168; d += 4) { a0 += w[d] * xn[d]; a1 += w[d + 1] * xn[d + 1]; a2 += w[d + 2] * xn[d + 2]; a3 += w[d + 3] * xn[d + 3]; }
                    sc[e] = { 1.0f / (1.0f + std::exp(-(a0 + a1 + a2 + a3))) + B[e], e }; }
                std::partial_sort(sc.begin(), sc.begin() + PF_TOP, sc.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
                for (int q = 0; q < PF_TOP; ++q) { want.insert(sc[q].second); }
            }
            PF_TPRED += secs(tp0);
            pf_fetch(Lt, want);
        }
    }
}
static Experts load_experts(ggml_backend_t be, const std::string & p, const std::vector<int32_t> & used) {
    const auto t0 = clk::now();
    const int Lnow = std::atoi(p.c_str() + 4); std::vector<int> hitslot(used.size(), -1);
    if (PF_ON) {                                                   // which of this layer's experts the prefetcher already has (wait briefly for ones still reading)
        std::unique_lock<std::mutex> lk(PFM); PF_CUR = Lnow;
        for (size_t u = 0; u < used.size(); ++u) {
            if (GPRED_D > 0) { GPRED_W++; if (ROUTE_SENT.count((Lnow << 9) | used[u])) { GPRED_WH++; } } const int s = pf_find(Lnow, used[u]);
            if (s < 0) { PF_MISS++; continue; }
            if (PF[s].state == 1) { PF_LATE++; PFCV.wait(lk, [&] { return PF[s].state == 2; }); }
            hitslot[u] = s; PF_HIT++; PF_HIT_SRC[PF[s].src & 3]++;
        }
        if (ROUTE_COVER[Lnow] == PASS_ID && CHAIN_TRACE > 0) { CHAIN_TRACE--; std::fprintf(stderr, "  [chain-check] pass %u L%d wants:", PASS_ID, Lnow); for (size_t u = 0; u < used.size(); ++u) { std::fprintf(stderr, " %d%s", used[u], hitslot[u] >= 0 ? (PF[hitslot[u]].src == 1 ? "(chain)" : "(router)") : ""); } std::fprintf(stderr, " | slots:"); for (auto & sl : PF) { if (sl.state && sl.layer == Lnow) { std::fprintf(stderr, " %d/s%d", sl.expert, sl.src); } } std::fprintf(stderr, "\n"); }
    }
    Experts w; ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true }; w.ctx = ggml_init(ip);
    const char * kinds[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    size_t tot = 0; std::vector<size_t> es(3);
    for (int k = 0; k < 3; ++k) { const TI & ti = TIDX.at(p + kinds[k]); es[k] = ggml_row_size(ti.type, ti.ne[0]) * (size_t) ti.ne[1]; tot += es[k] * used.size();
        w.t[kinds[k]] = ggml_new_tensor_3d(w.ctx, ti.type, ti.ne[0], ti.ne[1], (int64_t) used.size()); }
    const size_t al = 256, need = tot + 3 * al;
    const size_t EXP_CAP = (size_t) 520 << 20;                    // persistent arena up to ~40 experts; bigger sets (prefill) get a one-off buffer
    ggml_backend_buffer_t tgt;
    if (need <= EXP_CAP) {
        if (need > EXP_ARENA_SIZE) { if (EXP_ARENA) { ggml_backend_buffer_free(EXP_ARENA); } EXP_ARENA_SIZE = std::min(EXP_CAP, std::max(need, (size_t) (EXP_ARENA_SIZE * 1.25))); EXP_ARENA = ggml_backend_alloc_buffer(be, EXP_ARENA_SIZE); }
        tgt = EXP_ARENA; w.buf = nullptr;
    } else { if (EXP_ARENA) { ggml_backend_buffer_free(EXP_ARENA); EXP_ARENA = nullptr; EXP_ARENA_SIZE = 0; }   // never hold both
             tgt = ggml_backend_alloc_buffer(be, need); w.buf = tgt; }  // freed by the caller after the layer
    if (!EXP_STREAM) { cudaStreamCreateWithFlags(&EXP_STREAM, cudaStreamNonBlocking); }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(tgt); size_t go = 0;
    for (int k = 0; k < 3; ++k) { ggml_backend_tensor_alloc(tgt, w.t[kinds[k]], base + go); go += (es[k] * used.size() + al - 1) / al * al; }
    ensure_pin_stage(tot);
    // one full-parallel read batch; each piece is copied to the GPU the moment it lands
    std::vector<Job> jobs; std::vector<uint8_t *> dev; size_t so = 0; size_t from_disk = 0;
    for (int k = 0; k < 3; ++k) { const TI & ti = TIDX.at(p + kinds[k]); size_t ko = 0; for (int q = 0; q < k; ++q) { ko += es[q]; }
        for (size_t u = 0; u < used.size(); ++u) {
            uint8_t * d = (uint8_t *) w.t[kinds[k]]->data + u * es[k];
            if (hitslot[u] >= 0) { if (SPEC_READ && !FUSED) { ggml_backend_tensor_set_async(be, w.t[kinds[k]], PF[hitslot[u]].mem + ko, u * es[k], es[k]); } else if (FUSED && FL) { fl_h2d(FL, d, PF[hitslot[u]].mem + ko, es[k]); } else { cudaMemcpyAsync(d, PF[hitslot[u]].mem + ko, es[k], cudaMemcpyHostToDevice, EXP_STREAM); } continue; }   // prefetched: no disk
            jobs.push_back({ ti.file, ti.off + (uint64_t) used[u] * es[k], es[k], PIN_STAGE_BASE + so + u * es[k] }); dev.push_back(d); from_disk += es[k];
        }
        so += es[k] * used.size(); }
    if (!jobs.empty()) { RD_BUSY++; read_jobs(jobs, dev.data(), EXP_STREAM); RD_BUSY--; }
    B_DISK += (double) from_disk; T_EX += secs(t0);
    const auto t1 = clk::now(); if (!FUSED) { cudaStreamSynchronize(EXP_STREAM); } T_EXUP += secs(t1);   // only the copy tail that did not overlap
    return w;
}

int main(int argc, char ** argv) { std::setvbuf(stdout, nullptr, _IONBF, 0); std::setvbuf(stderr, nullptr, _IONBF, 0);
    namespace fs = std::filesystem;
    std::string prompt_text = "What color is the sky?", prompt_ids_path, base_gguf, siblings_flag;
    std::string shards_dir = R"(models)";
    std::string vocab_path = R"(ptx_treesocab.bin)";
    std::string results_dir = R"(results\server_core)";
    int n_new = 1 << 20, topk = 16, draft_k = 0, experts_keep = 2, sieve_c = 0; std::vector<int32_t> corpus; bool share_experts = false; std::string rmem_path, omem_path, gmem_path; int rmem_layers = 12, vram_keep_mb = 1100; std::string dump_path, drafter_path, trace_path, moe_path, mla_kbt = R"(ptx_trees\mla_kbT.gguf)"; int ctx_len = 384; std::string corpus_path, draft_text_path; int prefetch_slots = 0, prefetch_top = 4, prefetch_d0 = 2, prefetch_d1 = 2;
    std::string sieve_path = R"(ptx_trees\head_sieve.bin)"; double pinned_gb = 5.0; bool siblings = false; std::vector<int32_t> forced;   // --force-ids: feed this path, score it against the full head
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (k == "--prompt") { prompt_text = nx(); } else if (k == "--prompt-ids") { prompt_ids_path = nx(); } else if (k == "--tokens") { n_new = std::atoi(nx().c_str()); }
        else if (k == "--threads") { NTHREADS = std::atoi(nx().c_str()); } else if (k == "--pinned-gb") { pinned_gb = std::atof(nx().c_str()); } else if (k == "--base-gguf") { base_gguf = nx(); }
        else if (k == "--shards-dir") { shards_dir = nx(); } else if (k == "--vocab") { vocab_path = nx(); } else if (k == "--results") { results_dir = nx(); } else if (k == "--siblings") { siblings = true; }
        else if (k == "--topk") { topk = std::atoi(nx().c_str()); }
        else if (k == "--sieve") { sieve_c = std::atoi(nx().c_str()); }                 // head: score cluster centroids, then only the top-C clusters' rows
        else if (k == "--sieve-file") { sieve_path = nx(); }
        else if (k == "--prefetch") { prefetch_slots = std::atoi(nx().c_str()); }        // pinned host slots for experts read ahead (one expert each)
        else if (k == "--prefetch-ahead") { const std::string v = nx(); prefetch_d0 = std::atoi(v.c_str()); const size_t c = v.find('-'); prefetch_d1 = c == std::string::npos ? prefetch_d0 : std::atoi(v.c_str() + c + 1); }   // e.g. 2 or 2-3
        else if (k == "--prefetch-rows") { PF_ROWS = std::atoi(nx().c_str()); }
        else if (k == "--prefetch-top") { prefetch_top = std::atoi(nx().c_str()); }   // experts predicted per row per layer
        else if (k == "--route-memory") { rmem_path = nx(); }   // file of (prev token, token, layer) -> experts labels; loaded at start, saved at the end
        else if (k == "--memory-layers") { rmem_layers = std::atoi(nx().c_str()); }
        else if (k == "--prefill-chunk") { PREFILL_CHUNK = std::max(1, std::atoi(nx().c_str())); } else if (k == "--mla") { MLA = true; } else if (k == "--mla-kbt") { mla_kbt = nx(); }   /* compressed latent KV cache (576 values per position per layer) */
        else if (k == "--ctx") { ctx_len = std::atoi(nx().c_str()); }   // KV cache length (positions); smaller frees VRAM for resident layers
        else if (k == "--drafter") { drafter_path = nx(); }   // look-ahead maps from export_drafter.py
        else if (k == "--trace-layers") { trace_path = nx(); } else if (k == "--dump-moe") { moe_path = nx(); }   /* prefill: row 0 and last row after every layer, fp32 */
        else if (k == "--dump-states") { dump_path = nx(); }   // append per-position states (fp16) for fitting the drafter
        else if (k == "--mid-layer") { MID_L = std::atoi(nx().c_str()); }
        else if (k == "--vram-keep-mb") { vram_keep_mb = std::atoi(nx().c_str()); }   // VRAM left free after the rebalance (-1 = no rebalance)
        else if (k == "--geo-memory") { gmem_path = nx(); }   // state-geometry memory file (all MoE layers)
        else if (k == "--route-depth") { ROUTE_D = std::atoi(nx().c_str()); } else if (k == "--spec-probe") { SPEC_PROBE = std::atoi(nx().c_str()); } else if (k == "--spec-read") { SPEC_READ = std::atoi(nx().c_str()); RJ_SPLIT = SPEC_READ; } else if (k == "--loader-inflight") { LD_INFLIGHT = std::min(3, std::atoi(nx().c_str())); } else if (k == "--hchain") { HCHAIN = std::atoi(nx().c_str()); } else if (k == "--fused") { FUSED = std::atoi(nx().c_str()); } else if (k == "--slots") { FSLOTS = std::max(3, std::min(4, std::atoi(nx().c_str()))); } else if (k == "--attrib") { ATTRIB = nx(); } else if (k == "--attrib-step") { ATTRIB_STEP = std::atoi(nx().c_str()); } else if (k == "--temp") { SAMP_T = (float) std::atof(nx().c_str()); } else if (k == "--top-k") { SAMP_K = std::atoi(nx().c_str()); } else if (k == "--top-p") { SAMP_P = (float) std::atof(nx().c_str()); } else if (k == "--seed") { SAMP_RNG.seed((unsigned) std::atoll(nx().c_str())); } else if (k == "--geo-pred") { GPRED_D = std::atoi(nx().c_str()); } else if (k == "--geo-pred-share") { GPRED_T = (float) std::atof(nx().c_str()); } else if (k == "--geo-pred-k") { GPRED_K = std::max(1, std::atoi(nx().c_str())); } else if (k == "--route-sim") { ROUTE_SIM = (float) std::atof(nx().c_str()); }   /* layers ahead whose expert routes are replayed from the matched state successors (0 = off) */
        else if (k == "--chat") { CHAT = true; }   /* front-end mode: turns over stdin, tokens over stdout, logs to a file */
        else if (k == "--fam-sim") { FAM_SIM = (float) std::atof(nx().c_str()); } else if (k == "--fam-k") { FAM_K = std::max(1, std::atoi(nx().c_str())); } else if (k == "--fam-layer") { FAM_L = std::atoi(nx().c_str()); } else if (k == "--fam-share") { FAM_SHARE = (float) std::atof(nx().c_str()); } else if (k == "--fam-solo") { FAM_SOLO = std::atoi(nx().c_str()); } else if (k == "--run-prompts") { RUNP_PATH = nx(); } else if (k == "--geo-sim-prefill") { GSIM_PRE = (float) std::atof(nx().c_str()); }
        else if (k == "--geo-exact-since") { G_EXACT_SINCE = (uint32_t) std::strtoul(nx().c_str(), nullptr, 10); } else if (k == "--pass-width") { PASS_W = std::atoi(nx().c_str()); } else if (k == "--geo-run") { GEO_RUN = std::atoi(nx().c_str()); } else if (k == "--cold-share") { COLD_SHARE = (float) std::atof(nx().c_str()); } else if (k == "--cold-draft") { COLD_DRAFT = std::max(0, std::atoi(nx().c_str())); } else if (k == "--cold-text") { COLD_TEXT = std::atoi(nx().c_str()) != 0; } else if (k == "--prefill-experts") { PREFILL_EXPERTS = std::max(1, std::min(8, std::atoi(nx().c_str()))); } else if (k == "--renorm-experts") { RENORM_EXPERTS = std::atoi(nx().c_str()) != 0; } else if (k == "--moe-group") { MOE_G = std::max(1, std::atoi(nx().c_str())); }   /* 0 = passes take their natural width (text then depends on draft length) */
        else if (k == "--geo-vote") { GVOTE = std::max(1, std::atoi(nx().c_str())); }   /* k nearest states vote on the branch; 1 = nearest only */
        else if (k == "--geo-next-layer") { GNEXT_L = std::atoi(nx().c_str()); }   // continuations are taken from matches at this layer or deeper
        else if (k == "--geo-sim") { GSIM = (float) std::atof(nx().c_str()); }
        else if (k == "--geo-sim-draft") { GSIM_DRAFT = (float) std::atof(nx().c_str()); }
        else if (k == "--geo-cap") { GCAP = (size_t) std::atoi(nx().c_str()); }
        else if (k == "--geo-max-load") { GMAXLOAD = (size_t) std::atoi(nx().c_str()); }
        else if (k == "--drafter-margin") { DR_MARGIN = (float) std::atof(nx().c_str()); }
        else if (k == "--output-memory") { omem_path = nx(); }   // stored expert outputs per (prev token, token, layer), used in place of the expert pull
        else if (k == "--share-experts") { share_experts = true; }   // multi-row passes: rows reuse experts other rows already load
        else if (k == "--draft-text") { draft_text_path = nx(); }   // any text file, tokenized at startup as draft material
        else if (k == "--draft-corpus") { corpus_path = argv[i + 1 < argc ? i + 1 : i]; FILE * f = std::fopen(nx().c_str(), "rb"); int32_t v; while (f && std::fread(&v, 4, 1, f) == 1) { corpus.push_back(v); } if (f) { std::fclose(f); } }   // R1's saved outputs, -1 between runs
        else if (k == "--experts") { experts_keep = std::max(1, std::min(8, std::atoi(nx().c_str()))); }   // routed experts kept per position (router's highest weights)
        else if (k == "--draft-lookup") { draft_k = std::min(16, std::atoi(nx().c_str())); }
        else if (k == "--draft-text-min") { TEXT_MIN = std::max(2, std::atoi(nx().c_str())); }   /* shortest shared context a text draft may come from */
        else if (k == "--draft-text-cap") { TEXT_CAP = std::atoi(nx().c_str()); }   /* longest text draft; continuation drafts are not capped by this */   // draft up to K tokens from earlier text, confirm them in one pass
        else if (k == "--force-ids") { FILE * f = std::fopen(nx().c_str(), "rb"); int32_t v; while (f && std::fread(&v, 4, 1, f) == 1) { forced.push_back(v); } if (f) { std::fclose(f); } }
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    if (CHAT) {   // the protocol takes the real stdout; everything the runtime prints goes to a log file
        PROTO = _fdopen(_dup(_fileno(stdout)), "w"); std::filesystem::create_directories(results_dir); const std::string lp = results_dir + "\\chat_engine.log";
        if (std::freopen(lp.c_str(), "w", stdout)) { _dup2(_fileno(stdout), _fileno(stderr)); } std::thread(chat_reader).detach();
        n_new = 1 << 20; if (draft_k <= 0) { draft_k = 4; } if (ctx_len < 64) { ctx_len = 64; }
    }
    std::printf("================================================================\nINGOT CORE: ASYNCHRONOUS OVERLAP RUNTIME (DeepSeek-R1 671B, exact ops)\n================================================================\n");
    BPEDetokenizer detok; const bool has_vocab = detok.load(vocab_path);
    std::vector<int32_t> prompt;
    if (!draft_text_path.empty() && has_vocab) { std::ifstream tf(draft_text_path, std::ios::binary); std::string txt((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>()); const auto ids = detok.encode(txt); corpus.insert(corpus.end(), ids.begin(), ids.end()); corpus.push_back(-1); std::printf("[draft] %zu tokens from %s\n", ids.size(), draft_text_path.c_str()); }
    if (!prompt_ids_path.empty()) { FILE * f = std::fopen(prompt_ids_path.c_str(), "rb"); int32_t v; while (std::fread(&v, 4, 1, f) == 1) { prompt.push_back(v); } std::fclose(f); }
    else {
        if (!has_vocab) { std::fprintf(stderr, "vocab not found: %s\n", vocab_path.c_str()); return 1; }
        prompt = { 0, 128803 }; for (int t : detok.encode(prompt_text)) { prompt.push_back(t); } prompt.insert(prompt.end(), { 128804, 128798, 201 });
    }
    for (int k = 1; k <= 4; ++k) { PATHS.push_back(shards_dir + "\\DeepSeek-R1-UD-IQ2_XXS-0000" + std::to_string(k) + "-of-00004.gguf"); }
    if (MLA && std::filesystem::exists(mla_kbt)) { PATHS.push_back(mla_kbt); MLA_SIDE = true; std::printf("[mla] transposed key projection streams from %s\n", mla_kbt.c_str()); }
    if (FUSED && MLA_SIDE) { const std::string trim = (std::filesystem::path(mla_kbt).parent_path() / "mla_trim.gguf").string(); if (std::filesystem::exists(trim)) { PATHS.push_back(trim); TRIM = true; std::printf("[fused] trimmed kv_b (v half) and F16 router weights stream from %s\n", trim.c_str()); } }
    if (!base_gguf.empty()) { PATHS.push_back(base_gguf); std::printf("[step A] always-used weights come from %s where present\n", base_gguf.c_str()); }
    double pinned_cap = pinned_gb * 1e9; { MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); if (GlobalMemoryStatusEx(&ms)) { const double room = (double) ms.ullAvailPhys - 3.5e9; if (room > 0 && pinned_cap > room) { std::fprintf(stderr, "[pinned] %.2f GB asked, %.2f GB free: pinning %.2f GB\n", pinned_gb, ms.ullAvailPhys / 1e9, room / 1e9); pinned_cap = room; } } }
    load_index();
    ggml_backend_t be = ggml_backend_cuda_init(0);
    const int64_t E = 7168, KVP = ctx_len; const float eps = 1e-6f;
    if (FUSED) {   /* fused layer path (patch_fused.py): every format the kernels know must be the one in the shards */
        const char * why = nullptr; if (!MLA_SIDE) { why = "needs --mla with the k_bT sidecar"; } else if (siblings) { why = "no --siblings"; } else if (PASS_W > 8 || PREFILL_CHUNK > 1024) { why = "pass width must be <= 8 and the prefill chunk <= 1024"; }
        for (int L = 0; L < 61 && !why; ++L) { const std::string p = "blk." + std::to_string(L) + "."; auto ty = [&](const char * nm, ggml_type t) { auto it = TIDX.find(p + nm); return it != TIDX.end() && it->second.type == t; };
            if (!(ty("attn_q_a.weight", GGML_TYPE_Q4_K) && ty("attn_q_b.weight", GGML_TYPE_Q4_K) && ty("attn_kv_a_mqa.weight", GGML_TYPE_Q6_K) && (TRIM ? ty("attn_kv_b_v.weight", GGML_TYPE_Q6_K) : ty("attn_kv_b.weight", GGML_TYPE_Q6_K)) && ty("attn_k_bT.weight", GGML_TYPE_Q8_0) && ty("attn_output.weight", GGML_TYPE_Q4_K) && (!TRIM || L < 3 || ty("ffn_gate_inp_f16.weight", GGML_TYPE_F16)))) { why = "attention tensor types"; }
            else if (L < 3) { if (!(ty("ffn_gate.weight", GGML_TYPE_Q4_K) && ty("ffn_up.weight", GGML_TYPE_Q4_K) && ty("ffn_down.weight", GGML_TYPE_Q6_K))) { why = "dense FFN types"; } }
            else if (!(ty("ffn_gate_shexp.weight", GGML_TYPE_Q5_K) && ty("ffn_up_shexp.weight", GGML_TYPE_Q5_K) && ty("ffn_down_shexp.weight", GGML_TYPE_Q6_K) && ty("ffn_gate_exps.weight", GGML_TYPE_IQ2_XXS) && ty("ffn_up_exps.weight", GGML_TYPE_IQ2_XXS) && (ty("ffn_down_exps.weight", GGML_TYPE_Q2_K) || ty("ffn_down_exps.weight", GGML_TYPE_Q3_K)))) { why = "expert types"; } }
        if (!why) { FL = fl_init((int) KVP, (int) GCAP); if (!FL) { why = "context init failed"; } else { fl_wide_init(FL, (int) std::min<int64_t>(KVP, 512), std::max(1, std::min(8, experts_keep))); NSLOT = FSLOTS; for (int q = 0; q < 4; ++q) { cudaEventCreateWithFlags(&LD_EV[q], cudaEventDisableTiming); cudaEventCreate(&LT0[q]); cudaEventCreate(&LT1[q]); } } }
        if (why) { std::fprintf(stderr, "[fused] off: %s\n", why); FUSED = 0; } else { std::fprintf(stderr, "[fused] layers run as direct launches (attention + router + shared expert + spec + scan, then experts); ggml keeps the head\n"); }
    }
    const float freq_base = fbits(0x461c4000), freq_scale = fbits(0x3ccccccd), ext_factor = 1.0f, attn_factor = fbits(0x3f3b035c), beta_fast = 32.0f, beta_slow = 1.0f, kq_scale = fbits(0x3e0a7ab9);
    // ---- KV cache on the GPU for every layer
    ggml_init_params ipk = { ggml_tensor_overhead() * 130, nullptr, true }; ggml_context * kvctx = ggml_init(ipk);
    std::vector<ggml_tensor *> KC(61), VC(61);
    for (int L = 0; L < 61; ++L) { if (MLA) { KC[L] = ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, 576, KVP); VC[L] = FUSED ? ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, 512, KVP) : ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, KVP, 512); } else { KC[L] = ggml_new_tensor_3d(kvctx, GGML_TYPE_F16, 192, 128, KVP); VC[L] = ggml_new_tensor_3d(kvctx, GGML_TYPE_F16, KVP, 128, 128); } }
    ggml_backend_buffer_t kvbuf = ggml_backend_alloc_ctx_tensors(kvctx, be); ggml_backend_buffer_clear(kvbuf, 0);
    ggml_tensor * I512 = nullptr; ggml_tensor * IDS128 = nullptr;   /* identity used to dequantise attn_kv_b on the GPU under --mla; head ids for the per-head value matmul */
    if (MLA) { ggml_init_params ipi = { ggml_tensor_overhead() * 2, nullptr, true }; ggml_context * ictx = ggml_init(ipi); I512 = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, 512, 512); ggml_backend_alloc_ctx_tensors(ictx, be);
        std::vector<float> eye((size_t) 512 * 512, 0.0f); for (int i = 0; i < 512; ++i) { eye[(size_t) i * 512 + i] = 1.0f; } ggml_backend_tensor_set(I512, eye.data(), 0, eye.size() * 4);
        { ggml_init_params ipj = { ggml_tensor_overhead() * 2, nullptr, true }; ggml_context * jctx = ggml_init(ipj); IDS128 = ggml_new_tensor_2d(jctx, GGML_TYPE_I32, 128, 64); ggml_backend_alloc_ctx_tensors(jctx, be); std::vector<int32_t> hid((size_t) 128 * 64); for (int t = 0; t < 64; ++t) { for (int e = 0; e < 128; ++e) { hid[(size_t) t * 128 + e] = e; } } ggml_backend_tensor_set(IDS128, hid.data(), 0, hid.size() * 4); }
        std::printf("[mla] compressed latent KV cache: %lld positions x 61 layers = %.0f MB (expanded form would be %.0f MB)\n", (long long) KVP, KVP * 61 * (576 + 512) * 2 / 1048576.0, KVP * 61 * (192 * 128 + 128 * 128) * 2 / 1048576.0); }
    // ---- head resident on the GPU
    std::vector<std::string> hn = { "output_norm.weight", "output.weight" };
    ggml_init_params iph = { ggml_tensor_overhead() * 4, nullptr, true }; ggml_context * hctx = ggml_init(iph); std::map<std::string, ggml_tensor *> HT;
    for (auto & n : hn) { const TI & ti = TIDX.at(n); int nd = ti.ne[1] > 1 ? 2 : 1; HT[n] = ggml_new_tensor(hctx, ti.type, nd, ti.ne); }
    ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors(hctx, be);
    { size_t tot = 0; for (auto & n : hn) { tot += ggml_nbytes(HT[n]); } ensure_pin_stage(tot); std::vector<Job> jobs; size_t o = 0; std::vector<size_t> offs;
      for (auto & n : hn) { offs.push_back(o); add_jobs(jobs, TIDX.at(n), 0, ggml_nbytes(HT[n]), PIN_STAGE_BASE + o); o += ggml_nbytes(HT[n]); } read_jobs(jobs);
      for (size_t i = 0; i < hn.size(); ++i) { ggml_backend_tensor_set(HT[hn[i]], PIN_STAGE_BASE + offs[i], 0, ggml_nbytes(HT[hn[i]])); } }
    ggml_tensor * CENT = nullptr; std::vector<int32_t> SV_ASG; std::vector<std::vector<int32_t>> SV_MEM;
    if (sieve_c > 0) {
        FILE * f = std::fopen(sieve_path.c_str(), "rb"); int32_t hd[3]; std::fread(hd, 4, 3, f);
        std::vector<float> cv((size_t) hd[0] * hd[1]); std::fread(cv.data(), 4, cv.size(), f); SV_ASG.resize(hd[2]); std::fread(SV_ASG.data(), 4, hd[2], f); std::fclose(f);
        SV_MEM.resize(hd[0]); for (int i = 0; i < hd[2]; ++i) { SV_MEM[SV_ASG[i]].push_back(i); }
        ggml_init_params ips = { ggml_tensor_overhead() * 2, nullptr, true }; ggml_context * sctx = ggml_init(ips);
        CENT = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, hd[1], hd[0]); ggml_backend_alloc_ctx_tensors(sctx, be); ggml_backend_tensor_set(CENT, cv.data(), 0, ggml_nbytes(CENT));
        std::printf("[sieve] %d clusters resident; top-%d clusters scored per position\n", hd[0], sieve_c);
    }
    if (prefetch_slots > 0) {
        size_t slot = 0; std::vector<uint8_t> raw;
        for (int L = 3; L <= 60; ++L) {
            const std::string p = "blk." + std::to_string(L) + "."; size_t es = 0;
            for (const char * k : { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" }) { const TI & ti = TIDX.at(p + k); es += ggml_row_size(ti.type, ti.ne[0]) * (size_t) ti.ne[1]; }
            slot = std::max(slot, es);
            const char * nm[3] = { "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_norm.weight" }; std::vector<float> * dst[3] = { &RW_[L], &RB_[L], &RN_[L] };
            for (int q = 0; q < 3; ++q) {
                const TI & ti = TIDX.at(p + nm[q]); const int64_t rows = ti.ne[1] > 0 ? ti.ne[1] : 1; const size_t nb = ggml_row_size(ti.type, ti.ne[0]) * (size_t) rows;
                raw.resize(nb); std::vector<Job> jb; add_jobs(jb, ti, 0, nb, raw.data()); read_jobs(jb);
                dst[q]->resize((size_t) ti.ne[0] * rows);
                if (ti.type == GGML_TYPE_F32) { std::memcpy(dst[q]->data(), raw.data(), nb); } else { ggml_get_type_traits(ti.type)->to_float(raw.data(), dst[q]->data(), (int64_t) dst[q]->size()); }
            }
        }
        ggml_backend_buffer_t pfb = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), slot * (size_t) prefetch_slots);
        uint8_t * pb = (uint8_t *) ggml_backend_buffer_get_base(pfb); PF.resize(prefetch_slots);
        for (int i = 0; i < prefetch_slots; ++i) { PF[i].mem = pb + (size_t) i * slot; }
        PF_TOP = prefetch_top; PF_D0 = prefetch_d0; PF_D1 = prefetch_d1; PF_ON = true; std::thread(pf_worker).detach();
        std::printf("[prefetch] %d pinned slots x %.1f MB; predicting each row's top-%d experts %d-%d layers ahead\n", prefetch_slots, slot / 1048576.0, PF_TOP, PF_D0, PF_D1);
    }
    // ---- always-used weights per layer: decide a tier for each
    std::vector<LayerW> LW(61); size_t max_layer = 0;
    for (int L = 0; L < 61; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        std::vector<std::string> names = { "attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_kv_b.weight", "attn_output.weight", "ffn_norm.weight" };
        if (L < 3) { names.insert(names.end(), { "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" }); }
        else { names.insert(names.end(), { "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" }); }
        if (MLA_SIDE) { names.push_back("attn_k_bT.weight"); } if (FUSED && TRIM) { for (auto & n : names) { if (n == "attn_kv_b.weight") { n = "attn_kv_b_v.weight"; } else if (n == "ffn_gate_inp.weight") { n = "ffn_gate_inp_f16.weight"; } } }
        for (auto & n : names) { n = p + n; }
        LW[L].names = names; ggml_init_params ip = { ggml_tensor_overhead() * (names.size() + 2), nullptr, true }; LW[L].ctx = ggml_init(ip);
        size_t o = 0;
        for (auto & n : names) { const TI & ti = TIDX.at(n); int nd = ti.ne[2] > 1 ? 3 : ti.ne[1] > 1 ? 2 : 1; ggml_tensor * t = ggml_new_tensor(LW[L].ctx, ti.type, nd, ti.ne); LW[L].t[n] = t; LW[L].offs.push_back(o); o += GGML_PAD(ggml_nbytes(t), 256); }
        LW[L].total = o; if (L >= 3) { max_layer = std::max(max_layer, o); }   /* slots carry MoE layers only; dense layers stay resident */
    }
    max_layer += 4096; /* CUDA pads quantized rows whose width is not a multiple of 512; the last tensor in a slot needs the room */ size_t vfree = 0, vtot = 0; ggml_backend_cuda_get_device_memory(0, &vfree, &vtot);
    const size_t reserve = (size_t) 2200 << 20;                       // experts (prefill), compute buffers, scratch
    size_t vram_budget = vfree > reserve + NSLOT * max_layer ? vfree - reserve - NSLOT * max_layer : 0;
    for (int k = 0; k < NSLOT; ++k) { SCR[k] = ggml_backend_alloc_buffer(be, max_layer); SCRB[k] = (uint8_t *) ggml_backend_buffer_get_base(SCR[k]);
        if (!SPEC_READ) { WST[k] = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), max_layer); WSTB[k] = (uint8_t *) ggml_backend_buffer_get_base(WST[k]); } WST_MAX = max_layer; }
    double pinned_used = 0; size_t vram_used = 0; int n_t[3] = { 0, 0, 0 };
    const auto tl = clk::now();
    for (int L = 0; L < 61; ++L) {
        LayerW & w = LW[L];
        if (L < 3 || vram_used + w.total <= vram_budget) { w.tier = 0; w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, be); vram_used += ggml_backend_buffer_get_size(w.buf); }
        else {
            w.tier = (pinned_used + w.total <= pinned_cap) ? 1 : 2;
            for (size_t i = 0; i < w.names.size(); ++i) { ggml_backend_tensor_alloc(SCR[L % NSLOT], w.t[w.names[i]], SCRB[L % NSLOT] + w.offs[i]); }
            if (w.tier == 1) { w.hostbuf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), w.total); if (!w.hostbuf) { w.tier = 2; } else { w.host = (uint8_t *) ggml_backend_buffer_get_base(w.hostbuf); pinned_used += (double) w.total; } }
        }
        n_t[w.tier]++;
        if (w.tier != 2) {                                            // load once now
            uint8_t * dst = w.tier == 1 ? w.host : nullptr;
            if (w.tier == 0) { ensure_pin_stage(w.total); dst = PIN_STAGE_BASE; }
            std::vector<Job> jobs; for (size_t i = 0; i < w.names.size(); ++i) { add_jobs(jobs, TIDX.at(w.names[i]), 0, ggml_nbytes(w.t[w.names[i]]), dst + w.offs[i]); } read_jobs(jobs);
            if (w.tier == 0) { for (size_t i = 0; i < w.names.size(); ++i) { ggml_backend_tensor_set(w.t[w.names[i]], dst + w.offs[i], 0, ggml_nbytes(w.t[w.names[i]])); } }
        }
    }
    cudaGetLastError(); std::fprintf(stderr, "always-used weights: %d layers in VRAM (%.2f GB), %d in pinned RAM (%.2f GB), %d read from disk per token | head + KV on GPU | setup %.1fs\n", n_t[0], vram_used / 1e9, n_t[1], pinned_used / 1e9, n_t[2], secs(tl));
    // ---- the forward pass for N tokens at positions pos0.. (prefill N = prompt length, decode N = 1)
    std::vector<float> mask((size_t) KVP * (size_t) std::max(64, PREFILL_CHUNK));
    auto loader = [&]() {
        cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        cudaEvent_t ev[4]; for (int q = 0; q < 4; ++q) { cudaEventCreateWithFlags(&ev[q], cudaEventDisableTiming); }
        bool have_read[61] = { false }; std::thread rdt;
        auto disk_read = [&](int L) {                              // tier-2 layer L -> staging slot L&1 (slot's previous copy must be done)
            if (!WSTB[0]) { for (int k = 0; k < NSLOT; ++k) { WST[k] = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), WST_MAX); WSTB[k] = WST[k] ? (uint8_t *) ggml_backend_buffer_get_base(WST[k]) : (uint8_t *) std::malloc(WST_MAX); } std::fprintf(stderr, "[loader] disk-tier staging allocated on first use (%s)\n", WST[0] ? "pinned" : "pageable"); } LayerW & w = LW[L]; cudaEventSynchronize(ev[L % NSLOT]);
            std::vector<Job> jobs; for (size_t i = 0; i < w.names.size(); ++i) { add_jobs(jobs, TIDX.at(w.names[i]), 0, ggml_nbytes(w.t[w.names[i]]), WSTB[L % NSLOT] + w.offs[i]); }
            { static int LDBG = -1; if (LDBG < 0) { LDBG = std::getenv("FUSED_LOADER_DEBUG") ? 1 : 0; } if (LDBG) { std::fprintf(stderr, "[loader] disk_read L%d -> staging slot %d (%zu jobs)\n", L, L % NSLOT, jobs.size()); std::fflush(stderr); } } read_jobs(jobs); B_DISK_W += (double) w.total; have_read[L] = true; { if (std::getenv("FUSED_LOADER_DEBUG")) { std::fprintf(stderr, "[loader] disk_read L%d done\n", L); std::fflush(stderr); } }
        };
        if (SPEC_READ) { rdt = std::thread([&]() { std::vector<int> dl; for (int q = 0; q < 61; ++q) { if (LW[q].tier == 2) { dl.push_back(q); } }
            for (size_t i = 0; i < dl.size(); ++i) { const int Lq = dl[i]; int prev = -1; for (size_t j = 0; j < i; ++j) { if (dl[j] % NSLOT == Lq % NSLOT) { prev = dl[j]; } }   /* the slot's previous disk occupant must be uploaded first */
                if (prev >= 0) { std::unique_lock<std::mutex> lk(MU); CV.wait(lk, [&] { return UPL >= prev; }); } disk_read(Lq); { std::lock_guard<std::mutex> lk(MU); have_read[Lq] = true; } CV.notify_all(); } }); }
        for (int L = 0; L < 61; ++L) {
            LayerW & w = LW[L];
            if (w.tier != 0) {
                { const auto lw0 = clk::now(); std::unique_lock<std::mutex> lk(MU); CV.wait(lk, [&] { return CONS >= L - NSLOT; }); if (FUSED) { LD_WAIT += secs(lw0); } }   // GPU slot L&1 is free once layer L-2 is done
                if (SPEC_READ) { if (w.tier == 2) { std::unique_lock<std::mutex> lk(MU); CV.wait(lk, [&] { return have_read[L]; }); } } else if (w.tier == 2 && !have_read[L]) { disk_read(L); }
                if (!SPEC_READ && L + 1 <= 60 && LW[L + 1].tier == 2 && !have_read[L + 1]) { disk_read(L + 1); }   // next disk layer lands in the other staging slot while we wait for an upload window
                { const auto lw0 = clk::now(); std::unique_lock<std::mutex> lk(MU); CV.wait(lk, [&] { return UPWIN > 0 || CONS >= (SPEC_READ ? L - NSLOT : L - 1); }); if (FUSED) { LD_WAIT += secs(lw0); } }   // upload while the main thread reads experts, or as soon as this layer is next
                const uint8_t * src = w.tier == 2 ? WSTB[L % NSLOT] : w.host; if (std::getenv("FUSED_LOADER_DEBUG")) { std::fprintf(stderr, "[loader] upload L%d tier %d from %s\n", L, w.tier, w.tier == 2 ? "staging" : "pinned"); std::fflush(stderr); }
                if (FUSED) { if (LT_ON[L % NSLOT]) { float ms = 0.f; cudaEventSynchronize(LT1[L % NSLOT]); cudaEventElapsedTime(&ms, LT0[L % NSLOT], LT1[L % NSLOT]); LD_BUSY += ms / 1e3; } cudaEventRecord(LT0[L % NSLOT], st); LT_ON[L % NSLOT] = 1; LD_BYTES += (double) w.total; }
                { static cudaEvent_t LCE[4] = { 0 }; if (!LCE[0]) { for (int q = 0; q < 4; ++q) { cudaEventCreateWithFlags(&LCE[q], cudaEventDisableTiming | cudaEventBlockingSync); } } size_t c = 0; for (size_t i = 0; i < w.names.size(); ++i) { const size_t nb = ggml_nbytes(w.t[w.names[i]]); const size_t CH = FUSED ? nb : UPCHUNK; for (size_t o = 0; o < nb; o += CH) { cudaMemcpyAsync((uint8_t *) w.t[w.names[i]]->data + o, src + w.offs[i] + o, std::min(CH, nb - o), cudaMemcpyHostToDevice, st); if (SPEC_READ && LD_INFLIGHT > 0 && !FUSED) { cudaEventRecord(LCE[c & 3], st); if (c >= (size_t) LD_INFLIGHT) { cudaEventSynchronize(LCE[(c - LD_INFLIGHT) & 3]); } } ++c; } } }
                cudaEventRecord(ev[L % NSLOT], st); if (FUSED) { cudaEventRecord(LT1[L % NSLOT], st); cudaEventRecord(LD_EV[L % NSLOT], st); }
                for (int N = L + 1; N <= std::min(60, L + 2); ++N) {   // while that copy runs: read the next disk layer(s) whose staging slot is free
                    if (!SPEC_READ && LW[N].tier == 2 && !have_read[N]) { disk_read(N); break; }
                }
                if (!FUSED) { cudaEventSynchronize(ev[L % NSLOT]); }   /* fused: the layer stream waits on LD_EV instead of the host */
            }
            { std::lock_guard<std::mutex> lk(MU); UPL = L; } CV.notify_all();
        }
        if (rdt.joinable()) { rdt.join(); } for (int q = 0; q < 4; ++q) { cudaEventDestroy(ev[q]); } cudaStreamDestroy(st);
    };
    std::vector<size_t> UNION; bool SIB = false;
    bool ALLPOS = false;   // true: keep all rows through layer 60 and score every position with the head
    auto forward = [&](const std::vector<int32_t> & toks_in, int pos0, std::vector<float> & logits) {
        std::vector<int32_t> toks = toks_in; const int64_t N_REAL = (int64_t) toks.size(); N_PAD_FROM = N_REAL;   /* fixed-width pass: pad to PASS_W rows */
        if (PASS_W > 0) { const int64_t W = std::min<int64_t>(PASS_W, (int64_t) KVP - pos0); while ((int64_t) toks.size() < W) { toks.push_back(toks.back()); } }
        if ((int64_t) ROW_TOK.size() == N_REAL) { while ((int64_t) ROW_TOK.size() < (int64_t) toks.size()) { ROW_PREV.push_back(ROW_TOK.back()); ROW_TOK.push_back(toks.back()); } }
        GEO_NR.clear(); GEO_NRE.clear(); GEO_NRS.clear();
        PASS_ID++; ROUTE_SENT.clear(); H_DEV = false; HDEV = nullptr; HDEV_OK = false; if (PREV_C2) { ggml_free(PREV_C2); PREV_C2 = nullptr; } ROW_DERIVED.assign(toks.size(), 0); const auto tf0 = clk::now(); const double lk0 = G_TLOOK; UNION.clear(); { std::lock_guard<std::mutex> lk(MU); UPL = -1; CONS = -1; } std::thread ld(loader);
        const int64_t N = (int64_t) toks.size();
        std::vector<float> h((size_t) E * N);
        { const TI & te = TIDX.at("token_embd.weight"); const size_t rb = ggml_row_size(te.type, E); std::vector<uint8_t> row(rb); FILE * f = std::fopen(PATHS[te.file].c_str(), "rb");
          for (int64_t i = 0; i < N; ++i) { _fseeki64(f, (long long) (te.off + (size_t) toks[i] * rb), SEEK_SET); std::fread(row.data(), 1, rb, f); ggml_get_type_traits(te.type)->to_float(row.data(), h.data() + i * E, E); } std::fclose(f); }
        std::vector<int32_t> pos(N); for (int64_t i = 0; i < N; ++i) { pos[i] = SIB ? pos0 : pos0 + (int32_t) i; }
        std::fill(mask.begin(), mask.end(), -INFINITY); for (int64_t i = 0; i < N; ++i) { if (SIB) { for (int j = 0; j < pos0; ++j) { mask[i * KVP + j] = 0.0f; } mask[i * KVP + pos0 + i] = 0.0f; } else { for (int j = 0; j <= pos0 + i; ++j) { mask[i * KVP + j] = 0.0f; } } }
        int32_t last_row = (int32_t) N_PAD_FROM - 1;   /* the last real row */
        int64_t n_in = N;
        auto flw = [&](int Lq) { LayerW & wq = LW[Lq]; const std::string pq = "blk." + std::to_string(Lq) + "."; auto P = [&](const char * nm) -> const void * { auto it = wq.t.find(pq + nm); return it == wq.t.end() ? nullptr : (const void *) it->second->data; };
            FLW f; f.attn_norm = P("attn_norm.weight"); f.q_a = P("attn_q_a.weight"); f.q_a_norm = P("attn_q_a_norm.weight"); f.q_b = P("attn_q_b.weight"); f.kv_a = P("attn_kv_a_mqa.weight"); f.kv_a_norm = P("attn_kv_a_norm.weight"); f.kv_b = P("attn_kv_b_v.weight"); f.kv_b_v = f.kv_b != nullptr; if (!f.kv_b) { f.kv_b = P("attn_kv_b.weight"); } f.k_bT = P("attn_k_bT.weight"); f.attn_out = P("attn_output.weight"); f.ffn_norm = P("ffn_norm.weight"); f.gate_inp = P("ffn_gate_inp_f16.weight"); f.gate_f16 = f.gate_inp != nullptr; if (!f.gate_inp) { f.gate_inp = P("ffn_gate_inp.weight"); } f.probs_b = P("exp_probs_b.bias");
            f.sh_gate = P("ffn_gate_shexp.weight"); f.sh_up = P("ffn_up_shexp.weight"); f.sh_down = P("ffn_down_shexp.weight"); f.d_gate = P("ffn_gate.weight"); f.d_up = P("ffn_up.weight"); f.d_down = P("ffn_down.weight"); return f; };
        for (int L = 0; L < 61; ++L) {
            if (L == 0) { ATTR_PNO++; ATTR_ON = !ATTRIB.empty() && ATTR_PNO == ATTRIB_STEP + 1; ATTR_REC.clear(); } LayerW & w = LW[L]; { const auto tw = clk::now(); std::unique_lock<std::mutex> lk(MU); CV.wait(lk, [&] { return UPL >= L; }); T_W += secs(tw); } auto Wt = [&](const char * nm) { return w.t.at("blk." + std::to_string(L) + "." + nm); };
            if (ATTR_ON) { const int64_t ar = N > 1 ? last_row : 0; ATTR_X.assign(h.begin() + ar * E, h.begin() + (ar + 1) * E); } bool anchor_hit = false; const bool dense = L < 3, last = (L == 60) && !SIB && !ALLPOS; const bool FUSE = FUSED && FL && !SIB && N <= 512; const bool WIDE = FUSE && N > 8; FLOut FO = {}; std::vector<int32_t> WSEL; std::vector<float> WW, WCUR, WFI, WSK, WSC, WCC, WH; if (FUSED && !FUSE) { std::fprintf(stderr, "[fused] a pass of %lld rows (siblings %d) cannot take the ggml path on the fused cache layout\n", (long long) N, (int) SIB); std::exit(9); } const std::string p = "blk." + std::to_string(L) + ".";
            size_t K_scan = 0; ggml_tensor * tsc_attn = nullptr;
            if ((SPEC_READ || FUSE) && GMEM_ON && !SIB && L >= 3 && TPROJ && !dense) { auto & G = GMEM[L]; const size_t K = G.size(); if (K >= 64) { if (GSKV[L] != GVER[L]) {                      // table changed: append the new rows; rebuild only after an eviction or when capacity runs out
                            const bool rebuild = !GSKT[L] || K > (size_t) GSKT[L]->ne[1];
                            if (rebuild) {
                                if (GSKB[L]) { ggml_backend_buffer_free(GSKB[L]); ggml_free(GSKC[L]); }
                                const size_t cap = std::max<size_t>(K + K / 2, 512);
                                ggml_init_params ipk = { ggml_tensor_overhead() * 2, nullptr, true }; GSKC[L] = ggml_init(ipk);
                                GSKT[L] = ggml_new_tensor_2d(GSKC[L], GGML_TYPE_F16, GSK, (int64_t) cap); GSKB[L] = ggml_backend_alloc_ctx_tensors(GSKC[L], be); GSKN[L] = 0; GSKE[L] = GEVICT[L];
                            }
                            const size_t from = GSKN[L];
                            if (K > from) { std::vector<ggml_fp16_t> hk((size_t) GSK * (K - from)); for (size_t j = from; j < K; ++j) { ggml_fp32_to_fp16_row(G[j].sk.data(), hk.data() + (j - from) * GSK, GSK); }
                                ggml_backend_tensor_set(GSKT[L], hk.data(), from * GSK * sizeof(ggml_fp16_t), hk.size() * sizeof(ggml_fp16_t)); }
                            if (!rebuild) { for (uint32_t di : GDIRTY[L]) { if (di < from) { ggml_fp16_t hr[GSK]; ggml_fp32_to_fp16_row(G[di].sk.data(), hr, GSK); ggml_backend_tensor_set(GSKT[L], hr, (size_t) di * GSK * sizeof(ggml_fp16_t), GSK * sizeof(ggml_fp16_t)); } } } GDIRTY[L].clear();
                            GSKN[L] = K; GSKV[L] = GVER[L];
                        } K_scan = K; } }   /* stored-sketch table current before the graph that scans it (patch_scan_merge.py) */
            ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_tensor * skt = nullptr; ggml_tensor * ccat_t = nullptr; ggml_tensor * cur = nullptr; ggml_tensor * ffn_inp = nullptr; ggml_tensor * lout = nullptr; ggml_tensor * sel = nullptr; ggml_tensor * wts = nullptr; ggml_tensor * sel_s = nullptr; ggml_tensor * wts_s = nullptr; const int64_t n = last ? 1 : N;
            if (!FUSE) {
            const auto ta0 = clk::now(); ggml_init_params ip = { ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true }; ctx = ggml_init(ip);
            gf = ggml_new_graph_custom(ctx, 8192, false);
            ggml_tensor * hin = (SPEC_READ && HCHAIN && HDEV_OK && HDEV && HDEV_PASS == PASS_ID && !dense && HDEV->ne[0] == E && HDEV->ne[1] == n_in) ? HDEV : nullptr; const bool hin_dev = hin != nullptr; if (!hin_dev) { hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n_in); ggml_set_input(hin); } if (HCHAIN > 1) { std::fprintf(stderr, "[hc] pass %u L%d dev=%d n_in=%lld hdev=%p ok=%d\n", PASS_ID, L, (int) hin_dev, (long long) n_in, (void *) HDEV, (int) HDEV_OK); }   /* the previous layer output stays on the GPU unless the host changed it (patch_hchain.py) */
            ggml_tensor * tpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N); ggml_set_input(tpos);
            ggml_tensor * tmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, KVP, 64); ggml_set_input(tmask);
            ggml_tensor * tout = last ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1) : nullptr; if (tout) { ggml_set_input(tout); }
            auto norm = [&](ggml_context * c, ggml_tensor * x, ggml_tensor * wn) { return ggml_mul(c, ggml_rms_norm(c, x, eps), wn); };
            cur = norm(ctx, hin, Wt("attn_norm.weight"));
            ggml_tensor * q = ggml_mul_mat(ctx, Wt("attn_q_a.weight"), cur); q = norm(ctx, q, Wt("attn_q_a_norm.weight")); q = ggml_mul_mat(ctx, Wt("attn_q_b.weight"), q);
            q = ggml_reshape_3d(ctx, q, 192, 128, N);
            if (MLA) {   // absorbed MLA: the cache holds the 576-d latent per position; queries move into latent space, values are projected after the weighted sum
                ggml_tensor * kvpe = ggml_mul_mat(ctx, Wt("attn_kv_a_mqa.weight"), cur);
                ggml_tensor * kvc = ggml_view_2d(ctx, kvpe, 512, N, kvpe->nb[1], 0);
                ggml_tensor * kpe = ggml_view_3d(ctx, kvpe, 64, 1, N, kvpe->nb[1], kvpe->nb[1], ggml_row_size(kvpe->type, 512));
                kpe = ggml_rope_ext(ctx, kpe, tpos, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                kvc = norm(ctx, kvc, Wt("attn_kv_a_norm.weight"));
                ggml_tensor * qcur = ggml_rope_ext(ctx, q, tpos, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); qcur = ggml_rope_set_offset(qcur, 128);
                ggml_tensor * deq = MLA_SIDE ? nullptr : ggml_mul_mat(ctx, Wt("attn_kv_b.weight"), I512);   /* [32768 out, 512 in] f32 (only without the sidecar) */
                ggml_tensor * kbT = MLA_SIDE ? Wt("attn_k_bT.weight") : ggml_cont(ctx, ggml_view_3d(ctx, deq, 128, 512, 128, deq->nb[1], 256 * sizeof(float), 0));   /* deq is [32768 out, 512 in]: per head the 128 k_nope outputs are contiguous -> [128 nope, 512 latent, heads] with no transpose */
                ggml_tensor * vb = MLA_SIDE ? nullptr : ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, deq, 128, 512, 128, deq->nb[1], 256 * sizeof(float), 128 * sizeof(float)), 1, 0, 2, 3));   /* the 128 v outputs per head, permuted to [512 latent, 128 v, heads] */
                ggml_tensor * qn = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, qcur, 128, 128, N, qcur->nb[1], qcur->nb[2], 0), 0, 2, 1, 3));     /* [128, N, heads] */
                ggml_tensor * qpe = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, qcur, 64, 128, N, qcur->nb[1], qcur->nb[2], 128 * sizeof(float)), 0, 2, 1, 3));   /* [64, N, heads] */
                ggml_tensor * qcat = ggml_concat(ctx, ggml_mul_mat(ctx, kbT, qn), qpe, 0);   /* [576, N, heads] */
                ggml_tensor * ccat = ggml_concat(ctx, kvc, ggml_reshape_2d(ctx, kpe, 64, N), 0); ggml_set_output(ccat); ccat_t = ccat;   /* [576, N] */
                ggml_tensor * cw = ggml_cpy(ctx, ccat, ggml_view_2d(ctx, KC[L], 576, N, KC[L]->nb[1], (size_t) pos0 * KC[L]->nb[1]));
                ggml_tensor * ctw = ggml_cpy(ctx, ggml_cont(ctx, ggml_transpose(ctx, kvc)), ggml_view_2d(ctx, VC[L], N, 512, VC[L]->nb[1], (size_t) pos0 * VC[L]->nb[0]));
                ggml_build_forward_expand(gf, cw); ggml_build_forward_expand(gf, ctw);
                ggml_tensor * kq = ggml_mul_mat(ctx, KC[L], qcat); ggml_prec_set_acc(kq, GGML_PREC_F32);   /* [KVP, N, heads] */
                kq = ggml_soft_max_ext(ctx, kq, tmask, kq_scale, 0.0f);
                ggml_tensor * o = ggml_mul_mat(ctx, VC[L], kq);   /* [512, N, heads] */
                if (MLA_SIDE) {   /* value projection per head straight from the quantised attn_kv_b: one 256-row slab per head, head = 'expert' */
                    ggml_tensor * as3 = ggml_reshape_3d(ctx, Wt("attn_kv_b.weight"), 512, 256, 128);
                    ggml_tensor * ob = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));   /* [512, heads, N] */
                    ggml_tensor * full = ggml_mul_mat_id(ctx, as3, ob, ggml_view_2d(ctx, IDS128, 128, N, IDS128->nb[1], 0));   /* [256, heads, N]: rows 128..255 are the v outputs */
                    cur = ggml_cont_2d(ctx, ggml_view_3d(ctx, full, 128, 128, N, full->nb[1], full->nb[2], 128 * sizeof(float)), 128 * 128, N);
                } else {
                    ggml_tensor * ov = ggml_mul_mat(ctx, vb, o);      /* [128, N, heads] */
                    cur = ggml_cont_2d(ctx, ggml_permute(ctx, ov, 0, 2, 1, 3), 128 * 128, N);
                }
                cur = ggml_mul_mat(ctx, Wt("attn_output.weight"), cur);
            } else {
            ggml_tensor * kvpe = ggml_mul_mat(ctx, Wt("attn_kv_a_mqa.weight"), cur);
            ggml_tensor * kvc = ggml_view_2d(ctx, kvpe, 512, N, kvpe->nb[1], 0);
            ggml_tensor * kpe = ggml_view_3d(ctx, kvpe, 64, 1, N, kvpe->nb[1], kvpe->nb[1], ggml_row_size(kvpe->type, 512));
            kpe = ggml_rope_ext(ctx, kpe, tpos, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            kvc = norm(ctx, kvc, Wt("attn_kv_a_norm.weight"));
            ggml_tensor * kv = ggml_mul_mat(ctx, Wt("attn_kv_b.weight"), kvc);
            ggml_tensor * knope = ggml_view_3d(ctx, kv, 128, 128, N, ggml_row_size(kv->type, 256), ggml_row_size(kv->type, 256) * 128, 0);
            ggml_tensor * vcur = ggml_cont(ctx, ggml_view_3d(ctx, kv, 128, 128, N, ggml_row_size(kv->type, 256), ggml_row_size(kv->type, 256) * 128, ggml_row_size(kv->type, 128)));
            ggml_tensor * qcur = ggml_rope_ext(ctx, q, tpos, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); qcur = ggml_rope_set_offset(qcur, 128);
            ggml_tensor * kcur = ggml_concat(ctx, knope, ggml_repeat_4d(ctx, kpe, 64, 128, N, 1), 0);
            // write this step's K and V into the cache (F32 -> F16), then attend over the whole cache
            ggml_tensor * kw = ggml_cpy(ctx, kcur, ggml_view_3d(ctx, KC[L], 192, 128, N, KC[L]->nb[1], KC[L]->nb[2], (size_t) pos0 * KC[L]->nb[2]));
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vcur, 1, 2, 0, 3));
            ggml_tensor * vw = ggml_cpy(ctx, vt, ggml_view_3d(ctx, VC[L], N, 128, 128, VC[L]->nb[1], VC[L]->nb[2], (size_t) pos0 * VC[L]->nb[0]));
            ggml_build_forward_expand(gf, kw); ggml_build_forward_expand(gf, vw);
            ggml_tensor * kq = ggml_mul_mat(ctx, ggml_permute(ctx, KC[L], 0, 2, 1, 3), ggml_permute(ctx, qcur, 0, 2, 1, 3)); ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, tmask, kq_scale, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, VC[L], kq);
            cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), 128 * 128, N);
            cur = ggml_mul_mat(ctx, Wt("attn_output.weight"), cur);
            }
            ggml_tensor * inpSA = hin;
            if (last) { cur = ggml_get_rows(ctx, cur, tout); inpSA = ggml_get_rows(ctx, inpSA, tout); }
            ffn_inp = ggml_add(ctx, cur, inpSA);
            cur = norm(ctx, ffn_inp, Wt("ffn_norm.weight"));
                        if (dense) {
                ggml_tensor * up = ggml_mul_mat(ctx, Wt("ffn_up.weight"), cur); ggml_tensor * gate = ggml_mul_mat(ctx, Wt("ffn_gate.weight"), cur);
                lout = ggml_add(ctx, ggml_mul_mat(ctx, Wt("ffn_down.weight"), ggml_swiglu_split(ctx, gate, up)), ffn_inp); ggml_set_output(lout); ggml_build_forward_expand(gf, lout);
            } else {
                ggml_tensor * probs = ggml_sigmoid(ctx, ggml_mul_mat(ctx, Wt("ffn_gate_inp.weight"), cur));
                sel = ggml_argsort_top_k(ctx, ggml_add(ctx, probs, Wt("exp_probs_b.bias")), 8);
                wts = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, 256, n), sel); wts = ggml_reshape_2d(ctx, wts, 8, n);
                ggml_tensor * wsum = ggml_clamp(ctx, ggml_sum_rows(ctx, wts), 6.103515625e-5f, INFINITY);
                wts = ggml_div(ctx, wts, wsum); wts = ggml_reshape_3d(ctx, wts, 1, 8, n); wts = ggml_scale(ctx, wts, 2.5f);
                sel = ggml_cont(ctx, sel); ggml_set_output(sel); ggml_set_output(wts); ggml_set_output(cur); ggml_set_output(ffn_inp);
                ggml_build_forward_expand(gf, ffn_inp); ggml_build_forward_expand(gf, cur); ggml_build_forward_expand(gf, sel); ggml_build_forward_expand(gf, wts);
                if (GMEM_ON && TPROJ) { skt = ggml_mul_mat(ctx, TPROJ, cur); ggml_set_output(skt); ggml_build_forward_expand(gf, skt); }
                if (skt && K_scan >= 64) { ggml_tensor * sxn = ggml_scale(ctx, ggml_rms_norm(ctx, skt, 1e-20f), 1.0f / std::sqrt((float) GSK)); tsc_attn = ggml_mul_mat(ctx, ggml_view_2d(ctx, GSKT[L], GSK, (int64_t) K_scan, GSKT[L]->nb[1], 0), sxn); ggml_set_output(tsc_attn); ggml_build_forward_expand(gf, tsc_attn); }   /* the scan rides this graph (patch_scan_merge.py) */
            }
                        if ((SPEC_PROBE || SPEC_READ) && !SIB && L >= 3 && L < 60 && !dense && sel) {   /* speculative attention+router of layer L+1 on (ffn_inp + shared expert of L), in this graph (patch_spec_merge.py) */
                const auto tsw = clk::now(); { std::unique_lock<std::mutex> lk(MU); UPWIN = 1; CV.notify_all(); CV.wait(lk, [&] { return UPL >= L + 1; }); UPWIN = 0; } T_SPECW += secs(tsw);
                const auto ts0 = clk::now(); LayerW & w1 = LW[L + 1]; auto Wt1 = [&](const char * nm) { return w1.t.at("blk." + std::to_string(L + 1) + "." + nm); };
                const int64_t NS = std::max<int64_t>(1, std::min<int64_t>(N, (int64_t) N_PAD_FROM)); ggml_tensor * curS = ggml_view_2d(ctx, cur, E, NS, cur->nb[1], 0); ggml_tensor * fiS = ggml_view_2d(ctx, ffn_inp, E, NS, ffn_inp->nb[1], 0); ggml_tensor * tposS = ggml_view_1d(ctx, tpos, NS, 0); ggml_tensor * ups_s = ggml_mul_mat(ctx, Wt("ffn_up_shexp.weight"), curS); ggml_tensor * gas_s = ggml_mul_mat(ctx, Wt("ffn_gate_shexp.weight"), curS);
                ggml_tensor * hspec = ggml_add(ctx, fiS, ggml_mul_mat(ctx, Wt("ffn_down_shexp.weight"), ggml_swiglu_split(ctx, gas_s, ups_s)));
                {
                    ggml_tensor * cur = norm(ctx, hspec, Wt1("attn_norm.weight"));
            ggml_tensor * q = ggml_mul_mat(ctx, Wt1("attn_q_a.weight"), cur); q = norm(ctx, q, Wt1("attn_q_a_norm.weight")); q = ggml_mul_mat(ctx, Wt1("attn_q_b.weight"), q);
            q = ggml_reshape_3d(ctx, q, 192, 128, NS);
            if (MLA) {   // absorbed MLA: the cache holds the 576-d latent per position; queries move into latent space, values are projected after the weighted sum
                ggml_tensor * kvpe = ggml_mul_mat(ctx, Wt1("attn_kv_a_mqa.weight"), cur);
                ggml_tensor * kvc = ggml_view_2d(ctx, kvpe, 512, NS, kvpe->nb[1], 0);
                ggml_tensor * kpe = ggml_view_3d(ctx, kvpe, 64, 1, NS, kvpe->nb[1], kvpe->nb[1], ggml_row_size(kvpe->type, 512));
                kpe = ggml_rope_ext(ctx, kpe, tposS, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                kvc = norm(ctx, kvc, Wt1("attn_kv_a_norm.weight"));
                ggml_tensor * qcur = ggml_rope_ext(ctx, q, tposS, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); qcur = ggml_rope_set_offset(qcur, 128);
                ggml_tensor * deq = MLA_SIDE ? nullptr : ggml_mul_mat(ctx, Wt1("attn_kv_b.weight"), I512);   /* [32768 out, 512 in] f32 (only without the sidecar) */
                ggml_tensor * kbT = MLA_SIDE ? Wt1("attn_k_bT.weight") : ggml_cont(ctx, ggml_view_3d(ctx, deq, 128, 512, 128, deq->nb[1], 256 * sizeof(float), 0));   /* deq is [32768 out, 512 in]: per head the 128 k_nope outputs are contiguous -> [128 nope, 512 latent, heads] with no transpose */
                ggml_tensor * vb = MLA_SIDE ? nullptr : ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, deq, 128, 512, 128, deq->nb[1], 256 * sizeof(float), 128 * sizeof(float)), 1, 0, 2, 3));   /* the 128 v outputs per head, permuted to [512 latent, 128 v, heads] */
                ggml_tensor * qn = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, qcur, 128, 128, NS, qcur->nb[1], qcur->nb[2], 0), 0, 2, 1, 3));     /* [128, NS, heads] */
                ggml_tensor * qpe = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, qcur, 64, 128, NS, qcur->nb[1], qcur->nb[2], 128 * sizeof(float)), 0, 2, 1, 3));   /* [64, NS, heads] */
                ggml_tensor * qcat = ggml_concat(ctx, ggml_mul_mat(ctx, kbT, qn), qpe, 0);   /* [576, NS, heads] */
                ggml_tensor * ccat = ggml_concat(ctx, kvc, ggml_reshape_2d(ctx, kpe, 64, NS), 0); ggml_set_output(ccat);    /* [576, NS] */
                ggml_tensor * cw = ggml_cpy(ctx, ccat, ggml_view_2d(ctx, KC[L + 1], 576, NS, KC[L + 1]->nb[1], (size_t) pos0 * KC[L + 1]->nb[1]));
                ggml_tensor * ctw = ggml_cpy(ctx, ggml_cont(ctx, ggml_transpose(ctx, kvc)), ggml_view_2d(ctx, VC[L + 1], NS, 512, VC[L + 1]->nb[1], (size_t) pos0 * VC[L + 1]->nb[0]));
                ggml_build_forward_expand(gf, cw); ggml_build_forward_expand(gf, ctw);
                ggml_tensor * kq = ggml_mul_mat(ctx, KC[L + 1], qcat); ggml_prec_set_acc(kq, GGML_PREC_F32);   /* [KVP, NS, heads] */
                kq = ggml_soft_max_ext(ctx, kq, tmask, kq_scale, 0.0f);
                ggml_tensor * o = ggml_mul_mat(ctx, VC[L + 1], kq);   /* [512, NS, heads] */
                if (MLA_SIDE) {   /* value projection per head straight from the quantised attn_kv_b: one 256-row slab per head, head = 'expert' */
                    ggml_tensor * as3 = ggml_reshape_3d(ctx, Wt1("attn_kv_b.weight"), 512, 256, 128);
                    ggml_tensor * ob = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));   /* [512, heads, NS] */
                    ggml_tensor * full = ggml_mul_mat_id(ctx, as3, ob, ggml_view_2d(ctx, IDS128, 128, NS, IDS128->nb[1], 0));   /* [256, heads, NS]: rows 128..255 are the v outputs */
                    cur = ggml_cont_2d(ctx, ggml_view_3d(ctx, full, 128, 128, NS, full->nb[1], full->nb[2], 128 * sizeof(float)), 128 * 128, NS);
                } else {
                    ggml_tensor * ov = ggml_mul_mat(ctx, vb, o);      /* [128, NS, heads] */
                    cur = ggml_cont_2d(ctx, ggml_permute(ctx, ov, 0, 2, 1, 3), 128 * 128, NS);
                }
                cur = ggml_mul_mat(ctx, Wt1("attn_output.weight"), cur);
            } else {
            ggml_tensor * kvpe = ggml_mul_mat(ctx, Wt1("attn_kv_a_mqa.weight"), cur);
            ggml_tensor * kvc = ggml_view_2d(ctx, kvpe, 512, NS, kvpe->nb[1], 0);
            ggml_tensor * kpe = ggml_view_3d(ctx, kvpe, 64, 1, NS, kvpe->nb[1], kvpe->nb[1], ggml_row_size(kvpe->type, 512));
            kpe = ggml_rope_ext(ctx, kpe, tposS, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            kvc = norm(ctx, kvc, Wt1("attn_kv_a_norm.weight"));
            ggml_tensor * kv = ggml_mul_mat(ctx, Wt1("attn_kv_b.weight"), kvc);
            ggml_tensor * knope = ggml_view_3d(ctx, kv, 128, 128, NS, ggml_row_size(kv->type, 256), ggml_row_size(kv->type, 256) * 128, 0);
            ggml_tensor * vcur = ggml_cont(ctx, ggml_view_3d(ctx, kv, 128, 128, NS, ggml_row_size(kv->type, 256), ggml_row_size(kv->type, 256) * 128, ggml_row_size(kv->type, 128)));
            ggml_tensor * qcur = ggml_rope_ext(ctx, q, tposS, nullptr, 64, 0, 4096, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow); qcur = ggml_rope_set_offset(qcur, 128);
            ggml_tensor * kcur = ggml_concat(ctx, knope, ggml_repeat_4d(ctx, kpe, 64, 128, NS, 1), 0);
            // write this step's K and V into the cache (F32 -> F16), then attend over the whole cache
            ggml_tensor * kw = ggml_cpy(ctx, kcur, ggml_view_3d(ctx, KC[L + 1], 192, 128, NS, KC[L + 1]->nb[1], KC[L + 1]->nb[2], (size_t) pos0 * KC[L + 1]->nb[2]));
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vcur, 1, 2, 0, 3));
            ggml_tensor * vw = ggml_cpy(ctx, vt, ggml_view_3d(ctx, VC[L + 1], NS, 128, 128, VC[L + 1]->nb[1], VC[L + 1]->nb[2], (size_t) pos0 * VC[L + 1]->nb[0]));
            ggml_build_forward_expand(gf, kw); ggml_build_forward_expand(gf, vw);
            ggml_tensor * kq = ggml_mul_mat(ctx, ggml_permute(ctx, KC[L + 1], 0, 2, 1, 3), ggml_permute(ctx, qcur, 0, 2, 1, 3)); ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, tmask, kq_scale, 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, VC[L + 1], kq);
            cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), 128 * 128, NS);
            cur = ggml_mul_mat(ctx, Wt1("attn_output.weight"), cur);
            }
            
                    ggml_tensor * ffn_s = ggml_add(ctx, cur, hspec); ggml_tensor * ncur = norm(ctx, ffn_s, Wt1("ffn_norm.weight"));
                    ggml_tensor * probs_s = ggml_sigmoid(ctx, ggml_mul_mat(ctx, Wt1("ffn_gate_inp.weight"), ncur));
                    ggml_tensor * ss = ggml_argsort_top_k(ctx, ggml_add(ctx, probs_s, Wt1("exp_probs_b.bias")), 8);
                    ggml_tensor * ws = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs_s, 1, 256, NS), ss); ws = ggml_reshape_2d(ctx, ws, 8, NS);
                    ss = ggml_cont(ctx, ss); ggml_set_output(ss); ggml_set_output(ws); ggml_build_forward_expand(gf, ss); ggml_build_forward_expand(gf, ws); sel_s = ss; wts_s = ws;
                }
                SPEC_LAYERS++; T_SPEC += secs(ts0);
            }
            T_AB_BUILD += secs(ta0); const auto tal0 = clk::now(); if (!GA_ATTN) { GA_ATTN = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); } ggml_gallocr_alloc_graph(GA_ATTN, gf); T_AB_ALLOC += secs(tal0); const auto tst0 = clk::now();
            if (!hin_dev) { pset(be, hin, h.data(), 0, ggml_nbytes(hin)); } pset(be, tpos, pos.data(), 0, ggml_nbytes(tpos)); pset(be, tmask, mask.data(), 0, ggml_nbytes(tmask)); T_AB_SET += secs(tst0);
            if (tout) { pset(be, tout, &last_row, 0, 4); }
            T_AB += secs(ta0); { const auto c0 = clk::now(); ggml_backend_graph_compute(be, gf); T_C += secs(c0); }
            } else {   /* fused path: attention + router (+ shared expert) of this layer, the spec pass of the next, the geometry scan, as launches (patch_fused.py) */
                const auto ta0 = clk::now(); FLW fw = flw(L);
                if (LW[L].tier != 0) { fl_wait_event(FL, LD_EV[L % NSLOT]); }
                if (WIDE) {   /* wide pass (cold prefill): the rows in groups of 64 (8 kernel row-blocks per launch set); sel/wts/ncur/ffn_inp (and sketch, scan, latents) gathered per group; no spec pass */
                    static int WG = -1; if (WG < 0) { const char * e = std::getenv("FUSED_WIDE_WG"); WG = e ? std::max(8, std::min(64, std::atoi(e))) : 64; } const int NG = (int) ((N + WG - 1) / WG); if (!dense) { WSEL.assign((size_t) 8 * N, 0); WW.assign((size_t) 8 * N, 0.f); WCUR.resize((size_t) E * N); WFI.resize((size_t) E * N); if (GMEM_ON && TPROJ) { WSK.resize((size_t) GSK * N); if (K_scan >= 64) { WSC.resize(K_scan * (size_t) N); } } } else { WH.resize((size_t) E * N); } if (GKVF) { WCC.resize((size_t) 576 * N); }
                    std::vector<int> ord(NG); for (int g = 0; g < NG; ++g) { ord[g] = g; } if (last) { const int gl = (int) (last_row / WG); ord.erase(ord.begin() + gl); ord.push_back(gl); }   /* the last layer: the target row's group runs last so its router/ffn rows are the ones left in the narrow buffers */
                    for (int gi = 0; gi < NG; ++gi) { const int g = ord[gi], r0 = g * WG, ng = (int) std::min<int64_t>(WG, N - r0); const bool lastg = last && last_row >= r0 && last_row < r0 + ng;
                        fl_attn(FL, &fw, ng, pos0 + r0, pos.data() + r0, KC[L]->data, VC[L]->data, h.data() + (size_t) r0 * E, dense ? 1 : 0, lastg ? (int) (last_row - r0) : -1, (dense || last) ? -1 : r0);
                        int wg = dense ? 32 : (last && !lastg ? 0 : 1); if (GKVF) { wg |= 16; } if (GMEM_ON && TPROJ && !dense && !last) { fl_sketch(FL, (const float *) TPROJ->data, K_scan >= 64 ? GSKT[L]->data : nullptr, (int) K_scan); wg |= 2; if (K_scan >= 64) { wg |= 8; } }
                        FLOut FOg = {}; FLOut & FR = (last && !lastg) ? FOg : FO; fl_fetch(FL, lastg ? 1 : ng, wg, &FR);   /* non-target groups of the last layer fetch into a scratch so FO keeps the target row */
                        if (dense) { std::memcpy(WH.data() + (size_t) r0 * E, FO.h, (size_t) ng * E * 4); }
                        else if (!last) { std::memcpy(WSEL.data() + (size_t) r0 * 8, FO.sel, (size_t) ng * 32); std::memcpy(WW.data() + (size_t) r0 * 8, FO.wts, (size_t) ng * 32); std::memcpy(WCUR.data() + (size_t) r0 * E, FO.ncur, (size_t) ng * E * 4); std::memcpy(WFI.data() + (size_t) r0 * E, FO.fi, (size_t) ng * E * 4);
                            if (FO.skt) { std::memcpy(WSK.data() + (size_t) r0 * GSK, FO.skt, (size_t) ng * GSK * 4); } if (FO.sc && K_scan >= 64) { std::memcpy(WSC.data() + (size_t) r0 * K_scan, FO.sc, (size_t) ng * K_scan * 4); } }
                        if (GKVF && FR.ccat) { std::memcpy(WCC.data() + (size_t) r0 * 576, FR.ccat, (size_t) ng * 576 * 4); } }
                    T_FA += secs(ta0); T_AB += secs(ta0);
                } else {
                fl_attn(FL, &fw, (int) N, pos0, pos.data(), KC[L]->data, VC[L]->data, H_DEV ? nullptr : h.data(), dense ? 1 : 0, last ? (int) last_row : -1, -1);
                int what = dense ? 32 : 1; if (GKVF) { what |= 16; }
                if ((SPEC_PROBE || SPEC_READ) && !SIB && L >= 3 && L < 60 && !dense) {
                    const auto tsw = clk::now(); { std::unique_lock<std::mutex> lk(MU); UPWIN = 1; CV.notify_all(); CV.wait(lk, [&] { return UPL >= L + 1; }); UPWIN = 0; } T_SPECW += secs(tsw);
                    if (LW[L + 1].tier != 0) { fl_wait_event(FL, LD_EV[(L + 1) % NSLOT]); } FLW fw1 = flw(L + 1); fl_spec(FL, &fw1, KC[L + 1]->data, VC[L + 1]->data); SPEC_LAYERS++; what |= 4;
                }
                if (GMEM_ON && TPROJ && !dense) { fl_sketch(FL, (const float *) TPROJ->data, K_scan >= 64 ? GSKT[L]->data : nullptr, (int) K_scan); what |= 2; if (K_scan >= 64) { what |= 8; } }
                fl_fetch(FL, (int) n, what, &FO); T_FA += secs(ta0); T_AB += secs(ta0);
                }
            }
            if ((ccat_t || FO.ccat || WIDE) && GKVF) { if (L == 0) { KVLAT.assign((size_t) N * 61 * 576, 0); } std::vector<float> cl((size_t) 576 * N); if (WIDE) { std::memcpy(cl.data(), WCC.data(), cl.size() * 4); } else if (FO.ccat) { std::memcpy(cl.data(), FO.ccat, cl.size() * 4); } else { pget(be, ccat_t, cl.data(), 0, cl.size() * 4); pget_sync(be); } for (int64_t i = 0; i < N; ++i) { ggml_fp32_to_fp16_row(cl.data() + i * 576, KVLAT.data() + ((size_t) i * 61 + L) * 576, 576); } }   /* this pass's latents, kept for the layer-60 store */
            const auto tg1 = clk::now(); if (dense) { h.resize((size_t) E * n); if (WIDE) { h = WH; H_DEV = false; } else if (FUSE) { std::memcpy(h.data(), FO.h, h.size() * 4); H_DEV = true; } else { pget(be, lout, h.data(), 0, h.size() * 4); pget_sync(be); } }
            else {
                std::vector<int32_t> selh((size_t) 8 * n); std::vector<float> wh((size_t) 8 * n), curh((size_t) E * n), fih((size_t) E * n); std::vector<float> skh; std::vector<int32_t> ssel; std::vector<float> ssw; std::vector<float> SCA;
                if (WIDE && !last) { selh.swap(WSEL); wh.swap(WW); curh.swap(WCUR); fih.swap(WFI); if (!WSK.empty()) { skh.swap(WSK); } if (!WSC.empty()) { SCA.swap(WSC); } T_GET += secs(tg1); }
                else if (FUSE) {   /* the launches' outputs are already in pinned host memory (patch_fused.py) */
                    std::memcpy(selh.data(), FO.sel, selh.size() * 4); std::memcpy(wh.data(), FO.wts, wh.size() * 4); std::memcpy(curh.data(), FO.ncur, curh.size() * 4); std::memcpy(fih.data(), FO.fi, fih.size() * 4);
                    if (FO.skt) { skh.resize((size_t) GSK * n); std::memcpy(skh.data(), FO.skt, skh.size() * 4); }
                    if (FO.sel_s) { const int64_t NSf = std::max<int64_t>(1, std::min<int64_t>(N, (int64_t) N_PAD_FROM)); ssel.resize((size_t) 8 * NSf); ssw.resize((size_t) 8 * NSf); std::memcpy(ssel.data(), FO.sel_s, ssel.size() * 4); std::memcpy(ssw.data(), FO.wts_s, ssw.size() * 4); }
                    if (FO.sc && K_scan >= 64) { SCA.resize(K_scan * (size_t) n); std::memcpy(SCA.data(), FO.sc, SCA.size() * 4); }
                    T_GET += secs(tg1);
                } else {
                pget(be, sel, selh.data(), 0, selh.size() * 4); pget(be, wts, wh.data(), 0, wh.size() * 4);
                pget(be, cur, curh.data(), 0, curh.size() * 4); pget(be, ffn_inp, fih.data(), 0, fih.size() * 4);
                if (skt) { skh.resize((size_t) GSK * n); pget(be, skt, skh.data(), 0, skh.size() * 4); } if (sel_s) { ssel.resize((size_t) ggml_nelements(sel_s)); ssw.resize((size_t) ggml_nelements(wts_s)); pget(be, sel_s, ssel.data(), 0, ssel.size() * 4); pget(be, wts_s, ssw.data(), 0, ssw.size() * 4); } if (tsc_attn) { SCA.resize(K_scan * (size_t) n); pget(be, tsc_attn, SCA.data(), 0, SCA.size() * 4); } pget_sync(be); T_GET += secs(tg1);
                }
                if ((SPEC_PROBE || SPEC_READ) && SPEC_L == L && !dense && n > 0) {   /* score last layer's speculative route for this layer against the real one (row 0) */
                    int ro[8], so[8]; for (int k = 0; k < 8; ++k) { ro[k] = k; so[k] = k; }
                    std::sort(ro, ro + 8, [&](int a, int b) { return wh[a] > wh[b]; }); std::sort(so, so + 8, [&](int a, int b) { return SPEC_W[a] > SPEC_W[b]; });
                    for (int k = 0; k < 2; ++k) { SPEC_N++; const int32_t real = selh[ro[k]]; bool in2 = false, in4 = false; for (int j = 0; j < 4; ++j) { if (SPEC_SEL[so[j]] == real) { in4 = true; if (j < 2) { in2 = true; } } } SPEC_H2 += in2; SPEC_H4 += in4; }
                }
                if (sel_s || FO.sel_s) {   /* the merged graph's speculative route for layer L+1 (patch_spec_merge.py) */
                    std::copy(ssel.begin(), ssel.begin() + 8, SPEC_SEL); std::copy(ssw.begin(), ssw.begin() + 8, SPEC_W); SPEC_L = L + 1;
                    if (SPEC_READ && PF_ON) {   /* every real row's speculative top-KE experts of layer L+1 go to the prefetcher now */
                        bool sent = false; { std::lock_guard<std::mutex> lk(PFM); const int ke = std::min(experts_keep, 8);
                          for (int64_t i = 0; i < N && i < (int64_t) N_PAD_FROM; ++i) { int o[8]; for (int k = 0; k < 8; ++k) { o[k] = k; } std::sort(o, o + 8, [&](int a, int b) { return ssw[i * 8 + a] > ssw[i * 8 + b]; });
                              for (int k = 0; k < ke; ++k) { const int e = ssel[i * 8 + o[k]]; if (!ROUTE_SENT.insert(((L + 1) << 9) | e).second) { continue; } PFX.push_back({ L + 1, e | (3 << 16) }); SPEC_Q++; sent = true; } } }
                        if (sent) { PFCV.notify_all(); } }
                }
                if (RMEM_ON && !SIB && L <= RMEM_MAXL && (int64_t) ROW_TOK.size() == n) {   // label: this token pair at this layer went to these experts (strongest first)
                    for (int64_t i = 0; i < n; ++i) {
                        int ord[8]; for (int k = 0; k < 8; ++k) { ord[k] = k; }
                        std::sort(ord, ord + 8, [&](int a, int b) { return wh[i * 8 + a] > wh[i * 8 + b]; });
                        std::array<int16_t, 8> a; for (int k = 0; k < 8; ++k) { a[k] = (int16_t) selh[i * 8 + ord[k]]; }
                        if (i < N_PAD_FROM) { RMEM[rkey(ROW_PREV[i], ROW_TOK[i], L)] = a; }
                    }
                }
                const int KE = experts_keep;
                if (KE < 8) {                                   // keep only each row's KE highest-weight experts: compact to KE slots per row
                    std::vector<int32_t> sk((size_t) KE * n); std::vector<float> wk((size_t) KE * n);
                    std::vector<std::array<int, 8>> ords(n);
                    for (int64_t i = 0; i < n; ++i) { auto & ord = ords[i]; for (int k = 0; k < 8; ++k) { ord[k] = k; }
                        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return wh[i * 8 + a] > wh[i * 8 + b]; }); }
                    if (share_experts && n > 1 && PASS_W <= 0) {          // each row keeps its own top-2; its other slots prefer experts another row already loads (a cross-row dependence: off under fixed-width passes so a row's experts never depend on what was drafted beside it)
                        const int own = std::min(2, KE); std::set<int32_t> U;
                        for (int64_t i = 0; i < n && i < N_PAD_FROM; ++i) { for (int k = 0; k < own; ++k) { U.insert(selh[i * 8 + ords[i][k]]); } }
                        for (int64_t i = 0; i < n; ++i) {
                            std::vector<int> pick; for (int k = 0; k < own; ++k) { pick.push_back(ords[i][k]); }
                            for (int k = own; k < 8 && (int) pick.size() < KE; ++k) { if (U.count(selh[i * 8 + ords[i][k]])) { pick.push_back(ords[i][k]); } }
                            for (int k = own; k < 8 && (int) pick.size() < KE; ++k) { if (std::find(pick.begin(), pick.end(), ords[i][k]) == pick.end()) { pick.push_back(ords[i][k]); U.insert(selh[i * 8 + ords[i][k]]); } }
                            for (int k = 0; k < KE; ++k) { sk[i * KE + k] = selh[i * 8 + pick[k]]; wk[i * KE + k] = wh[i * 8 + pick[k]]; }
                        }
                    } else {
                        for (int64_t i = 0; i < n; ++i) { for (int k = 0; k < KE; ++k) { sk[i * KE + k] = selh[i * 8 + ords[i][k]]; wk[i * KE + k] = wh[i * 8 + ords[i][k]]; } }
                    }
                    if (RENORM_EXPERTS) { for (int64_t i = 0; i < n; ++i) { float sw = 0.0f; for (int k = 0; k < KE; ++k) { sw += wk[i * KE + k]; } if (sw > 1e-6f) { const float f = 2.5f / sw; for (int k = 0; k < KE; ++k) { wk[i * KE + k] *= f; } } } }   /* kept experts carry the full routed scale */
                    selh = sk; wh = wk;
                }
                if (MOEDF && !SIB && n > 0) { const int32_t hd[2] = { L, KE }; std::fwrite(hd, 4, 2, MOEDF); std::fwrite(selh.data(), 4, KE, MOEDF); std::fwrite(wh.data(), 4, KE, MOEDF); std::fwrite(curh.data(), 4, E, MOEDF); }
                for (int64_t i = N_PAD_FROM; i < n; ++i) { for (int k = 0; k < KE; ++k) { selh[i * KE + k] = selh[k]; wh[i * KE + k] = 0.0f; } }   /* pad rows: row 0's experts at weight 0 */
                std::vector<int32_t> used = selh; std::sort(used.begin(), used.end()); used.erase(std::unique(used.begin(), used.end()), used.end()); UNION.push_back(used.size());
                std::vector<int32_t> remap(256, -1); for (size_t u = 0; u < used.size(); ++u) { remap[used[u]] = (int32_t) u; }
                std::vector<int32_t> idsc(selh.size()); for (size_t i = 0; i < selh.size(); ++i) { idsc[i] = remap[selh[i]]; }
                const bool omem_rows = OMEM_ON && !SIB && L <= RMEM_MAXL && (int64_t) ROW_TOK.size() == n; OMEM_LAYERS_SEEN += omem_rows;
                bool skipall = omem_rows; std::vector<int> ghit; std::vector<float> SXK;
                if (GMEM_ON && !SIB && L >= 3) {                   // nearest stored state at this layer; reuse its expert output when close enough
                    auto & G = GMEM[L]; ghit.assign(n, -1); skipall = !G.empty(); const auto tg0 = clk::now();
                    std::vector<float> xu(E), SX((size_t) GSK * n), SC; const size_t K = G.size();
                    for (int64_t i = 0; i < n; ++i) {
                        const float * x = curh.data() + i * E; double nn = 0; for (int d = 0; d < E; ++d) { nn += (double) x[d] * x[d]; } const float inv = 1.0f / (float) std::sqrt(nn + 1e-12);
                        if (!skh.empty()) { double sn = 0; for (int q = 0; q < GSK; ++q) { sn += (double) skh[(size_t) i * GSK + q] * skh[(size_t) i * GSK + q]; } const float si = 1.0f / (float) std::sqrt(sn + 1e-20); for (int q = 0; q < GSK; ++q) { SX[(size_t) i * GSK + q] = skh[(size_t) i * GSK + q] * si; } }
                        else { for (int d = 0; d < E; ++d) { xu[d] = x[d] * inv; } gsketch(xu.data(), SX.data() + (size_t) i * GSK); }
                    }
                    SXK = SX;
                    if (K >= 64) {                                     // GPU scan: this layer's stored sketches (fp16, resident) x all rows, one matmul
                        if (GSKV[L] != GVER[L]) {                      // table changed: append the new rows; rebuild only after an eviction or when capacity runs out
                            const bool rebuild = !GSKT[L] || K > (size_t) GSKT[L]->ne[1];
                            if (rebuild) {
                                if (GSKB[L]) { ggml_backend_buffer_free(GSKB[L]); ggml_free(GSKC[L]); }
                                const size_t cap = std::max<size_t>(K + K / 2, 512);
                                ggml_init_params ipk = { ggml_tensor_overhead() * 2, nullptr, true }; GSKC[L] = ggml_init(ipk);
                                GSKT[L] = ggml_new_tensor_2d(GSKC[L], GGML_TYPE_F16, GSK, (int64_t) cap); GSKB[L] = ggml_backend_alloc_ctx_tensors(GSKC[L], be); GSKN[L] = 0; GSKE[L] = GEVICT[L];
                            }
                            const size_t from = GSKN[L];
                            if (K > from) { std::vector<ggml_fp16_t> hk((size_t) GSK * (K - from)); for (size_t j = from; j < K; ++j) { ggml_fp32_to_fp16_row(G[j].sk.data(), hk.data() + (j - from) * GSK, GSK); }
                                ggml_backend_tensor_set(GSKT[L], hk.data(), from * GSK * sizeof(ggml_fp16_t), hk.size() * sizeof(ggml_fp16_t)); }
                            if (!rebuild) { for (uint32_t di : GDIRTY[L]) { if (di < from) { ggml_fp16_t hr[GSK]; ggml_fp32_to_fp16_row(G[di].sk.data(), hr, GSK); ggml_backend_tensor_set(GSKT[L], hr, (size_t) di * GSK * sizeof(ggml_fp16_t), GSK * sizeof(ggml_fp16_t)); } } } GDIRTY[L].clear();
                            GSKN[L] = K; GSKV[L] = GVER[L];
                        }
                        if ((tsc_attn || !SCA.empty()) && K == K_scan) { SC.resize(K * n); std::copy(SCA.begin(), SCA.end(), SC.begin()); } else {
                        ggml_init_params ipq = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true }; ggml_context * cq = ggml_init(ipq);
                        ggml_tensor * tsx = ggml_new_tensor_2d(cq, GGML_TYPE_F32, GSK, n); ggml_set_input(tsx);
                        ggml_tensor * tsc = ggml_mul_mat(cq, ggml_view_2d(cq, GSKT[L], GSK, (int64_t) K, GSKT[L]->nb[1], 0), tsx); ggml_set_output(tsc);
                        ggml_cgraph * gq = ggml_new_graph(cq); ggml_build_forward_expand(gq, tsc);
                        if (!GA_SK) { GA_SK = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); } ggml_gallocr_alloc_graph(GA_SK, gq);
                        pset(be, tsx, SX.data(), 0, ggml_nbytes(tsx)); ggml_backend_graph_compute(be, gq);
                        SC.resize(K * n); pget(be, tsc, SC.data(), 0, SC.size() * 4); pget_sync(be); ggml_free(cq); }
                    }
                    for (int64_t i = 0; i < n; ++i) {
                        float best = -2.0f; int bi = -1; if (i >= N_PAD_FROM) { continue; }   /* pad row: no lookup, no store, no vote */
                        if (!SC.empty()) { const float * row = SC.data() + (size_t) i * K; for (size_t j = 0; j < K; ++j) { if (row[j] > best && (G[j].used & G_EXACT)) { best = row[j]; bi = (int) j; } } }   /* only exact-origin states stand in for a read */
                        else { const float * sx = SX.data() + (size_t) i * GSK; for (size_t j = 0; j < K; ++j) { const float * k = G[j].sk.data(); float a = 0; for (int q = 0; q < GSK; ++q) { a += k[q] * sx[q]; } if (a > best && (G[j].used & G_EXACT)) { best = a; bi = (int) j; } } }   // sketch dot ~ cosine (+-0.02 at 256 dims)
                        const float thr = ((i > 0 && n > 1) ? (ALLPOS ? GSIM_DRAFT : (GSIM_PRE > 0.0f ? GSIM_PRE : GSIM)) : GSIM);   /* draft rows at the draft threshold; prompt rows exact */
                        if (GVOTE > 1 && !SC.empty() && best >= thr) {   // family vote: the k nearest members' continuations choose the branch; substitute from that branch's best member
                            const float * row = SC.data() + (size_t) i * K; std::vector<std::pair<float, int>> cand;
                            for (size_t j = 0; j < K; ++j) { if (row[j] >= thr && (G[j].used & G_EXACT)) { cand.push_back({ row[j], (int) j }); } }
                            if (cand.size() > 1) {
                                const size_t kk = std::min<size_t>((size_t) GVOTE, cand.size());
                                std::partial_sort(cand.begin(), cand.begin() + kk, cand.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; }); cand.resize(kk);
                                std::map<int, float> vote; for (auto & c : cand) { const int32_t t = G[c.second].nxt[0]; if (t >= 0) { vote[t] += c.first; } }
                                if (!vote.empty()) { int wt = -1; float ws = -1.0f; for (auto & kv : vote) { if (kv.second > ws) { ws = kv.second; wt = kv.first; } }
                                    for (auto & c : cand) { if (G[c.second].nxt[0] == wt) { best = c.first; bi = c.second; break; } } }
                            }
                        }
                        if (L >= FAM_L && !SC.empty()) {   /* the row's family: k nearest stored states with continuations (deepest layer wins); voted on after the head */
                            const float * rw = SC.data() + (size_t) i * K; std::vector<std::pair<float, int>> fc; for (size_t j = 0; j < K; ++j) { if (rw[j] >= FAM_SIM && G[j].nxt[0] >= 0) { fc.push_back({ rw[j], (int) j }); } }
                            if (!fc.empty()) { const size_t kk = std::min<size_t>((size_t) FAM_K, fc.size()); std::partial_sort(fc.begin(), fc.begin() + kk, fc.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; }); fc.resize(kk);
                                if ((int64_t) GEO_FAM.size() != n) { GEO_FAM.assign((size_t) n, std::vector<FamM>()); GEO_FAML.assign((size_t) n, -1); } GEO_FAM[i].clear();
                                for (auto & c : fc) { FamM m; m.w = c.first; m.crun = G[c.second].crun; std::copy(G[c.second].nxt, G[c.second].nxt + 16, m.nxt.begin()); GEO_FAM[i].push_back(m); } GEO_FAML[i] = L; } }
                        if (PF_ON && GPRED_D > 0 && i == 0 && L < 60 && !SC.empty()) {   /* geometry route prediction (patch_geo_pred.py): the anchor row's nearest stored states vote on the experts of the layers ahead */
                            const float * rw = SC.data() + (size_t) i * K; std::vector<std::pair<float, int>> nb; for (size_t j = 0; j < K; ++j) { if (rw[j] > 0.0f && rw[j] < 0.9999f) { nb.push_back({ rw[j], (int) j }); } }
                            if (!nb.empty()) { const size_t kk = std::min<size_t>((size_t) GPRED_K, nb.size()); std::partial_sort(nb.begin(), nb.begin() + kk, nb.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; }); nb.resize(kk);
                                bool sent = false; { std::lock_guard<std::mutex> lk(PFM);
                                for (int L2 = std::max(3, L + 1); L2 <= std::min(60, L + GPRED_D); ++L2) { if (ROUTE_COVER[L2] == PASS_ID) { continue; } std::map<int, float> vote; float tot = 0.0f;
                                    for (auto & c : nb) { auto it = GIDX[L2].find(gkey(G[c.second].runid, G[c.second].pos)); if (it == GIDX[L2].end() || it->second >= GMEM[L2].size()) { continue; } const GEntry & se = GMEM[L2][it->second];
                                        for (int r = 0; r < 2 && se.route[r] >= 0; ++r) { const float v = c.first / (float) (1 + r); vote[se.route[r]] += v; tot += v; } }
                                    if (tot <= 0.0f) { continue; } GPRED_STEPS++; ROUTE_COVER[L2] = PASS_ID;
                                    for (auto & kv : vote) { if (kv.second / tot < GPRED_T) { continue; } GPRED_N++; if (!ROUTE_SENT.insert((L2 << 9) | kv.first).second) { continue; } PFX.push_back({ L2, kv.first | (3 << 16) }); sent = true; } } }
                                if (sent) { PFCV.notify_all(); } } }
                        G_ROWS++; if (best >= thr) { ghit[i] = bi; G[bi].used = G_CLOCK | (G[bi].used & G_EXACT); G_HITROWS++; if (i < (int64_t) ROW_DERIVED.size() && best < 0.9999f) { ROW_DERIVED[i] = 1; }   /* an identical state's output leaves the row exact */ if (G[bi].route[0] < 0) { for (int k = 0; k < KE && k < 8; ++k) { G[bi].route[k] = selh[i * KE + k]; } }   /* an old state without a route takes this row's */
                            if (PF_ON && i == 0 && ROUTE_D > 0 && L < 60 && best < ROUTE_SIM) {   // the anchor row sits on a stored state: its successors' expert routes feed the prefetcher now
                                const uint64_t ck = gkey(G[bi].runid, G[bi].pos); bool sent = false;
                                { std::lock_guard<std::mutex> lk(PFM);
                                  for (int L2 = L + 1; L2 <= std::min(60, L + ROUTE_D); ++L2) { auto it = GIDX[L2].find(ck); if (it == GIDX[L2].end() || it->second >= GMEM[L2].size()) { continue; }
                                      const GEntry & se = GMEM[L2][it->second]; bool any = false;
                                      for (int k = 0; k < 8 && k < PF_TOP && se.route[k] >= 0; ++k) { any = true; if (!ROUTE_SENT.insert((L2 << 9) | se.route[k]).second) { continue; } PFX.push_back({ L2, se.route[k] | (1 << 16) }); ROUTE_Q++; sent = true; if (CHAIN_TRACE > 0) { CHAIN_TRACE--; std::fprintf(stderr, "  [chain] pass %u: hit at L%d (entry run %u pos %u) -> L%d expert %d queued\n", PASS_ID, L, G[bi].runid, G[bi].pos, L2, se.route[k]); } }
                                      if (any) { ROUTE_COVER[L2] = PASS_ID; } } }
                                if (sent) { PFCV.notify_all(); } }
                            if (G[bi].nxt[0] < 0 && best >= GSIM) { G[bi].crun = G_RUNID; G[bi].cpos = G_POS + (uint32_t) i; std::fill(G[bi].nxt, G[bi].nxt + 16, -1); G_ADOPT++; }   // a stored state with no continuation, hit tightly: what follows in this run becomes its continuation
                            if (L >= GNEXT_L && G[bi].nxt[0] >= 0) { if ((int64_t) GEO_NR.size() != n) { GEO_NR.assign(n, { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }); } std::copy(G[bi].nxt, G[bi].nxt + 16, GEO_NR[i].begin()); if ((int64_t) GEO_NRE.size() != n) { GEO_NRE.assign((size_t) n, { 0u, 0u }); GEO_NRS.assign((size_t) n, 0.0f); } GEO_NRE[i] = { G[bi].crun, G[bi].cpos }; GEO_NRS[i] = best; } } else { skipall = false; }
                    }
                    for (int64_t i = 0; i < n && i < (int64_t) G_MISSROW.size(); ++i) { if (ghit[i] < 0) { G_MISSROW[i]++; } } G_TLOOK += secs(tg0); anchor_hit = !ghit.empty() && ghit[0] >= 0;
                    if (!skipall) {                                // per-row: matched rows drop their expert read; only the other rows' experts are loaded
                        int64_t anchor = -1; for (int64_t i = 0; i < n; ++i) { if (ghit[i] < 0 && i < N_PAD_FROM) { anchor = i; break; } }   // a row that still computes: matched rows borrow its (distinct) expert list at weight 0
                        for (int64_t i = 0; i < n; ++i) { if (ghit[i] >= 0 || i >= N_PAD_FROM) { for (int k = 0; k < KE; ++k) { selh[i * KE + k] = selh[anchor * KE + k]; wh[i * KE + k] = 0.0f; } if (ghit[i] >= 0) { G_ROWSKIP++; } } }
                        used = selh; std::sort(used.begin(), used.end()); used.erase(std::unique(used.begin(), used.end()), used.end());
                        std::fill(remap.begin(), remap.end(), -1); for (size_t u = 0; u < used.size(); ++u) { remap[used[u]] = (int32_t) u; }
                        for (size_t i = 0; i < selh.size(); ++i) { idsc[i] = remap[selh[i]]; }
                    }
                }
                if (skipall && !GMEM_ON) { for (int64_t i = 0; i < n; ++i) { if (!OMEM.count(rkey(ROW_PREV[i], ROW_TOK[i], L))) { skipall = false; break; } } }
                const auto tk0 = clk::now(); if (skipall) {        // every row's token pair has this layer's expert output stored: no expert bytes are read
                    OMEM_SKIPS++;
                    std::vector<float> mo((size_t) E * n);
                    if (GMEM_ON) { for (int64_t i = 0; i < n; ++i) { if (ghit[i] >= 0) { gout_read(GMEM[L][ghit[i]].off, mo.data() + i * E); if (L == 60 && GKVF && i < N_PAD_FROM && GMEM[L][ghit[i]].kvoff == ~0ull && KVLAT.size() >= (size_t) (i + 1) * 61 * 576) { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GKVF, 0, SEEK_END); GMEM[L][ghit[i]].kvoff = GKV_END; std::fwrite(KVLAT.data() + (size_t) i * 61 * 576, 2, 61 * 576, GKVF); GKV_END += (uint64_t) 61 * 576 * 2; } } } }   /* pad rows keep a zero expert output; a replayed walk gains its latents here too */
                    else { for (int64_t i = 0; i < n; ++i) { const auto & v = OMEM[rkey(ROW_PREV[i], ROW_TOK[i], L)]; std::copy(v.begin(), v.end(), mo.begin() + i * E); } }
                    if (WIDE && !last) { const auto tf0 = clk::now(); fl_wide_skip(FL, mo.data(), (int) n, &FO); h.resize((size_t) E * n); std::memcpy(h.data(), FO.h, h.size() * 4); H_DEV = false; HDEV_OK = false; T_FM += secs(tf0); T_C += secs(tf0); }
                    else if (FUSE) { const auto tf0 = clk::now(); fl_skip(FL, mo.data(), (int) n, &FO); h.resize((size_t) E * n); std::memcpy(h.data(), FO.h, h.size() * 4); H_DEV = true; HDEV_OK = false; T_FM += secs(tf0); T_C += secs(tf0); } else {
                    ggml_init_params ip4 = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true }; ggml_context * c4 = ggml_init(ip4);
                    ggml_tensor * tc4 = ggml_new_tensor_2d(c4, GGML_TYPE_F32, E, n); ggml_set_input(tc4);
                    ggml_tensor * tf4 = ggml_new_tensor_2d(c4, GGML_TYPE_F32, E, n); ggml_set_input(tf4);
                    ggml_tensor * tm4 = ggml_new_tensor_2d(c4, GGML_TYPE_F32, E, n); ggml_set_input(tm4);
                    ggml_tensor * u4 = ggml_mul_mat(c4, Wt("ffn_up_shexp.weight"), tc4); ggml_tensor * g4 = ggml_mul_mat(c4, Wt("ffn_gate_shexp.weight"), tc4);
                    ggml_tensor * s4 = ggml_mul_mat(c4, Wt("ffn_down_shexp.weight"), ggml_swiglu_split(c4, g4, u4));
                    ggml_tensor * l4 = ggml_add(c4, ggml_add(c4, tm4, s4), tf4); ggml_set_output(l4);
                    ggml_cgraph * gg4 = ggml_new_graph(c4); ggml_build_forward_expand(gg4, l4);
                    if (!GA_SKIP) { GA_SKIP = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); } ggml_gallocr_alloc_graph(GA_SKIP, gg4);
                    pset(be, tc4, curh.data(), 0, ggml_nbytes(tc4)); pset(be, tf4, fih.data(), 0, ggml_nbytes(tf4)); pset(be, tm4, mo.data(), 0, ggml_nbytes(tm4));
                    { const auto c0 = clk::now(); ggml_backend_graph_compute(be, gg4); T_C += secs(c0); }
                    h.resize((size_t) E * n); pget(be, l4, h.data(), 0, h.size() * 4); HDEV_OK = false; pget_sync(be);
                    ggml_free(c4); } T_SK += secs(tk0);
                } else {
                { std::lock_guard<std::mutex> lk(MU); UPWIN = 1; } CV.notify_all(); const auto tl0 = clk::now(); Experts ex; if (!(WIDE && !last)) { ex = load_experts(be, p, used); } T_LD += secs(tl0); { std::lock_guard<std::mutex> lk(MU); UPWIN = 0; }
                RD_BUSY++; const auto tm0 = clk::now(); ggml_context * c2 = nullptr; ggml_tensor * lo2 = nullptr; ggml_tensor * moe = nullptr;
                if (WIDE && !last) {   /* wide pass: every expert of the union read once from the drive, in arena batches, applied to the rows that use it */
                    const auto tf0 = clk::now(); const int dfmt = TIDX.at(p + "ffn_down_exps.weight").type == GGML_TYPE_Q3_K ? FL_Q3K : FL_Q2K; { std::lock_guard<std::mutex> lk(PFM); PF_CUR = L; }
                    const char * kinds[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" }; size_t es[3]; const TI * tis[3]; for (int k = 0; k < 3; ++k) { tis[k] = &TIDX.at(p + kinds[k]); es[k] = ggml_row_size(tis[k]->type, tis[k]->ne[0]) * (size_t) tis[k]->ne[1]; }
                    auto al = [](size_t x) { return (x + 255) & ~(size_t) 255; }; const size_t per = al(es[0]) + al(es[1]) + al(es[2]); int EB = (int) std::min<size_t>(16, used.size()); { static int EBF = -1; if (EBF < 0) { const char * e = std::getenv("FUSED_WIDE_EB"); EBF = e ? std::atoi(e) : 0; } if (EBF > 0) { EB = (int) std::min<size_t>(EBF, used.size()); } }
                    if ((size_t) EB * per > EXP_ARENA_SIZE) { if (EXP_ARENA) { ggml_backend_buffer_free(EXP_ARENA); EXP_ARENA = nullptr; EXP_ARENA_SIZE = 0; } const size_t want = std::min((size_t) 520 << 20, (size_t) EB * per); EXP_ARENA = ggml_backend_alloc_buffer(be, want); if (!EXP_ARENA) { std::fprintf(stderr, "[fused] wide pass: no VRAM for an expert arena of %.0f MB\n", want / 1048576.0); std::exit(10); } EXP_ARENA_SIZE = want; }
                    EB = (int) std::max<size_t>(1, std::min<size_t>(EB, EXP_ARENA_SIZE / per)); if (!EXP_STREAM) { cudaStreamCreateWithFlags(&EXP_STREAM, cudaStreamNonBlocking); }
                    uint8_t * ab = (uint8_t *) ggml_backend_buffer_get_base(EXP_ARENA); uint8_t * gb = ab, * ub = ab + al((size_t) EB * es[0]), * db = ub + al((size_t) EB * es[1]); const size_t half = al((size_t) EB * (es[0] + es[1] + es[2])); ensure_pin_stage(2 * half);   /* two staging halves: a batch's drive reads land while the previous batch's copies and kernels run */
                    std::vector<std::vector<std::pair<int, int>>> byexp(used.size()); for (int64_t i = 0; i < n; ++i) { for (int k = 0; k < KE; ++k) { byexp[idsc[i * KE + k]].push_back({ (int) i, k }); } }
                    double rt = 0; size_t nread = 0; int nsl = 0, maxmem = 0; static int WDBG = -1; if (WDBG < 0) { WDBG = std::getenv("FUSED_WIDE_DEBUG") ? 1 : 0; }
                    if (WDBG) { std::fprintf(stderr, "[wide] L%d start: n=%lld union=%zu EB=%d half=%.0f MB stage=%.0f MB arena=%.0f MB\n", L, (long long) n, used.size(), EB, half / 1048576.0, PIN_STAGE_SIZE / 1048576.0, EXP_ARENA_SIZE / 1048576.0); std::fflush(stderr); }
                    for (size_t b0 = 0; b0 < used.size(); b0 += EB) { const int nb = (int) std::min<size_t>(EB, used.size() - b0); const int par = (int) ((b0 / EB) & 1); fl_wide_wait(FL, par);   /* this parity's staging half and slot tables: the GPU finished with them two batches ago */
                        std::vector<Job> jobs; std::vector<uint8_t *> dev; size_t so = 0; uint8_t * bases[3] = { gb, ub, db };
                        for (int k = 0; k < 3; ++k) { for (int u = 0; u < nb; ++u) { jobs.push_back({ tis[k]->file, tis[k]->off + (uint64_t) used[b0 + u] * es[k], es[k], PIN_STAGE_BASE + par * half + so }); dev.push_back(bases[k] + (size_t) u * es[k]); so += es[k]; } }
                        const auto tr0 = clk::now(); RD_BUSY++; read_jobs(jobs, dev.data(), EXP_STREAM); RD_BUSY--; rt += secs(tr0); nread += so; B_DISK += (double) so;
                        std::vector<int> midx, rows, kidx; std::vector<float> wk;
                        for (int u = 0; u < nb; ++u) { const auto & lst = byexp[b0 + u]; maxmem = std::max(maxmem, (int) lst.size()); for (size_t i0 = 0; i0 < lst.size(); i0 += 8) { midx.push_back(u); for (int j = 0; j < 8; ++j) { if (i0 + j < lst.size()) { rows.push_back(lst[i0 + j].first); kidx.push_back(lst[i0 + j].second); wk.push_back(wh[(size_t) lst[i0 + j].first * KE + lst[i0 + j].second]); } else { rows.push_back(-1); kidx.push_back(0); wk.push_back(0.f); } } } }
                        if (!midx.empty()) { fl_wide_moe(FL, gb, ub, db, dfmt, nb, midx.data(), rows.data(), kidx.data(), wk.data(), (int) midx.size(), KE, par); nsl += (int) midx.size(); } fl_wide_mark(FL, par);
                    }
                    { if (WDBG) { std::fprintf(stderr, "[wide] L%d n=%lld union=%zu EB=%d slots=%d maxpairs=%d read %.0f MB in %.2fs | attn gpu so far: proj %.0f | core %.0f | o-proj %.0f | router/shared %.0f ms\n", L, (long long) n, used.size(), EB, nsl, maxmem, nread / 1048576.0, rt, fl_gpu_ms(FL, 4), fl_gpu_ms(FL, 5), fl_gpu_ms(FL, 6), fl_gpu_ms(FL, 7)); } }
                    fl_wide_finish(FL, (int) n, KE, &FO); if (ATTR_ON) { attr_moe(selh, wh, idsc, fih, FO, KE, last_row, true); } T_EX += rt; T_FM += secs(tf0) - rt; T_C += secs(tf0) - rt; T_MB += secs(tm0); RD_BUSY--;
                    if (!WIDE_SAID) { WIDE_SAID = true; std::fprintf(stderr, "[fused] wide pass: layer %d, %lld rows, %zu experts read once (%.2f GB) in batches of %d\n", L, (long long) n, used.size(), nread / 1e9, EB); }
                } else if (FUSE) {   /* routed experts as launches on the arena (patch_fused.py) */
                    const auto tf0 = clk::now(); const int dfmt = TIDX.at(p + "ffn_down_exps.weight").type == GGML_TYPE_Q3_K ? FL_Q3K : FL_Q2K;
                    fl_moe(FL, ex.t.at("ffn_gate_exps.weight")->data, ex.t.at("ffn_up_exps.weight")->data, ex.t.at("ffn_down_exps.weight")->data, (int) used.size(), dfmt, idsc.data(), wh.data(), KE, (int) n, &FO); if (ATTR_ON) { attr_moe(selh, wh, idsc, fih, FO, KE, (N > 1 && !last) ? last_row : 0, false); }
                    T_FM += secs(tf0); T_C += secs(tf0); T_MB += secs(tm0); RD_BUSY--;
                } else {
                ggml_init_params ip2 = { ggml_tensor_overhead() * 1024 + ggml_graph_overhead(), nullptr, true }; c2 = ggml_init(ip2);
                ggml_tensor * tcur = ggml_new_tensor_2d(c2, GGML_TYPE_F32, E, n); ggml_set_input(tcur);
                ggml_tensor * tfi = ggml_new_tensor_2d(c2, GGML_TYPE_F32, E, n); ggml_set_input(tfi);
                ggml_tensor * tids = ggml_new_tensor_2d(c2, GGML_TYPE_I32, KE, n); ggml_set_input(tids);
                ggml_tensor * tw = ggml_new_tensor_3d(c2, GGML_TYPE_F32, 1, KE, n); ggml_set_input(tw);
                ggml_tensor * cur3 = ggml_reshape_3d(c2, tcur, E, 1, n);
                ggml_tensor * exo = nullptr;   /* row groups of MOE_G: the fused per-row expert kernel runs whatever the pass width */
                for (int64_t c0 = 0; c0 < n; c0 += MOE_G) { const int64_t g = std::min<int64_t>(MOE_G, n - c0);
                    ggml_tensor * cur_c = ggml_view_3d(c2, cur3, E, 1, g, cur3->nb[1], cur3->nb[2], c0 * cur3->nb[2]);
                    ggml_tensor * ids_c = ggml_view_2d(c2, tids, KE, g, tids->nb[1], c0 * tids->nb[1]);
                    ggml_tensor * up = ggml_mul_mat_id(c2, ex.t.at("ffn_up_exps.weight"), cur_c, ids_c);
                    ggml_tensor * gate = ggml_mul_mat_id(c2, ex.t.at("ffn_gate_exps.weight"), cur_c, ids_c);
                    ggml_tensor * ex_c = ggml_mul_mat_id(c2, ex.t.at("ffn_down_exps.weight"), ggml_swiglu_split(c2, gate, up), ids_c);
                    exo = exo ? ggml_concat(c2, exo, ex_c, 2) : ex_c; }
                exo = ggml_mul(c2, exo, tw);
                moe = ggml_view_2d(c2, exo, E, n, exo->nb[2], 0);
                for (int i = 1; i < KE; ++i) { moe = ggml_add(c2, moe, ggml_view_2d(c2, exo, E, n, exo->nb[2], i * exo->nb[1])); }
                ggml_tensor * ups = ggml_mul_mat(c2, Wt("ffn_up_shexp.weight"), tcur); ggml_tensor * gas = ggml_mul_mat(c2, Wt("ffn_gate_shexp.weight"), tcur);
                ggml_tensor * sh = ggml_mul_mat(c2, Wt("ffn_down_shexp.weight"), ggml_swiglu_split(c2, gas, ups));
                lo2 = ggml_add(c2, ggml_add(c2, moe, sh), tfi); ggml_set_output(lo2); ggml_set_output(moe);
                ggml_cgraph * g2 = ggml_new_graph(c2); ggml_build_forward_expand(g2, lo2);
                if (!GA_MOE) { GA_MOE = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); } ggml_gallocr_alloc_graph(GA_MOE, g2);
                pset(be, tcur, curh.data(), 0, ggml_nbytes(tcur)); pset(be, tfi, fih.data(), 0, ggml_nbytes(tfi));
                pset(be, tids, idsc.data(), 0, ggml_nbytes(tids)); pset(be, tw, wh.data(), 0, ggml_nbytes(tw));
                if (!NODES_PRINTED && L == 10) { NODES_PRINTED = true; std::fprintf(stderr, "[graph] layer 10: attention(+spec+scan) graph %d nodes, moe graph %d nodes\n", ggml_graph_n_nodes(gf), ggml_graph_n_nodes(g2)); } T_MB += secs(tm0); RD_BUSY--; { const auto c0 = clk::now(); ggml_backend_graph_compute(be, g2); T_C += secs(c0); }
                }
                h.resize((size_t) E * n); bool any_hit = false; for (int64_t i = 0; i < n; ++i) { if (!ghit.empty() && ghit[i] >= 0) { any_hit = true; break; } } const bool need_h = !SPEC_READ || !HCHAIN || SIB || any_hit || L == MID_L || L >= 59 || FUSE; if (need_h) { if (FUSE) { std::memcpy(h.data(), FO.h, h.size() * 4); H_DEV = !WIDE; } else { pget(be, lo2, h.data(), 0, h.size() * 4); pget_sync(be); } } HDEV = lo2; HDEV_PASS = PASS_ID; HDEV_OK = SPEC_READ && HCHAIN && !any_hit && !last && !SIB && !FUSE;
                const auto ts0 = clk::now(); if (GMEM_ON && !SIB && L >= 3) { std::vector<float> mv((size_t) E * n); if (FUSE) { std::memcpy(mv.data(), FO.moe, mv.size() * 4); } else { pget(be, moe, mv.data(), 0, mv.size() * 4); pget_sync(be); } auto & G = GMEM[L]; std::vector<float> tmp(E);
                    for (int64_t i = 0; i < n; ++i) { if (!ghit.empty() && ghit[i] >= 0) { gout_read(G[ghit[i]].off, tmp.data()); for (int d = 0; d < E; ++d) { h[i * E + d] += tmp[d]; } H_DEV = false; if (L == 60 && GKVF && i < N_PAD_FROM && G[ghit[i]].kvoff == ~0ull && KVLAT.size() >= (size_t) (i + 1) * 61 * 576) { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GKVF, 0, SEEK_END); G[ghit[i]].kvoff = GKV_END; std::fwrite(KVLAT.data() + (size_t) i * 61 * 576, 2, 61 * 576, GKVF); GKV_END += (uint64_t) 61 * 576 * 2; } } }   /* a stored walk gains its latents the first time it is replayed */   // matched rows: the stored expert output stands in for the read
                    for (int64_t i = 0; i < n; ++i) { if (i >= N_PAD_FROM) { continue; } if (!ghit.empty() && ghit[i] >= 0) { continue; } const float * x = curh.data() + i * E; double nn = 0; for (int d = 0; d < E; ++d) { nn += (double) x[d] * x[d]; }
                        GEntry ge; if ((int64_t) SXK.size() == (int64_t) GSK * n) { ge.sk.assign(SXK.begin() + i * GSK, SXK.begin() + (i + 1) * GSK); ge.off = gout_append(mv.data() + i * E); } else { const float inv = 1.0f / (float) std::sqrt(nn + 1e-12); for (int d = 0; d < E; ++d) { tmp[d] = x[d] * inv; } ge = gmake(tmp.data(), mv.data() + i * E); } ge.pos = G_POS + (uint32_t) i; ge.runid = G_RUNID; ge.crun = G_RUNID; ge.cpos = ge.pos; if (L == 60 && GKVF && KVLAT.size() >= (size_t) (i + 1) * 61 * 576) { std::lock_guard<std::mutex> lk(GOUTM); _fseeki64(GKVF, 0, SEEK_END); ge.kvoff = GKV_END; std::fwrite(KVLAT.data() + (size_t) i * 61 * 576, 2, 61 * 576, GKVF); GKV_END += (uint64_t) 61 * 576 * 2; } for (int k = 0; k < KE && k < 8; ++k) { ge.route[k] = selh[i * KE + k]; }   // the sketch the lookup computed on the GPU is what gets stored, with the experts this row used
                        ge.used = G_CLOCK | ((i < (int64_t) ROW_DERIVED.size() && ROW_DERIVED[i]) ? 0u : G_EXACT); const uint64_t gk = gkey(ge.runid, ge.pos); if (G.size() >= GCAP) { size_t ev = 0; for (size_t q = 1; q < G.size(); ++q) { if (G[q].used < G[ev].used) { ev = q; } } { auto io = GIDX[L].find(gkey(G[ev].runid, G[ev].pos)); if (io != GIDX[L].end() && io->second == (uint32_t) ev) { GIDX[L].erase(io); } } G[ev] = std::move(ge); GIDX[L][gk] = (uint32_t) ev; GDIRTY[L].push_back((uint32_t) ev); GEVICT[L]++; } else { GIDX[L][gk] = (uint32_t) G.size(); G.push_back(std::move(ge)); } GVER[L]++; } } T_ST += secs(ts0);
                if (omem_rows) { std::vector<float> mv((size_t) E * n); if (FUSE) { std::memcpy(mv.data(), FO.moe, mv.size() * 4); } else { pget(be, moe, mv.data(), 0, mv.size() * 4); pget_sync(be); }   // store each row's expert output under its token pair
                    for (int64_t i = 0; i < n && i < N_PAD_FROM; ++i) { OMEM[rkey(ROW_PREV[i], ROW_TOK[i], L)].assign(mv.begin() + i * E, mv.begin() + (i + 1) * E); } }
                { const auto tr0 = clk::now(); if (c2) { if (SPEC_READ) { if (PREV_C2) { ggml_free(PREV_C2); } PREV_C2 = c2; } else { ggml_free(c2); } } ggml_backend_buffer_free(ex.buf); ggml_free(ex.ctx); T_FR += secs(tr0); }
                }
            }
            if (TRACEF && (G_POS == 0 || (FUSED && N > 1)) && !SIB) { const int32_t hdr[4] = { (int32_t) G_RUNID, L, (int32_t) n, (int32_t) N }; std::fwrite(hdr, 4, 4, TRACEF); std::fwrite(h.data(), 4, E, TRACEF); std::fwrite(h.data() + (size_t) (n - 1) * E, 4, E, TRACEF); std::fflush(TRACEF); }
            if (ATTR_ON) { attr_layer(L, dense, h, (N > 1 && !last) ? last_row : 0); } n_in = n; { const auto tr0 = clk::now(); if (ctx) { ggml_free(ctx); } T_FR += secs(tr0); }
            if (L == MID_L) { HMID = h; }
            if (PF_ON && !SIB && L < 60 && ROUTE_COVER[std::min(60, L + PF_D0)] != PASS_ID) { { std::lock_guard<std::mutex> lk(PFM); while (PFQ.size() > 2) { PFQ.pop_front(); PF_DROP++; } PFQ.push_back({ L, h }); } PFCV.notify_all(); }
            { std::lock_guard<std::mutex> lk(MU); CONS = L; } CV.notify_all();
        }
        ld.join(); HFIN = h;
        // head on the last position
        const auto t0 = clk::now();
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true }; ggml_context * ctx = ggml_init(ip);
        ggml_tensor * hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n_in); ggml_set_input(hin);
        ggml_tensor * xnh = ggml_mul(ctx, ggml_rms_norm(ctx, hin, eps), HT["output_norm.weight"]);
        ggml_tensor * lg = CENT ? ggml_mul_mat(ctx, CENT, xnh) : ggml_mul_mat(ctx, HT["output.weight"], xnh); ggml_set_output(lg); if (CENT) { ggml_set_output(xnh); }
        ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, lg);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(hin, h.data(), 0, ggml_nbytes(hin)); ggml_backend_graph_compute(be, gf);
        if (!CENT) { logits.resize(ggml_nelements(lg)); ggml_backend_tensor_get(lg, logits.data(), 0, logits.size() * 4); if ((int64_t) n_in > N_PAD_FROM) { logits.resize((size_t) (logits.size() / n_in) * N_PAD_FROM); } }   /* drop the pad rows' logits */ if (ATTR_ON) { attr_finish(logits, h, HT, TIDX, PATHS, eps); }
        else {
            const int NCL = (int) CENT->ne[1]; std::vector<float> sc((size_t) NCL * n_in), xh((size_t) E * n_in);
            ggml_backend_tensor_get(lg, sc.data(), 0, sc.size() * 4); ggml_backend_tensor_get(xnh, xh.data(), 0, xh.size() * 4);
            std::vector<char> pick(NCL, 0);
            for (int64_t r = 0; r < n_in; ++r) {
                std::vector<int> o(NCL); for (int c = 0; c < NCL; ++c) { o[c] = c; }
                std::partial_sort(o.begin(), o.begin() + sieve_c, o.end(), [&](int a, int b) { return sc[r * NCL + a] > sc[r * NCL + b]; });
                for (int k = 0; k < sieve_c; ++k) { pick[o[k]] = 1; }
            }
            std::vector<int32_t> cand; for (int c = 0; c < NCL; ++c) { if (pick[c]) { cand.insert(cand.end(), SV_MEM[c].begin(), SV_MEM[c].end()); } }
            const size_t VOC = SV_ASG.size(); logits.assign(VOC * n_in, -1e30f);
            for (size_t c0 = 0; c0 < cand.size(); c0 += 8192) {               // gather candidate rows in chunks (fp32 gather of 8192 rows = 235 MB)
                const size_t m = std::min<size_t>(8192, cand.size() - c0);
                ggml_init_params ip3 = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true }; ggml_context * c3 = ggml_init(ip3);
                ggml_tensor * tid = ggml_new_tensor_1d(c3, GGML_TYPE_I32, (int64_t) m); ggml_set_input(tid);
                ggml_tensor * tx = ggml_new_tensor_2d(c3, GGML_TYPE_F32, E, n_in); ggml_set_input(tx);
                ggml_tensor * lc = ggml_mul_mat(c3, ggml_get_rows(c3, HT["output.weight"], tid), tx); ggml_set_output(lc);
                ggml_cgraph * g3 = ggml_new_graph(c3); ggml_build_forward_expand(g3, lc);
                ggml_gallocr_t ga3 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga3, g3);
                ggml_backend_tensor_set(tid, cand.data() + c0, 0, ggml_nbytes(tid)); ggml_backend_tensor_set(tx, xh.data(), 0, ggml_nbytes(tx)); ggml_backend_graph_compute(be, g3);
                std::vector<float> lv(m * n_in); ggml_backend_tensor_get(lc, lv.data(), 0, lv.size() * 4);
                for (int64_t r = 0; r < n_in; ++r) { for (size_t j = 0; j < m; ++j) { logits[r * VOC + cand[c0 + j]] = lv[r * m + j]; } }
                ggml_gallocr_free(ga3); ggml_free(c3);
            }
        }
        ggml_gallocr_free(ga); ggml_free(ctx); T_HEAD += secs(t0);
        std::fprintf(stderr, "  [pass] rows %lld wall %.2fs | wait %.2f attn-build %.2f gets %.2f lookup %.2f experts %.2f (read %.2f) moe-build %.2f skip-path %.2f store %.2f free %.2f gpu %.2f head %.2f | disk %.2f GB\n", (long long) N, secs(tf0), T_W, T_AB, T_GET, G_TLOOK - lk0, T_LD, T_EX, T_MB, T_SK, T_ST, T_FR, T_C, T_HEAD, (B_DISK + B_DISK_W) / 1e9);
        if (FUSED && FL) { std::fprintf(stderr, "  [fused] gpu: attention+router+shared %.2f | spec+scan %.2f | experts %.2f s (phase %.2f, of it gpu idle on weight uploads %.2f) | host: attention phase %.2f (spec weight wait %.2f), expert phase %.2f | loader: copies %.2f s for %.1f GB (%.1f GB/s while copying), slot waits %.2f s\n", fl_gpu_ms(FL, 2) / 1e3, fl_gpu_ms(FL, 3) / 1e3, fl_gpu_ms(FL, 1) / 1e3, fl_gpu_ms(FL, 0) / 1e3, (fl_gpu_ms(FL, 0) - fl_gpu_ms(FL, 2) - fl_gpu_ms(FL, 3)) / 1e3, T_FA, T_SPECW, T_FM, LD_BUSY, LD_BYTES / 1e9, LD_BUSY > 0 ? LD_BYTES / 1e9 / LD_BUSY : 0.0, LD_WAIT); }
    };
    char stamp[32]; { std::time_t tt = std::time(nullptr); std::tm g; gmtime_s(&g, &tt); std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%SZ", &g); }
    const fs::path log_dir = fs::path(results_dir) / std::string(stamp).substr(0, 8); fs::create_directories(log_dir);
    const fs::path log_path = log_dir / (std::string(stamp) + "_async.jsonl"); FILE * jl = std::fopen(log_path.string().c_str(), "wb");
    std::string text_out; double dec_sum = 0; int dec_n = 0;
    auto esc = [](const std::string & x) { std::string o; for (unsigned char c : x) { if (c == '"' || c == '\\') { o += '\\'; o += (char) c; } else if (c == '\n') { o += "\\n"; } else if (c < 0x20) { char b2[8]; std::snprintf(b2, sizeof(b2), "\\u%04x", c); o += b2; } else { o += (char) c; } } return o; };
    std::vector<float> logits;
    auto emit = [&](int step, int tok, double sec, int forced) {
        const int shown = forced >= 0 ? forced : tok;
        const std::string piece = has_vocab ? detok.decode(shown) : ("[" + std::to_string(shown) + "]"); text_out += piece;
        std::printf("%s", piece.c_str()); std::fflush(stdout); if (CHAT) { chat_send("T " + std::to_string(shown) + " " + chat_esc(piece)); }
        if (step > 0) { dec_sum += sec; dec_n++; }
        // full output head at this position: exact top-K logits, softmax over the whole vocab, top-1 margin
        std::vector<int> ix(logits.size()); for (size_t i = 0; i < ix.size(); ++i) { ix[i] = (int) i; }
        const int K = std::min<int>(topk, (int) ix.size());
        std::partial_sort(ix.begin(), ix.begin() + K, ix.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        const double mx = logits[ix[0]]; double Z = 0; for (float v : logits) { Z += std::exp((double) v - mx); }
        std::string tk = "[";
        for (int k = 0; k < K; ++k) { const int id = ix[k]; char b3[96]; std::snprintf(b3, sizeof(b3), "%s[%d,%.4f,%.5f,\"", k ? "," : "", id, logits[id], std::exp(logits[id] - mx) / Z);
            tk += b3; tk += esc(has_vocab ? detok.decode(id) : std::string()); tk += "\"]"; }
        tk += "]";
        std::string fz;
        if (forced >= 0) { int rank = 0; for (float v : logits) { rank += v > logits[forced]; }
            char b4[200]; std::snprintf(b4, sizeof(b4), ",\"fed_token\":%d,\"fed_rank\":%d,\"fed_logit\":%.4f,\"fed_prob\":%.5f,\"top1_agree\":%s", forced, rank, logits[forced], std::exp(logits[forced] - mx) / Z, forced == ix[0] ? "true" : "false"); fz = b4; }
        size_t vf = 0, vt = 0; cudaMemGetInfo(&vf, &vt);
        std::fprintf(jl, "{\"step\":%d,\"head_top1\":%d,\"text\":\"%s\",\"top1_margin\":%.4f%s,\"topk\":%s,\"seconds\":%.4f,\"wait_weights_s\":%.4f,\"disk_read_experts_s\":%.4f,\"expert_upload_s\":%.4f,\"kernel_s\":%.4f,\"head_s\":%.4f,\"disk_gb\":%.3f,\"vram_used_mb\":%.0f}\n",
                     step, tok, esc(piece).c_str(), K > 1 ? logits[ix[0]] - logits[ix[1]] : 0.0, fz.c_str(), tk.c_str(), sec, T_W, T_EX, T_EXUP, T_C, T_HEAD, (B_DISK + B_DISK_W) / 1e9, (vt - vf) / 1048576.0); std::fflush(jl);
    };
    std::printf("\nprompt: %zu tokens | streaming R1 output:\n\n", prompt.size());
    auto reset = [&]() { T_W = T_EX = T_EXUP = T_C = T_HEAD = 0; T_AB = T_GET = T_LD = T_MB = T_ST = T_SK = T_FR = 0; B_DISK = 0; B_DISK_W = 0; };
    reset(); auto t0 = clk::now(); auto turn_clock = t0;
    if (!rmem_path.empty()) {
        RMEM_ON = true; RMEM_MAXL = rmem_layers; FILE * f = std::fopen(rmem_path.c_str(), "rb");
        if (f) { uint64_t k; std::array<int16_t, 8> a; while (std::fread(&k, 8, 1, f) == 1 && std::fread(a.data(), 2, 8, f) == 8) { RMEM[k] = a; } std::fclose(f); }
        std::printf("[route memory] %zu labels loaded from %s; layers 3-%d served from labels\n", RMEM.size(), rmem_path.c_str(), RMEM_MAXL);
    }
    if (!gmem_path.empty()) {
        GMEM_ON = true; gproj_init();
        { ggml_init_params ipp = { ggml_tensor_overhead() * 2, nullptr, true }; ggml_context * pc = ggml_init(ipp); TPROJ = ggml_new_tensor_2d(pc, GGML_TYPE_F32, 7168, GSK); ggml_backend_alloc_ctx_tensors(pc, be); ggml_backend_tensor_set(TPROJ, GPROJ.data(), 0, ggml_nbytes(TPROJ)); } FILE * f = std::fopen(gmem_path.c_str(), "rb"); size_t ne = 0;
        { const std::string op = gmem_path + ".out"; GOUTF = std::fopen(op.c_str(), "r+b"); if (!GOUTF) { GOUTF = std::fopen(op.c_str(), "w+b"); } _fseeki64(GOUTF, 0, SEEK_END); GOUT_END = (uint64_t) _ftelli64(GOUTF); } { const std::string op = gmem_path + ".out32"; GOUTF32 = std::fopen(op.c_str(), "r+b"); if (!GOUTF32) { GOUTF32 = std::fopen(op.c_str(), "w+b"); } _fseeki64(GOUTF32, 0, SEEK_END); GOUT32_END = (uint64_t) _ftelli64(GOUTF32); } if (MLA) { const std::string kp = gmem_path + ".kv"; GKVF = std::fopen(kp.c_str(), "r+b"); if (!GKVF) { GKVF = std::fopen(kp.c_str(), "w+b"); } if (GKVF) { _fseeki64(GKVF, 0, SEEK_END); GKV_END = (uint64_t) _ftelli64(GKVF); } }
        if (f) { char magic[4] = { 0 }; const bool hasmg = std::fread(magic, 1, 4, f) == 4; const bool v2 = hasmg && std::memcmp(magic, "GEO2", 4) == 0, v3 = hasmg && std::memcmp(magic, "GEO3", 4) == 0, v4 = hasmg && std::memcmp(magic, "GEO4", 4) == 0, v5 = hasmg && std::memcmp(magic, "GEO5", 4) == 0, v6 = hasmg && std::memcmp(magic, "GEO6", 4) == 0, v7 = hasmg && std::memcmp(magic, "GEO7", 4) == 0, v8 = hasmg && std::memcmp(magic, "GEO8", 4) == 0, v9 = hasmg && std::memcmp(magic, "GEO9", 4) == 0, va = hasmg && std::memcmp(magic, "GEOA", 4) == 0; if (!v2 && !v3 && !v4 && !v5 && !v6 && !v7 && !v8 && !v9 && !va) { std::fseek(f, 0, SEEK_SET); }
            int32_t L; std::vector<float> kf(7168), of(7168); std::vector<ggml_fp16_t> kh(7168);
            while (std::fread(&L, 4, 1, f) == 1) {
                GEntry ge;
                if (v5 || v6 || v7 || v8 || v9 || va) { const size_t nn6 = (v9 || va) ? 16 : ((v6 || v7 || v8) ? 8 : 4); ge.sk.resize(GSK); if (std::fread(ge.sk.data(), 4, GSK, f) != (size_t) GSK || std::fread(&ge.off, 8, 1, f) != 1 || std::fread(&ge.used, 4, 1, f) != 1 || std::fread(&ge.pos, 4, 1, f) != 1 || std::fread(&ge.runid, 4, 1, f) != 1 || ((v8 || v9 || va) && (std::fread(&ge.crun, 4, 1, f) != 1 || std::fread(&ge.cpos, 4, 1, f) != 1)) || std::fread(ge.nxt, 4, nn6, f) != nn6 || ((v7 || v8 || v9 || va) && std::fread(ge.route, 4, 8, f) != 8) || (va && std::fread(&ge.kvoff, 8, 1, f) != 1)) { break; } if (!v8 && !v9 && !va) { ge.crun = ge.runid; ge.cpos = ge.pos; } { double sn = 0; for (int q = 0; q < GSK; ++q) { sn += (double) ge.sk[q] * ge.sk[q]; } const float si = 1.0f / (float) std::sqrt(sn + 1e-20); for (int q = 0; q < GSK; ++q) { ge.sk[q] *= si; } } if (ge.runid < G_EXACT_SINCE) { ge.used &= ~G_EXACT; } }
                else if (v4) { ge.sk.resize(GSK); if (std::fread(ge.sk.data(), 4, GSK, f) != (size_t) GSK || std::fread(&ge.off, 8, 1, f) != 1 || std::fread(&ge.used, 4, 1, f) != 1) { break; } }
                else if (v3) { ge.sk.resize(GSK); if (std::fread(ge.sk.data(), 4, GSK, f) != (size_t) GSK || std::fread(kh.data(), 2, 7168, f) != 7168) { break; } ggml_fp16_to_fp32_row(kh.data(), of.data(), 7168); ge.off = gout_append(of.data()); }
                else {
                    if (v2) { if (std::fread(kh.data(), 2, 7168, f) != 7168) { break; } ggml_fp16_to_fp32_row(kh.data(), kf.data(), 7168); } else { if (std::fread(kf.data(), 4, 7168, f) != 7168) { break; } }
                    if (std::fread(kh.data(), 2, 7168, f) != 7168) { break; } ggml_fp16_to_fp32_row(kh.data(), of.data(), 7168); ge = gmake(kf.data(), of.data());
                }
                if (L >= 0 && L < 61) { GMEM[L].push_back(std::move(ge)); ne++; } }
            std::fclose(f); }
        for (int L = 0; L < 61; ++L) { if (GMEM[L].size() > GMAXLOAD) { GMEM[L].erase(GMEM[L].begin(), GMEM[L].end() - GMAXLOAD); } }
        ne = 0; for (int L = 0; L < 61; ++L) { ne += GMEM[L].size(); }
        for (int L = 0; L < 61; ++L) { GIDX[L].clear(); for (size_t j = 0; j < GMEM[L].size(); ++j) { GIDX[L][gkey(GMEM[L][j].runid, GMEM[L][j].pos)] = (uint32_t) j; } } runp_load();
        std::printf("[geometry memory] %zu stored states loaded from %s; reuse when cosine >= %.3f (cap %zu per layer)\n", ne, gmem_path.c_str(), GSIM, GCAP);
    }
    if (!omem_path.empty()) {
        OMEM_ON = true; if (!RMEM_ON) { RMEM_MAXL = rmem_layers; }
        FILE * f = std::fopen(omem_path.c_str(), "rb");
        if (f) { uint64_t k; std::vector<float> v(7168); while (std::fread(&k, 8, 1, f) == 1 && std::fread(v.data(), 4, 7168, f) == 7168) { OMEM[k] = v; } std::fclose(f); }
        std::printf("[output memory] %zu stored expert outputs loaded from %s; layers 3-%d use them in place of the expert pull\n", OMEM.size(), omem_path.c_str(), RMEM_MAXL);
    }
    auto set_rows = [&](const std::vector<int32_t> & toks, int32_t prev0) { ROW_TOK = toks; ROW_PREV.assign(toks.size(), -1); for (size_t i = 0; i < toks.size(); ++i) { ROW_PREV[i] = i ? toks[i - 1] : prev0; } };
    auto mem_queue = [&]() {                                       // every row whose ids are labelled: its experts for the early layers go to the prefetcher now
        if (!RMEM_ON || !PF_ON) { return; }
        std::lock_guard<std::mutex> lk(PFM);
        for (size_t i = 0; i < ROW_TOK.size(); ++i) { for (int L = 3; L <= RMEM_MAXL; ++L) {
            auto it = RMEM.find(rkey(ROW_PREV[i], ROW_TOK[i], L)); if (it == RMEM.end()) { continue; }
            for (int k = 0; k < std::min(experts_keep, 8); ++k) { PFX.push_back({ L, it->second[k] | (2 << 16) }); RMEM_QUEUED++; } } }
        PFCV.notify_all();
    };
    if (!dump_path.empty()) { DUMPF = std::fopen(dump_path.c_str(), "ab"); } if (!trace_path.empty()) { TRACEF = std::fopen(trace_path.c_str(), "ab"); } if (!moe_path.empty()) { MOEDF = std::fopen(moe_path.c_str(), "ab"); }
    if (!drafter_path.empty()) {
        FILE * f = std::fopen(drafter_path.c_str(), "rb"); char mg[4] = { 0 }; int32_t ee = 0;
        if (f && std::fread(mg, 1, 4, f) == 4 && std::memcmp(mg, "DRF1", 4) == 0 && std::fread(&ee, 4, 1, f) == 1 && ee == E) {
            for (int m = 0; m < 2; ++m) { int32_t nn = 0; std::fread(&nn, 4, 1, f); DR_N[m] = nn; DR_MX[m].resize(E); DR_MY[m].resize(E); DR_X[m].resize((size_t) nn * E); DR_A[m].resize((size_t) nn * E);
                std::fread(DR_MX[m].data(), 4, E, f); std::fread(DR_MY[m].data(), 4, E, f); std::fread(DR_X[m].data(), 4, DR_X[m].size(), f); std::fread(DR_A[m].data(), 4, DR_A[m].size(), f); }
            DR_ON = true; std::printf("[drafter] look-ahead maps loaded (%d / %d training positions): a second and third token are drafted from the state's geometry\n", DR_N[0], DR_N[1]);
        }
        if (f) { std::fclose(f); }
    }
    auto dr_pred = [&](const float * x, int m, float * out) {
        const int nn = DR_N[m]; std::vector<float> xc(E), kk(nn); for (int d = 0; d < E; ++d) { xc[d] = x[d] - DR_MX[m][d]; }
        for (int i = 0; i < nn; ++i) { const float * row = DR_X[m].data() + (size_t) i * E; float a0 = 0, a1 = 0, a2 = 0, a3 = 0; for (int d = 0; d < E; d += 4) { a0 += row[d] * xc[d]; a1 += row[d + 1] * xc[d + 1]; a2 += row[d + 2] * xc[d + 2]; a3 += row[d + 3] * xc[d + 3]; } kk[i] = a0 + a1 + a2 + a3; }
        for (int d = 0; d < E; ++d) { out[d] = DR_MY[m][d]; }
        for (int i = 0; i < nn; ++i) { const float * row = DR_A[m].data() + (size_t) i * E; const float k = kk[i]; for (int d = 0; d < E; ++d) { out[d] += k * row[d]; } }
    };
    auto head_top1 = [&](const std::vector<float> & st, int rows) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true }; ggml_context * c = ggml_init(ip);
        ggml_tensor * hin = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, rows); ggml_set_input(hin);
        ggml_tensor * lg = ggml_mul_mat(c, HT["output.weight"], ggml_mul(c, ggml_rms_norm(c, hin, eps), HT["output_norm.weight"])); ggml_set_output(lg);
        ggml_cgraph * gf = ggml_new_graph(c); ggml_build_forward_expand(gf, lg);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(hin, st.data(), 0, ggml_nbytes(hin)); ggml_backend_graph_compute(be, gf);
        std::vector<float> z(ggml_nelements(lg)); ggml_backend_tensor_get(lg, z.data(), 0, z.size() * 4); const size_t VV = z.size() / rows;
        std::vector<int32_t> o(rows); std::vector<float> mg(rows);
        for (int q = 0; q < rows; ++q) { float a = -1e30f, b = -1e30f; int bi = 0; for (size_t v = 0; v < VV; ++v) { const float zz = z[q * VV + v]; if (zz > a) { b = a; a = zz; bi = (int) v; } else if (zz > b) { b = zz; } } o[q] = bi; mg[q] = a - b; }
        ggml_gallocr_free(ga); ggml_free(c); return std::make_pair(o, mg);
    };
    auto chat_encode = [&](const std::string & text) {   // plain text through the tokenizer, chat markers to their ids
        static const std::pair<const char *, int32_t> MK[4] = { { "<\xef\xbd\x9c" "User" "\xef\xbd\x9c" ">", 128803 }, { "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">", 128804 }, { "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">", 1 }, { "<think>", 128798 } };
        std::vector<int32_t> tt; size_t a = 0;
        while (a < text.size()) { size_t best = std::string::npos; int bk = -1; for (int k = 0; k < 4; ++k) { const size_t q = text.find(MK[k].first, a); if (q < best) { best = q; bk = k; } }
            const size_t e = best == std::string::npos ? text.size() : best; if (e > a) { for (int t : detok.encode(text.substr(a, e - a))) { tt.push_back(t); } }
            if (bk < 0) { break; } tt.push_back(MK[bk].second); a = e + std::strlen(MK[bk].first); }
        return tt; };
    auto chat_turn_tokens = [&](const std::string & cmd, bool first) {   // "U <think> <answer> <escaped text>"
        size_t p = 2, q = cmd.find(' ', p); CHAT_THINK = std::max(0, std::atoi(cmd.substr(p, q - p).c_str())); p = q + 1; q = cmd.find(' ', p); CHAT_ANSWER = std::max(0, std::atoi(cmd.substr(p, q - p).c_str()));   /* 0 = no cap */
        std::vector<int32_t> tt; if (first) { tt.push_back(0); } const std::vector<int32_t> body = chat_encode(chat_unesc(cmd.substr(q + 1))); tt.insert(tt.end(), body.begin(), body.end()); return tt; };
    if (CHAT) { char cb[64]; std::snprintf(cb, sizeof(cb), "READY %lld", (long long) KVP); chat_send(cb); std::string cmd;
        for (;;) { cmd = chat_next(); if (cmd == "Q") { return 0; } if (cmd == "R") { chat_send("RESET"); continue; } if (cmd.size() > 2 && cmd[0] == 'U') { prompt = chat_turn_tokens(cmd, true); if ((int) prompt.size() + 4 > (int) KVP) { chat_send("FULL"); continue; } break; } } }
    G_RUNID = (uint32_t) std::time(nullptr); runp_add(G_RUNID, prompt); OURQ = fam_question(prompt); G_POS = 0; set_rows(prompt, -1); mem_queue();
    { const int sv = experts_keep; if (PREFILL_EXPERTS > 0) { experts_keep = PREFILL_EXPERTS; } if (prompt.size() <= (size_t) PREFILL_CHUNK) { forward(prompt, 0, logits); } else { for (size_t a = 0; a < prompt.size(); a += PREFILL_CHUNK) { std::vector<int32_t> ch(prompt.begin() + a, prompt.begin() + std::min(prompt.size(), a + PREFILL_CHUNK)); G_POS = (uint32_t) a; set_rows(ch, a ? prompt[a - 1] : -1); forward(ch, (int) a, logits); } } experts_keep = sv; }   /* the mask holds 64 rows: longer prompts go through in slices */
    if ((int64_t) HFIN.size() >= E) { GEO_X.assign(HFIN.end() - E, HFIN.end()); } GEO_LAST = (int) prompt.size() - 1; if (prompt.size() > (size_t) PREFILL_CHUNK) { GEO_LAST = (int) ((prompt.size() - 1) % PREFILL_CHUNK); }
    if (vram_keep_mb >= 0) {
        shrink_pin_stage((size_t) 320 << 20);   /* prefill staged ~80 experts at once; decode needs < 30 */
        size_t vf = 0, vt = 0; const size_t keep = (size_t) vram_keep_mb << 20; int mv = 0, mp = 0;
        for (int L = 0; L < 61; ++L) {
            LayerW & w = LW[L]; if (w.tier != 1) { continue; }
            cudaMemGetInfo(&vf, &vt); if (vf < keep + w.total + ((size_t) 64 << 20)) { continue; }   // too big for what's left: try the next (smaller) layer
            for (auto & nm : w.names) { w.t[nm]->data = nullptr; w.t[nm]->buffer = nullptr; }
            w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, be);
            if (!w.buf) { for (size_t i = 0; i < w.names.size(); ++i) { ggml_backend_tensor_alloc(SCR[L % NSLOT], w.t[w.names[i]], SCRB[L % NSLOT] + w.offs[i]); } break; }
            for (size_t i = 0; i < w.names.size(); ++i) { ggml_backend_tensor_set(w.t[w.names[i]], w.host + w.offs[i], 0, ggml_nbytes(w.t[w.names[i]])); }
            ggml_backend_buffer_free(w.hostbuf); w.hostbuf = nullptr; w.host = nullptr; pinned_used -= (double) w.total; vram_used += w.total; w.tier = 0; n_t[0]++; n_t[1]--; mv++;
        }
        for (int L = 0; L < 61; ++L) {
            LayerW & w = LW[L]; if (w.tier != 2 || pinned_used + w.total > pinned_cap) { continue; }
            w.hostbuf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), w.total); if (!w.hostbuf) { break; }
            w.host = (uint8_t *) ggml_backend_buffer_get_base(w.hostbuf);
            std::vector<Job> jobs; for (size_t i = 0; i < w.names.size(); ++i) { add_jobs(jobs, TIDX.at(w.names[i]), 0, ggml_nbytes(w.t[w.names[i]]), w.host + w.offs[i]); } read_jobs(jobs);
            pinned_used += (double) w.total; w.tier = 1; n_t[1]++; n_t[2]--; mp++;
        }
        cudaMemGetInfo(&vf, &vt);
        std::fprintf(stderr, "rebalance after prefill: %d layers pinned->VRAM, %d disk->pinned; now %d in VRAM, %d pinned, %d disk; VRAM free %.0f MB\n", mv, mp, n_t[0], n_t[1], n_t[2], vf / 1048576.0);
    }
    int tok = pick_tok(logits.data(), logits.size());
    std::fprintf(stderr, "prefill %zu tokens: %.2fs (waited for weights %.2f | experts read %.2f upload %.2f | compute %.2f | head %.2f | disk %.2f GB) -> first token %d (logit %.4f)\n",
                 prompt.size(), secs(t0), T_W, T_EX, T_EXUP, T_C, T_HEAD, (B_DISK + B_DISK_W) / 1e9, tok, logits[tok]);
    emit(0, tok, secs(t0), forced.empty() ? -1 : forced[0]); if (!forced.empty()) { tok = forced[0]; }
    if (siblings) {
        std::vector<int> idx(logits.size()); for (size_t i = 0; i < idx.size(); ++i) { idx[i] = (int) i; }
        std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        for (int K : { 1, 2, 4, 8, 16, 32 }) {
            std::vector<int32_t> cand(idx.begin(), idx.begin() + K); SIB = true; reset(); auto ts = clk::now(); std::vector<float> lg2;
            forward(cand, (int) prompt.size(), lg2); SIB = false; const double tt = secs(ts);
            double mu = 0; size_t mx = 0; for (size_t u : UNION) { mu += (double) u; mx = std::max(mx, u); } mu /= std::max<size_t>(1, UNION.size());
            std::fprintf(stderr, "K=%2d candidates side by side: pass %.2fs | experts pulled per layer: mean %.1f max %zu | expert read %.2fs (%.2f GB) | weights %.2fs | compute %.2fs\n",
                         K, tt, mu, mx, T_EX, B_DISK / 1e9, T_W, T_C);
        }
        return 0;
    }
    int pos = (int) prompt.size(); double dec_total = 0;
    if (PASS_W > 0 && draft_k > PASS_W - 1) { draft_k = PASS_W - 1; }   /* a pass holds the head's token plus the draft: never wider than the fixed width */
    if (!forced.empty()) { n_new = (int) forced.size(); draft_k = 0; }
    std::vector<int32_t> hist(prompt.begin(), prompt.end()); hist.push_back(tok); size_t gen_from = prompt.size(); int think_n = 0, answer_n = 0; bool in_think = true; std::string turn_end = "cap";
    auto note = [&](int t) { if (!CHAT) { return; } if (in_think) { if (t == 128799) { in_think = false; } else { think_n++; } } else { answer_n++; } };
    auto feed = [&](const std::vector<int32_t> & tt, bool exact) {   // chat: these tokens go through at the current position (64 rows per pass); the head's pick becomes the next token
        const int sv = experts_keep; if (exact && PREFILL_EXPERTS > 0) { experts_keep = PREFILL_EXPERTS; }
        for (size_t a = 0; a < tt.size(); a += PREFILL_CHUNK) { std::vector<int32_t> ch(tt.begin() + a, tt.begin() + std::min(tt.size(), a + PREFILL_CHUNK)); G_POS = (uint32_t) hist.size(); set_rows(ch, hist.empty() ? -1 : hist.back()); mem_queue(); reset(); t0 = clk::now(); forward(ch, pos, logits); hist.insert(hist.end(), ch.begin(), ch.end()); pos += (int) ch.size(); GEO_LAST = (int) ch.size() - 1; }
        experts_keep = sv; if ((int64_t) HFIN.size() >= E) { GEO_X.assign(HFIN.end() - E, HFIN.end()); }
        tok = pick_tok(logits.data(), logits.size()); };
    for (int turn = 0;; ++turn) {
        think_n = 0; answer_n = 0; in_think = true; turn_end = "cap"; CHAT_STOP = false;
        if (turn > 0) {   // chat: the next user turn (U), a reset (R) or quit (Q)
            std::string cmd; std::vector<int32_t> tt;
            for (;;) { cmd = chat_next(); if (cmd == "Q") { break; }
                if (cmd == "R") { ggml_backend_buffer_clear(kvbuf, 0); hist.clear(); pos = 0; chat_send("RESET"); continue; }
                if (cmd.size() > 2 && cmd[0] == 'U') { tt = chat_turn_tokens(cmd, hist.empty()); if (pos + (int) tt.size() + 4 > (int) KVP) { chat_send("FULL"); continue; } break; } }
            if (cmd == "Q") { break; }
            { const uint32_t nr = (uint32_t) std::time(nullptr); G_RUNID = nr > G_RUNID ? nr : G_RUNID + 1; } turn_clock = clk::now(); feed(tt, true); emit(0, tok, secs(turn_clock), -1); hist.push_back(tok); gen_from = hist.size() - 1; { std::vector<int32_t> pp(hist.begin(), hist.begin() + gen_from); runp_add(G_RUNID, pp); OURQ = fam_question(pp); }
        }
        note(tok);
        {
        int dk = draft_k; int s_emit = 1, n_passes = 0, n_drafted = 0, n_accepted = 0; double spec_time = 0;
        while (draft_k > 0 && s_emit < n_new && pos < KVP - 2 - draft_k && !CHAT_STOP) {
            if (CHAT && in_think && CHAT_THINK > 0 && think_n >= CHAT_THINK) {   // think budget spent: close the thought in R1's own voice and let it answer
                std::vector<int32_t> cl = detok.encode("\n\nOkay, I have thought about this enough. Let me answer now.\n"); cl.insert(cl.end(), { 128799, 201, 201 });
                if (pos + (int) cl.size() + 2 >= (int) KVP) { break; }
                const auto tc = clk::now(); feed(cl, false); const double tcs = secs(tc);
                for (int32_t t : cl) { const std::string pc = detok.decode(t); text_out += pc; chat_send("F " + std::to_string(t) + " " + chat_esc(pc)); if (t == 128799) { in_think = false; } }   /* forced tokens are not counted as thinking */
                emit(s_emit++, tok, tcs, -1); hist.push_back(tok); note(tok); if (tok == 1) { turn_end = "eos"; break; }
                continue;
            }
            std::vector<int32_t> draft; int draft_ng = 0;
            if (GEO_RUN > 0 && GKVF && MLA && GEO_LAST >= 0 && GEO_LAST < (int) GEO_NRE.size() && GEO_LAST < (int) GEO_NR.size() && GEO_NRS[GEO_LAST] >= 0.9999f && GEO_NR[GEO_LAST][0] == tok && geo_same_question(GEO_NRE[GEO_LAST].first)) {   /* unverified runs only along a walk recorded under this very question */
                /* geometry run: the last accepted row sits exactly on a stored state whose walk continues with the head's token; follow that walk through its stored latents, no pass */
                const uint32_t cr = GEO_NRE[GEO_LAST].first, cp = GEO_NRE[GEO_LAST].second; std::vector<int32_t> rt; std::vector<uint64_t> ro;
                for (int j = 1; j <= GEO_RUN && pos + (int) rt.size() < (int) KVP - 2 - draft_k; ++j) {
                    auto it = GIDX[60].find(gkey(cr, cp + (uint32_t) j)); if (it == GIDX[60].end() || it->second >= GMEM[60].size()) { break; }
                    const GEntry & e = GMEM[60][it->second]; if (e.kvoff == ~0ull || e.nxt[0] < 0 || e.runid != cr || e.pos != cp + (uint32_t) j) { break; }
                    ro.push_back(e.kvoff); rt.push_back(e.nxt[0]); if (rt.back() == 1) { break; }   /* latents for the position of the token just emitted; the walk's next token is new */
                }
                if (rt.size() >= 2) {
                    const auto tg0 = clk::now(); const int k = (int) rt.size(); std::vector<ggml_fp16_t> lat((size_t) k * 61 * 576);
                    { std::lock_guard<std::mutex> lk(GOUTM); for (int j = 0; j < k; ++j) { _fseeki64(GKVF, (long long) ro[j], SEEK_SET); std::fread(lat.data() + (size_t) j * 61 * 576, 2, 61 * 576, GKVF); } }
                    ggml_init_params ipl = { ggml_tensor_overhead() * 61 * 8 + ggml_graph_overhead_custom(61 * 8, false), nullptr, true }; ggml_context * cl = ggml_init(ipl); ggml_cgraph * gl = ggml_new_graph_custom(cl, 61 * 8, false);
                    std::vector<ggml_tensor *> ins(61);
                    for (int L = 0; L < 61; ++L) { ins[L] = ggml_new_tensor_2d(cl, GGML_TYPE_F16, 576, k); ggml_set_input(ins[L]);
                        ggml_build_forward_expand(gl, ggml_cpy(cl, ins[L], ggml_view_2d(cl, KC[L], 576, k, KC[L]->nb[1], (size_t) pos * KC[L]->nb[1])));
                        if (FUSED) { ggml_build_forward_expand(gl, ggml_cpy(cl, ggml_view_2d(cl, ins[L], 512, k, ins[L]->nb[1], 0), ggml_view_2d(cl, VC[L], 512, k, VC[L]->nb[1], (size_t) pos * VC[L]->nb[1]))); } else { ggml_build_forward_expand(gl, ggml_cpy(cl, ggml_cont(cl, ggml_transpose(cl, ggml_view_2d(cl, ins[L], 512, k, ins[L]->nb[1], 0))), ggml_view_2d(cl, VC[L], k, 512, VC[L]->nb[1], (size_t) pos * VC[L]->nb[0]))); } }
                    if (!GA_KV) { GA_KV = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); } ggml_gallocr_alloc_graph(GA_KV, gl);
                    std::vector<ggml_fp16_t> col((size_t) 576 * k);
                    for (int L = 0; L < 61; ++L) { for (int j = 0; j < k; ++j) { std::memcpy(col.data() + (size_t) j * 576, lat.data() + ((size_t) j * 61 + L) * 576, 576 * 2); } ggml_backend_tensor_set(ins[L], col.data(), 0, col.size() * 2); }
                    ggml_backend_graph_compute(be, gl); ggml_free(cl);
                    const double tgs = secs(tg0); GR_STRETCH++; GR_TOK += k; spec_time += tgs;
                    std::fprintf(stderr, "  [geometry] %d tokens from stored walk %u@%u, no pass (%.2fs)\n", k, cr, cp, tgs);
                    { const int got = s_emit - 1 + k; std::fprintf(stderr, "  [live] %d tokens in %.1fs = %.2f tok/s\n", got, spec_time, got / std::max(spec_time, 1e-9)); }
                    bool gstop = false;
                    for (int j = 0; j < k && !gstop; ++j) { emit(s_emit++, rt[j], tgs / k, -1); hist.push_back(rt[j]); if (CHAT) { note(rt[j]); } gstop = rt[j] == 1 || s_emit >= n_new; if (rt[j] == 1) { turn_end = "eos"; } }
                    pos += k; tok = rt.back(); GEO_LAST = -1; GEO_NR.clear(); GEO_NRE.clear(); GEO_NRS.clear(); GEO_FAM.clear();
                    GEO_SEED.clear(); for (int j = k + 1; j <= k + 7; ++j) { auto it = GIDX[60].find(gkey(cr, cp + (uint32_t) j)); if (it == GIDX[60].end() || it->second >= GMEM[60].size()) { break; } const GEntry & e = GMEM[60][it->second]; if (e.nxt[0] < 0 || e.runid != cr) { break; } GEO_SEED.push_back(e.nxt[0]); }   /* the checkpoint pass verifies the walk's next tokens */
                    if (gstop) { break; }
                    continue;
                }
            }
            // draft sources, in order: this conversation's own text, then R1's saved outputs (--draft-corpus); longest recent n-gram match wins
            auto lookup = [&](const std::vector<int32_t> & src, int lim) {
                for (int ng = 8; ng >= TEXT_MIN && draft.empty(); --ng) {     // longest shared context first (shared openers point into the wrong topic otherwise); a 1-token match drafts junk
                    if ((int) hist.size() < ng) { continue; }
                    for (int i = lim - ng - 1; i >= 0; --i) {
                        bool m = true; for (int k = 0; k < ng; ++k) { if (src[i + k] != hist[hist.size() - ng + k]) { m = false; break; } }
                        if (m) { for (int k = i + ng; k < (int) src.size() && src[k] >= 0 && (int) draft.size() < std::min(dk, TEXT_CAP); ++k) { draft.push_back(src[k]); } if (!draft.empty()) { draft_ng = ng; break; } }
                    }
                }
            };
            bool geo_seeded = false; if (!GEO_SEED.empty()) { draft = GEO_SEED; GEO_SEED.clear(); geo_seeded = true; }   /* after a geometry run: verify the walk's next tokens */
            const bool cold = !G_MISSROW.empty() && G_MISSROW[0] * 2 > 58;   /* the last pass found nothing stored near row 0: a wrong draft row now costs its own expert bytes */
            if (cold) { COLD_PASSES++; }
            if (draft.empty() && (!cold || COLD_TEXT)) { lookup(hist, (int) hist.size());
            if (draft.empty() && !corpus.empty()) { lookup(corpus, (int) corpus.size() + 1); } }
            bool geo_draft = false, geo_next = false, fam_draft = false;
            if (geo_seeded) { geo_next = true; GN_USED++; }
            else if (GEO_LAST >= 0 && GEO_LAST < (int) GEO_NR.size() && GEO_NR[GEO_LAST][0] == tok && GEO_NR[GEO_LAST][1] >= 0) {   // state converged on a stored state whose own next token agrees with the head: replay what followed it (takes priority over the text match)
                draft.clear(); for (int k = 1; k < 16 && GEO_NR[GEO_LAST][k] >= 0 && (int) draft.size() < draft_k; ++k) { draft.push_back(GEO_NR[GEO_LAST][k]); }
                geo_next = true; GN_USED++;
                std::fprintf(stderr, "  [continuation] last row %d matched; drafting:", GEO_LAST); for (int32_t t : draft) { std::fprintf(stderr, " %d", t); } std::fprintf(stderr, "\n");
            }
            if (!geo_next && GEO_LAST >= 0 && GEO_LAST < (int) GEO_FAM.size() && !GEO_FAM[GEO_LAST].empty()) {   // no exact continuation: the row's family votes a span, question stretches realised for this prompt
                const std::vector<int32_t> fd = fam_vote(GEO_FAM[GEO_LAST], tok, OURQ, std::min(draft_k, cold ? COLD_DRAFT : 15), cold ? COLD_SHARE : FAM_SHARE);
                if (!fd.empty()) { draft = fd; geo_next = true; fam_draft = true; FAM_USED++; std::fprintf(stderr, "  [family] L%d, %zu members near row %d; drafting:", GEO_FAML[GEO_LAST], GEO_FAM[GEO_LAST].size(), GEO_LAST); for (int32_t t : draft) { std::fprintf(stderr, " %d", t); } std::fprintf(stderr, "\n"); }
            }
            if (draft.empty() && DR_ON && (int64_t) GEO_X.size() == E) {              // no text to copy: draft the next two tokens from the state's geometry
                std::vector<float> P((size_t) 2 * E); dr_pred(GEO_X.data(), 0, P.data()); dr_pred(GEO_X.data(), 1, P.data() + E);
                const auto hm = head_top1(P, 2);
                if (hm.second[0] >= DR_MARGIN) { draft.push_back(hm.first[0]); if (hm.second[1] >= DR_MARGIN) { draft.push_back(hm.first[1]); } geo_draft = true; DR_USED++; }
            }
            std::vector<int32_t> in = { (int32_t) tok }; in.insert(in.end(), draft.begin(), draft.end());
            G_POS = (uint32_t) hist.size() - 1; GEO_NR.assign(in.size(), { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }); G_MISSROW.assign(in.size(), 0); GEO_FAM.assign(in.size(), std::vector<FamM>()); GEO_FAML.assign(in.size(), -1); set_rows(in, hist.size() >= 2 ? hist[hist.size() - 2] : -1); mem_queue();
            reset(); t0 = clk::now(); ALLPOS = true; forward(in, pos, logits); ALLPOS = false; n_passes++; n_drafted += (int) draft.size();
            std::vector<float> all = logits; size_t VV = all.size() / in.size();
            std::vector<int> outs;
            for (size_t i = 0; i < in.size(); ++i) {
                const int o = pick_tok(all.data() + (size_t) i * VV, (size_t) VV); outs.push_back(o);
                if (!(i + 1 < in.size() && in[i + 1] == o)) { break; }
            }
            const double ts = secs(t0); spec_time += ts; { const int got = s_emit - 1 + (int) outs.size(); std::fprintf(stderr, "  [live] %d tokens in %.1fs = %.2f tok/s\n", got, spec_time, got / std::max(spec_time, 1e-9)); }
            if (DUMPF && (int64_t) HMID.size() == (int64_t) E * (int64_t) in.size() && all.size() == VV * in.size()) {   // one record per confirmed row: step, input token, output token, mid state, final state
                std::vector<ggml_fp16_t> f16(2 * E);
                for (size_t k = 0; k < outs.size(); ++k) {
                    const int32_t rec[3] = { (int32_t) (s_emit + (int) k), in[k], outs[k] }; std::fwrite(rec, 4, 3, DUMPF);
                    ggml_fp32_to_fp16_row(HMID.data() + k * E, f16.data(), E); ggml_fp32_to_fp16_row(HFIN.data() + k * E, f16.data() + E, E); std::fwrite(f16.data(), 2, 2 * E, DUMPF); DUMPED++;
                }
                std::fflush(DUMPF);
            }
            if (geo_next) { std::fprintf(stderr, "  [continuation] actual:"); for (int o : outs) { std::fprintf(stderr, " %d", o); } std::fprintf(stderr, "  (%zu of %zu drafted accepted)\n", outs.size() - 1, draft.size()); }
            G_CLOCK++; n_accepted += (int) outs.size() - 1; if (geo_draft) { DR_OK += (int) outs.size() - 1; } if (geo_next) { GN_OK += (int) outs.size() - 1; } if (fam_draft) { FAM_OK += (int) outs.size() - 1; } if (!draft.empty() && !geo_next && !geo_draft) { TD_STATS[draft_ng].first += (int) draft.size(); TD_STATS[draft_ng].second += (int) outs.size() - 1; } GEO_LAST = (int) outs.size() - 1;
            if (GMEM_ON) { for (int L = 3; L <= 60; ++L) { for (auto & ge : GMEM[L]) { if (ge.crun != G_RUNID || ge.nxt[15] >= 0 || ge.nxt[0] == -2) { continue; }
                if (ge.cpos >= G_POS + outs.size() && ge.cpos < G_POS + in.size()) { ge.nxt[0] = -2; ge.used = 0; continue; }   /* a rejected draft row's state: off the path taken, evicted first */   // a rejected draft row: its state is fine, its continuation is unknown
                for (int k = 0; k < 16; ++k) { const size_t q = (size_t) ge.cpos + 1 + k; if (ge.nxt[k] >= 0) { continue; } if (q < hist.size()) { ge.nxt[k] = hist[q]; } else if (q - hist.size() < outs.size()) { ge.nxt[k] = outs[q - hist.size()]; } } } } }
            if (HFIN.size() >= outs.size() * (size_t) E) { GEO_X.assign(HFIN.begin() + (outs.size() - 1) * E, HFIN.begin() + outs.size() * E); }
            if (!draft.empty()) { const int acc = (int) outs.size() - 1; dk = acc == (int) draft.size() ? std::min(draft_k, dk * 2) : (acc * 2 < (int) draft.size() ? std::max(2, dk / 2) : dk); }
            { std::fprintf(stderr, "  [miss] layers missed per row:"); for (int c : G_MISSROW) { std::fprintf(stderr, " %d", c); } std::fprintf(stderr, "\n"); }
            std::fprintf(stderr, "pass %2d: %zu drafted, %zu accepted + 1 from the head | %.2fs -> %.2f tok/s this pass (experts read %.2f upload %.2f | compute %.2f | disk %.2f GB)\n",
                         n_passes, draft.size(), outs.size() - 1, ts, outs.size() / ts, T_EX, T_EXUP, T_C, (B_DISK + B_DISK_W) / 1e9);
            bool stop = false;
            for (size_t k = 0; k < outs.size() && !stop; ++k) {
                logits.assign(all.begin() + k * VV, all.begin() + (k + 1) * VV);
                emit(s_emit++, outs[k], ts / outs.size(), -1); hist.push_back(outs[k]); stop = outs[k] == 1 || s_emit >= n_new; if (outs[k] == 1) { turn_end = "eos"; } if (CHAT) { note(outs[k]); if (CHAT_ANSWER > 0 && !in_think && answer_n >= CHAT_ANSWER) { stop = true; } }
            }
            pos += (int) outs.size(); tok = outs.back();
            if (stop) { break; }
        }
        if (draft_k > 0 && pos >= KVP - 2 - draft_k) { std::fprintf(stderr, "[context full] %d of %lld positions used; the thought was cut here. Raise --ctx (costs VRAM: ~5 MB per position) to let it finish.\n", pos, (long long) KVP); }
        if (draft_k > 0) { std::fprintf(stderr, "draft-lookup: %d passes, %d drafted, %d accepted, %d tokens in %.1fs = %.3f tok/s\n", n_passes, n_drafted, n_accepted, s_emit - 1, spec_time, (s_emit - 1) / spec_time); }
        if (draft_k > 0 && !corpus_path.empty() && hist.size() > gen_from) { FILE * cf = std::fopen(corpus_path.c_str(), "ab"); if (cf) { std::fwrite(hist.data() + gen_from, 4, hist.size() - gen_from, cf); const int32_t sep = -1; std::fwrite(&sep, 4, 1, cf); std::fclose(cf); } }   // R1 learns its own phrasing for next time
        }
        if (CHAT) { if (CHAT_STOP) { turn_end = "stop"; } else if (turn_end == "cap" && !(CHAT_ANSWER > 0 && !in_think && answer_n >= CHAT_ANSWER)) { turn_end = "full"; }
            char eb[160]; std::snprintf(eb, sizeof(eb), "END %s %d %d %.2f %d %lld", turn_end.c_str(), think_n, answer_n, secs(turn_clock), pos, (long long) KVP); chat_send(eb); }
        if (!CHAT) { break; }
    }
    ROW_TOK.clear();
    for (int s = 1; draft_k == 0 && s < n_new && pos < KVP - 1; ++s) {
        reset(); t0 = clk::now();
        forward({ tok }, pos, logits); pos++;
        tok = pick_tok(logits.data(), logits.size());
        const double ts = secs(t0); dec_total += ts;
        std::fprintf(stderr, "token %2d: %.2fs = %.2f tok/s (waited for weights %.2f | experts read %.2f upload %.2f | compute %.2f | head %.2f | disk %.2f GB) -> %d\n",
                     s, ts, 1.0 / ts, T_W, T_EX, T_EXUP, T_C, T_HEAD, (B_DISK + B_DISK_W) / 1e9, tok);
        const int fd = s < (int) forced.size() ? forced[s] : -1; emit(s, tok, ts, fd); if (fd >= 0) { tok = fd; }
        if (tok == 1 && forced.empty()) { break; }
    }
    std::fprintf(jl, "{\"summary\":true,\"prompt_tokens\":%zu,\"generated\":%d,\"decode_tok_s\":%.4f,\"text\":\"%s\"}\n", prompt.size(), dec_n + 1, dec_n ? dec_n / dec_sum : 0.0, esc(text_out).c_str()); std::fclose(jl);
    std::printf("\n\nlog: %s\n", log_path.string().c_str());
    if (DR_ON) { std::fprintf(stderr, "drafter: used on %zu passes, %zu drafted tokens accepted\n", DR_USED, DR_OK); }
    if (!TD_STATS.empty()) { std::fprintf(stderr, "text drafts by shared-context length:"); for (auto & kv : TD_STATS) { std::fprintf(stderr, " %d-gram %d/%d", kv.first, kv.second.second, kv.second.first); } std::fprintf(stderr, " (accepted/drafted)\n"); }
    if (GMEM_ON) { std::fprintf(stderr, "geometry runs: %zu stretches, %zu tokens emitted without a pass\n", GR_STRETCH, GR_TOK); std::fprintf(stderr, "family drafts: %zu passes, %zu tokens accepted; %zu run prompts known; %zu cold passes (no text drafts, family share %.2f)\n", FAM_USED, FAM_OK, RUNP.size(), COLD_PASSES, COLD_SHARE); std::fprintf(stderr, "continuation memory: drafted on %zu passes, %zu tokens accepted; %zu stored states adopted a continuation this run\n", GN_USED, GN_OK, G_ADOPT); }
    if (TRACEF) { std::fclose(TRACEF); } if (MOEDF) { std::fclose(MOEDF); }
    if (DUMPF) { std::fclose(DUMPF); std::fprintf(stderr, "dump-states: %zu positions appended to %s (mid layer %d)\n", DUMPED, dump_path.c_str(), MID_L); }
    if (GMEM_ON) { FILE * f = std::fopen((gmem_path + ".tmp").c_str(), "wb"); size_t ne = 0; if (f) { std::fwrite("GEOA", 1, 4, f); for (int L = 0; L < 61; ++L) { for (auto & ge : GMEM[L]) { const int32_t l32 = L; std::fwrite(&l32, 4, 1, f); std::fwrite(ge.sk.data(), 4, GSK, f); std::fwrite(&ge.off, 8, 1, f); std::fwrite(&ge.used, 4, 1, f); std::fwrite(&ge.pos, 4, 1, f); std::fwrite(&ge.runid, 4, 1, f); std::fwrite(&ge.crun, 4, 1, f); std::fwrite(&ge.cpos, 4, 1, f); std::fwrite(ge.nxt, 4, 16, f); std::fwrite(ge.route, 4, 8, f); std::fwrite(&ge.kvoff, 8, 1, f); ne++; } } std::fclose(f); std::remove(gmem_path.c_str()); std::rename((gmem_path + ".tmp").c_str(), gmem_path.c_str()); } if (GOUTF) { std::fclose(GOUTF); } if (GOUTF32) { std::fclose(GOUTF32); } if (GKVF) { std::fclose(GKVF); }
        if (GPRED_D > 0) { std::fprintf(stderr, "geometry route prediction: %zu layer-steps voted, %zu experts queued (share >= %.2f, depth %d, k %d); prefetch hits by source router/chain/labels/geo: %zu/%zu/%zu/%zu, misses %zu, late %zu, slots full %zu | live: experts used %zu, of which predicted this pass %zu\n", GPRED_STEPS, GPRED_N, (double) GPRED_T, GPRED_D, GPRED_K, (size_t) PF_HIT_SRC[0], (size_t) PF_HIT_SRC[1], (size_t) PF_HIT_SRC[2], (size_t) PF_HIT_SRC[3], (size_t) PF_MISS, (size_t) PF_LATE, (size_t) PF_FULL, GPRED_W, GPRED_WH); } if (SPEC_PROBE || SPEC_READ) { std::fprintf(stderr, "[spec] next-layer route from the pre-MoE state + shared expert (row 0): real top-2 in spec top-2 %zu/%zu (%.0f%%), in spec top-4 %zu/%zu (%.0f%%); spec graphs %.2fs, weight waits %.2fs over %zu layer-steps\n", SPEC_H2, SPEC_N, 100.0 * SPEC_H2 / std::max<size_t>(1, SPEC_N), SPEC_H4, SPEC_N, 100.0 * SPEC_H4 / std::max<size_t>(1, SPEC_N), T_SPEC, T_SPECW, SPEC_LAYERS); if (SPEC_READ) { std::fprintf(stderr, "[spec-read] %zu experts queued ahead; prefetch hits %zu (late %zu), misses %zu, slots full %zu\n", SPEC_Q, (size_t) PF_HIT_SRC[3], (size_t) PF_LATE, (size_t) PF_MISS, (size_t) PF_FULL); std::fprintf(stderr, "[attn-build split] graph %.2fs alloc %.2fs input-set %.2fs\n", T_AB_BUILD, T_AB_ALLOC, T_AB_SET); } } std::fprintf(stderr, "geometry memory: %zu states saved; %zu of %zu row-layers matched a stored state; %zu layer-passes ran with no expert pull; %zu rows skipped their expert read; lookup time %.1fs\n", ne, G_HITROWS, G_ROWS, OMEM_SKIPS, G_ROWSKIP, G_TLOOK); }
    if (OMEM_ON) { FILE * f = std::fopen(omem_path.c_str(), "wb"); if (f) { for (auto & kv : OMEM) { std::fwrite(&kv.first, 8, 1, f); std::fwrite(kv.second.data(), 4, 7168, f); } std::fclose(f); }
        std::fprintf(stderr, "output memory: %zu stored outputs saved; %zu of %zu labelled layer-passes ran with no expert pull\n", OMEM.size(), OMEM_SKIPS, OMEM_LAYERS_SEEN); }
    if (RMEM_ON) { FILE * f = std::fopen(rmem_path.c_str(), "wb"); if (f) { for (auto & kv : RMEM) { std::fwrite(&kv.first, 8, 1, f); std::fwrite(kv.second.data(), 2, 8, f); } std::fclose(f); }
        std::fprintf(stderr, "route memory: %zu labels saved; %zu labelled experts sent to the prefetcher before their state existed\n", RMEM.size(), RMEM_QUEUED); }
    if (PF_ON) { std::fprintf(stderr, "prefetch hits by source: router %zu, geometry chain %zu, route memory %zu; chain experts queued %zu\n", PF_HIT_SRC[0], PF_HIT_SRC[1], PF_HIT_SRC[2], ROUTE_Q); } if (PF_ON) { std::fprintf(stderr, "prefetch: %zu hits (%zu still reading when needed), %zu misses = %.1f%% hit; %.2f GB read ahead | worker: %zu jobs, %zu dropped, %zu too late, %zu no free slot, predict %.1fs, read %.1fs\n", PF_HIT, PF_LATE, PF_MISS, 100.0 * PF_HIT / std::max<size_t>(1, PF_HIT + PF_MISS), PF_BYTES / 1e9, PF_JOBS, PF_DROP, PF_SKIP, PF_FULL, PF_TPRED, PF_TREAD); }
    if (n_new > 1) { std::fprintf(stderr, "decode average: %.3f tok/s\n", dec_n ? dec_n / dec_sum : 0.0); }
    return 0;
}
