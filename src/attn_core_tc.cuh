// attn_core_tc.cuh -- the attention core over the latent cache on tensor cores: block = (row b, 16 heads) with 4 warps; scores
//   S[16 heads x positions] = Q[16 x 576] K^T through mma.m16n8k16 f16 (f32 accumulate), tiles of 32 cached positions in shared
//   (row stride 584 halves against bank conflicts), each warp 8 positions per tile; per-head softmax over the global score row;
//   o[16 x 512] = P[16 x positions] V[positions x 512] through the same mma with V fragments by ldmatrix.trans, each warp 128 dims.
//   Inputs as attn_core: qlat [b][h][512] f32, q [b][h][192] f32 (rope part at 128..191), KC [p][576] f16, VC [p][512] f16;
//   scg [b][h][KVP] f32 scratch. Shared: Q 16 x 584 halves + tile 32 x 584 halves = 56 KB (opt-in).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#define TC_KS 584   // padded row stride (halves) of the Q and K tiles in shared
#define TC_TP 32    // cached positions per tile
__device__ __forceinline__ void mma16816f(float & c0, float & c1, float & c2, float & c3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned & r0, unsigned & r1, const void * smem_row) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(smem_row); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a)); }
__device__ __forceinline__ unsigned pack_h2(float a, float b) { const half2 h = __floats2half2_rn(a, b); return *(const unsigned *) &h; }
__global__ void __launch_bounds__(128) attn_core_tc(const float * __restrict__ qlat, const float * __restrict__ q, const half * __restrict__ KC, const half * __restrict__ VC, float * __restrict__ o, int KVP, int p0, float kq_scale, float * __restrict__ scg) {
    extern __shared__ __align__(16) half tcs[]; half * Qs = tcs; half * Kt = tcs + 16 * TC_KS;   // Q [16][584], K/V tile [32][584]
    const int b = blockIdx.x, hg = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3; const int h0 = hg * 16; const int plim = p0 + b + 1;
    const int HEADS = 128, QD = 192, NOPE = 128;
    // Q of the 16 heads as halves: dims 0..511 from qlat, 512..575 from the rope part of q
    for (int i = threadIdx.x; i < 16 * 576; i += 128) { const int hh = i / 576, d = i % 576; const float v = d < 512 ? qlat[((size_t) b * HEADS + h0 + hh) * 512 + d] : q[((size_t) b * HEADS + h0 + hh) * QD + NOPE + (d - 512)]; Qs[hh * TC_KS + d] = __float2half(v); }
    __syncthreads();
    // ---- scores: per tile of 32 positions, warp w takes positions 8w..8w+7 of the tile; 36 k-steps of 16 dims
    for (int t0 = 0; t0 < plim; t0 += TC_TP) {
        const int np = min(TC_TP, plim - t0); __syncthreads();
        const uint4 * src = (const uint4 *) (KC + (size_t) t0 * 576);
        for (int i = threadIdx.x; i < np * 72; i += 128) { const int r = i / 72, c = i % 72; *(uint4 *) (Kt + r * TC_KS + c * 8) = src[i]; }
        __syncthreads();
        const int n0 = warp * 8; if (n0 < np) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            #pragma unroll 4
            for (int k0 = 0; k0 < 576; k0 += 16) {
                const unsigned a0 = *(const unsigned *) (Qs + g * TC_KS + k0 + 2 * t), a1 = *(const unsigned *) (Qs + (g + 8) * TC_KS + k0 + 2 * t), a2 = *(const unsigned *) (Qs + g * TC_KS + k0 + 8 + 2 * t), a3 = *(const unsigned *) (Qs + (g + 8) * TC_KS + k0 + 8 + 2 * t);
                const int pr = n0 + g < np ? n0 + g : np - 1;   // rows past the tile end read a valid row (their scores are discarded)
                const unsigned b0 = *(const unsigned *) (Kt + pr * TC_KS + k0 + 2 * t), b1 = *(const unsigned *) (Kt + pr * TC_KS + k0 + 8 + 2 * t);
                mma16816f(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            }
            const int p = t0 + n0 + 2 * t;   // c0: head g, position p; c1: head g, p+1; c2: head g+8, p; c3: head g+8, p+1
            if (n0 + 2 * t < np) { scg[((size_t) b * HEADS + h0 + g) * KVP + p] = c0 * kq_scale; scg[((size_t) b * HEADS + h0 + g + 8) * KVP + p] = c2 * kq_scale; }
            if (n0 + 2 * t + 1 < np) { scg[((size_t) b * HEADS + h0 + g) * KVP + p + 1] = c1 * kq_scale; scg[((size_t) b * HEADS + h0 + g + 8) * KVP + p + 1] = c3 * kq_scale; }
        }
    }
    __syncthreads();
    // ---- softmax statistics per head: warp w handles heads 4w..4w+3
    __shared__ float smx[16], sinv[16];
    for (int hh = warp * 4; hh < warp * 4 + 4; ++hh) {
        const float * sc = scg + ((size_t) b * HEADS + h0 + hh) * KVP; float mx = -1e30f; for (int p = lane; p < plim; p += 32) mx = fmaxf(mx, sc[p]); for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        float sum = 0.f; for (int p = lane; p < plim; p += 32) sum += __expf(sc[p] - mx); for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) { smx[hh] = mx; sinv[hh] = 1.f / sum; }
    }
    __syncthreads();
    // ---- o = P V: warp w owns dims 128w..128w+127 (16 n-tiles of 8); k over positions in steps of 16; V fragments by ldmatrix.trans from the tile
    float acc[16][4]; for (int i = 0; i < 16; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    const float mxg = smx[g], invg = sinv[g], mxg8 = smx[g + 8], invg8 = sinv[g + 8]; const float * scg_g = scg + ((size_t) b * HEADS + h0 + g) * KVP; const float * scg_g8 = scg + ((size_t) b * HEADS + h0 + g + 8) * KVP;
    for (int t0 = 0; t0 < plim; t0 += TC_TP) {
        const int np = min(TC_TP, plim - t0); __syncthreads();
        const uint4 * src = (const uint4 *) (VC + (size_t) t0 * 512);
        for (int i = threadIdx.x; i < TC_TP * 64; i += 128) { const int r = i / 64, c = i % 64; *(uint4 *) (Kt + r * TC_KS + c * 8) = r < np ? src[i] : make_uint4(0u, 0u, 0u, 0u); }   // rows past the end are zero (their P is zero too)
        __syncthreads();
        for (int k0 = 0; k0 < np; k0 += 16) {
            // A = P[16 heads][16 positions] from the scores: a0 (g; k0+2t, +1), a1 (g+8; ...), a2 (g; k0+8+2t, +1), a3 (g+8; ...)
            auto pv = [&](const float * s, float mx, float inv, int p) { return (p < np) ? __expf(s[t0 + p] - mx) * inv : 0.f; };
            const unsigned a0 = pack_h2(pv(scg_g, mxg, invg, k0 + 2 * t), pv(scg_g, mxg, invg, k0 + 2 * t + 1)), a1 = pack_h2(pv(scg_g8, mxg8, invg8, k0 + 2 * t), pv(scg_g8, mxg8, invg8, k0 + 2 * t + 1));
            const unsigned a2 = pack_h2(pv(scg_g, mxg, invg, k0 + 8 + 2 * t), pv(scg_g, mxg, invg, k0 + 8 + 2 * t + 1)), a3 = pack_h2(pv(scg_g8, mxg8, invg8, k0 + 8 + 2 * t), pv(scg_g8, mxg8, invg8, k0 + 8 + 2 * t + 1));
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int d0 = warp * 128 + nt * 8;   // B = V[k0..k0+15][d0..d0+7]: lanes 0-7 point at rows k0+lane, lanes 8-15 at rows k0+8+(lane-8), each at column d0 (16 B)
                const int r = k0 + (lane & 15); unsigned b0, b1; ldmatrix_x2_trans(b0, b1, Kt + r * TC_KS + d0);
                mma16816f(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3, b0, b1);
            }
        }
    }
    for (int nt = 0; nt < 16; ++nt) { const int d = warp * 128 + nt * 8 + 2 * t;
        o[((size_t) b * HEADS + h0 + g) * 512 + d] = acc[nt][0]; o[((size_t) b * HEADS + h0 + g) * 512 + d + 1] = acc[nt][1]; o[((size_t) b * HEADS + h0 + g + 8) * 512 + d] = acc[nt][2]; o[((size_t) b * HEADS + h0 + g + 8) * 512 + d + 1] = acc[nt][3]; }
}
