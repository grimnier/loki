// fused_layer.cu -- one DeepSeek-R1 layer (MLA attention + router, shared expert, routed experts, dense FFN) as direct CUDA launches
//   for server_core_fused: the kernels validated in bench_matvec/ (tensor-core mat-vecs per format, attention core, router, MoE block)
//   bound to the engine's buffers. nvcc builds this into fused_layer.obj; the MSVC-built engine links it (see build_server_core_fused.bat).
//   Everything runs on the context's own stream; the engine syncs through fl_fetch / fl_moe / fl_skip, which mirror the outputs to pinned host memory.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "fused_layer.h"
#include "iq2_tables.cuh"
#include "attn_core_tc.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "[fused] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(7); } } while (0)
static const int NB = 8, QK = 256, E = 7168, HEADS = 128, QD = 192, NOPE = 128, LAT = 512, VD = 128, HSH = 2048, HD = 18432;
static const int BLK_Q4 = 144, BLK_Q5 = 176, BLK_Q6 = 210, SROW_Q6 = 212, BLK_Q2 = 84, BLK_Q3 = 110, SROW_Q3 = 112, BLK_IQ2 = 66;
enum { F_Q4K = 0, F_Q6K = 1, F_Q80 = 2, F_Q5K = 3, F_IQ2 = 4, F_Q2K = 5, F_Q3K = 6 };
__device__ __forceinline__ void mma16832(int & c0, int & c1, int & c2, int & c3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
__device__ __forceinline__ void mma16816(int & c0, int & c1, int & c2, int & c3, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3) : "r"(a0), "r"(a1), "r"(b0)); }
__device__ __forceinline__ void scale_min(const uint8_t * sc, int j, int & s, int & m) {
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); } }
__device__ __forceinline__ unsigned q6_unpack4(const uint8_t * blk, int h, int q, int l0) {
    const unsigned ql = *(const unsigned *) (blk + h * 64 + ((q & 1) ? 32 : 0) + l0); const unsigned qh = *(const unsigned *) (blk + 128 + h * 32 + l0);
    const unsigned lo = (q >= 2 ? (ql >> 4) : ql) & 0x0F0F0F0Fu; const unsigned hi = ((qh >> (2 * q)) & 0x03030303u) << 4; return __vsub4(lo | hi, 0x20202020u); }
__device__ __forceinline__ unsigned iq2_unpack4(const uint8_t * sb, int l, int j0, const unsigned * grid_s, const uint8_t * ksig_s) {
    const uint8_t idx = sb[l]; const unsigned aux32 = *(const uint16_t *) (sb + 4) | ((unsigned) *(const uint16_t *) (sb + 6) << 16);
    const unsigned gv = grid_s[idx * 2 + (j0 ? 1 : 0)]; const unsigned sbits = (ksig_s[(aux32 >> (7 * l)) & 127] >> j0) & 15; const unsigned m = ((sbits * 0x204081u) & 0x01010101u) * 0xFFu;
    return __vsub4(gv ^ m, m); }
__device__ __forceinline__ void q3_scales(const uint8_t * sc, unsigned * aux) {   // 16 six-bit scales -> 16 bytes, as dequantize_row_q3_K
    const unsigned a0 = *(const unsigned *) sc, a1 = *(const unsigned *) (sc + 4), tmp = *(const unsigned *) (sc + 8);
    aux[2] = ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4); aux[3] = ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
    aux[0] = (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4); aux[1] = (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4); }
// ---- activations: x -> q8 per 32 with per-32 scale, per-32 sum*d and (optional) per-16 sums; nmat matrices per row at stride mstride
struct QX { int8_t * q; float * d; float * s; float * s16; };
__global__ void quant_x(const float * __restrict__ x, int K, int nmat, int xstride, int mstride, QX o) {
    const int NSUB = K / 32; const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nmat * NB * NSUB) return;
    const int m = i / (NB * NSUB), b = (i / NSUB) % NB, s = i % NSUB; const float * xs = x + (size_t) b * xstride + (size_t) m * mstride + s * 32; float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xs[j]));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f; int s0 = 0, s1 = 0; const size_t r = (size_t) m * NB + b;
    for (int j = 0; j < 32; ++j) { const int q = __float2int_rn(xs[j] * id); o.q[r * (K + 16) + s * 32 + j] = (int8_t) q; if (j < 16) s0 += q; else s1 += q; }
    o.d[r * NSUB + s] = d; if (o.s) o.s[r * NSUB + s] = d * (float) (s0 + s1); if (o.s16) { o.s16[r * (K / 16) + 2 * s] = d * (float) s0; o.s16[r * (K / 16) + 2 * s + 1] = d * (float) s1; }
}
__global__ void quant_xg(const float * __restrict__ x, int K, int nsub, int G, int xstride, int mstride, size_t gstride, QX o) {   // quant_x over G row-blocks: set (g*nsub + h, b) <- x + b*xstride + g*gstride + h*mstride
    const int NSUB = K / 32; const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= G * nsub * NB * NSUB) return;
    const int m = i / (NB * NSUB), b = (i / NSUB) % NB, s = i % NSUB; const int gb = m / nsub, h = m % nsub; const float * xs = x + (size_t) b * xstride + (size_t) gb * gstride + (size_t) h * mstride + s * 32; float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xs[j]));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f; int s0 = 0, s1 = 0; const size_t r = (size_t) m * NB + b;
    for (int j = 0; j < 32; ++j) { const int q = __float2int_rn(xs[j] * id); o.q[r * (K + 16) + s * 32 + j] = (int8_t) q; if (j < 16) s0 += q; else s1 += q; }
    o.d[r * NSUB + s] = d; if (o.s) o.s[r * NSUB + s] = d * (float) (s0 + s1); if (o.s16) { o.s16[r * (K / 16) + 2 * s] = d * (float) s0; o.s16[r * (K / 16) + 2 * s + 1] = d * (float) s1; }
}
// ---- generic tensor-core mat-vec: nmat matrices [rows x K] in format F at stride wstride, each against its own 8-row activation set;
//      K split KS ways across warps into part[((ks*nmat+m)*NB+b)*rows+r]; reduce_parts sums the splits in a fixed order (deterministic)
template <int F, int WPB, int G>
__global__ void __launch_bounds__(WPB * 32) tc_matvec(const uint8_t * __restrict__ W, size_t wstride, int rows, int K, int nmat, const QX x, int xm, int KS, const int * __restrict__ midx, float * __restrict__ part) {   // xm: 1 = activation set per matrix, 0 = one shared set; midx: matrix index per slot (null = identity); G: row-blocks of NB rows per matrix, every weight fragment dequantised once for all of them
    extern __shared__ uint8_t smem[]; const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tiles = rows / 16; const int wid = blockIdx.x * WPB + warp; const int m = midx ? wid % nmat : wid / (tiles * KS), rem = midx ? wid / nmat : wid % (tiles * KS), tile = rem / KS, ks = rem % KS;   // with a slot table (wide MoE) the matrix index runs fastest: adjacent warps share a weight tile through L2
    unsigned * grid_s = (unsigned *) smem; uint8_t * ksig_s = smem + 2048;
    if (F == F_IQ2) {   // sign/grid tables into shared by every thread of the block, before any warp can leave
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid_s[i] = ((const unsigned *) iq2xxs_grid)[i]; for (int i = threadIdx.x; i < 128; i += blockDim.x) ksig_s[i] = ksigns_iq2xs[i]; __syncthreads(); }
    if (m >= nmat) return;
    const int XS = K + 16, NSUB = K / 32, r0 = tile * 16, mx = xm ? m : 0; const uint8_t * Wm = W + (size_t) (midx ? midx[m] : m) * wstride; const int8_t * xq = x.q + (size_t) mx * NB * XS; const float * dx = x.d + (size_t) mx * NB * NSUB; const float * sx = x.s ? x.s + (size_t) mx * NB * NSUB : nullptr; const float * sx16 = x.s16 ? x.s16 + (size_t) mx * NB * (K / 16) : nullptr;
    const size_t XSTEP = (size_t) (xm ? nmat : 1) * NB * XS, DSTEP = (size_t) (xm ? nmat : 1) * NB * NSUB, S16STEP = (size_t) (xm ? nmat : 1) * NB * (K / 16);   // activation set of row-block rb: rb*nmat + m
    float acc[G][4];
    #pragma unroll
    for (int rbk = 0; rbk < G; ++rbk) { acc[rbk][0] = 0.f; acc[rbk][1] = 0.f; acc[rbk][2] = 0.f; acc[rbk][3] = 0.f; }
    if (F == F_Q4K) {
        const int NBLK = K / QK, b0 = ks * NBLK / KS, b1 = (ks + 1) * NBLK / KS; uint8_t * stg2 = smem + warp * 4608;
        auto stage = [&](int blk, uint8_t * dst) { for (int i = lane; i < 144; i += 32) { const uint8_t * src = Wm + (size_t) (r0 + i / 9) * NBLK * BLK_Q4 + (size_t) blk * BLK_Q4 + (i % 9) * 16; const unsigned sd = (unsigned) __cvta_generic_to_shared(dst + i * 16); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(src)); } asm volatile("cp.async.commit_group;"); };
        if (b0 < b1) { stage(b0, stg2); asm volatile("cp.async.wait_group 0;"); } __syncwarp();
        for (int blk = b0; blk < b1; ++blk) {
            uint8_t * stg = stg2 + (blk & 1) * 2304; if (blk + 1 < b1) stage(blk + 1, stg2 + ((blk + 1) & 1) * 2304);
            const uint8_t * ra = stg + g * BLK_Q4; const uint8_t * rb = stg + (g + 8) * BLK_Q4;
            const float da = __half2float(*(const half *) ra), dma = __half2float(*(const half *) (ra + 2)), db = __half2float(*(const half *) rb), dmb = __half2float(*(const half *) (rb + 2));
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int sa0, ma0, sa1, ma1, sb0, mb0, sb1, mb1; scale_min(ra + 4, 2 * c, sa0, ma0); scale_min(ra + 4, 2 * c + 1, sa1, ma1); scale_min(rb + 4, 2 * c, sb0, mb0); scale_min(rb + 4, 2 * c + 1, sb1, mb1);
                const int sub = blk * 8 + 2 * c;
                const unsigned wa0 = *(const unsigned *) (ra + 16 + c * 32 + t * 4), wa2 = *(const unsigned *) (ra + 16 + c * 32 + 16 + t * 4), wb1 = *(const unsigned *) (rb + 16 + c * 32 + t * 4), wb3 = *(const unsigned *) (rb + 16 + c * 32 + 16 + t * 4);
                const unsigned al0 = wa0 & 0x0F0F0F0Fu, al1 = wb1 & 0x0F0F0F0Fu, al2 = wa2 & 0x0F0F0F0Fu, al3 = wb3 & 0x0F0F0F0Fu, ah0 = (wa0 >> 4) & 0x0F0F0F0Fu, ah1 = (wb1 >> 4) & 0x0F0F0F0Fu, ah2 = (wa2 >> 4) & 0x0F0F0F0Fu, ah3 = (wb3 >> 4) & 0x0F0F0F0Fu;
                #pragma unroll
                for (int rbk = 0; rbk < G; ++rbk) {
                    const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP; const float * sxr = sx + rbk * DSTEP; int cl[4] = { 0, 0, 0, 0 }, ch[4] = { 0, 0, 0, 0 };
                    const unsigned b0l = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + t * 4)), b1l = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + 16 + t * 4));
                    const unsigned b0h = __ldg((const unsigned *) (xr + (size_t) g * XS + (sub + 1) * 32 + t * 4)), b1h = __ldg((const unsigned *) (xr + (size_t) g * XS + (sub + 1) * 32 + 16 + t * 4));
                    mma16832(cl[0], cl[1], cl[2], cl[3], al0, al1, al2, al3, b0l, b1l);
                    mma16832(ch[0], ch[1], ch[2], ch[3], ah0, ah1, ah2, ah3, b0h, b1h);
                    const float dxl0 = dxr[(2 * t) * NSUB + sub], dxl1 = dxr[(2 * t + 1) * NSUB + sub], dxh0 = dxr[(2 * t) * NSUB + sub + 1], dxh1 = dxr[(2 * t + 1) * NSUB + sub + 1];
                    const float sxl0 = sxr[(2 * t) * NSUB + sub], sxl1 = sxr[(2 * t + 1) * NSUB + sub], sxh0 = sxr[(2 * t) * NSUB + sub + 1], sxh1 = sxr[(2 * t + 1) * NSUB + sub + 1];
                    acc[rbk][0] += da * ((float) sa0 * dxl0 * (float) cl[0] + (float) sa1 * dxh0 * (float) ch[0]) - dma * ((float) ma0 * sxl0 + (float) ma1 * sxh0);
                    acc[rbk][1] += da * ((float) sa0 * dxl1 * (float) cl[1] + (float) sa1 * dxh1 * (float) ch[1]) - dma * ((float) ma0 * sxl1 + (float) ma1 * sxh1);
                    acc[rbk][2] += db * ((float) sb0 * dxl0 * (float) cl[2] + (float) sb1 * dxh0 * (float) ch[2]) - dmb * ((float) mb0 * sxl0 + (float) mb1 * sxh0);
                    acc[rbk][3] += db * ((float) sb0 * dxl1 * (float) cl[3] + (float) sb1 * dxh1 * (float) ch[3]) - dmb * ((float) mb0 * sxl1 + (float) mb1 * sxh1);
                }
            }
            asm volatile("cp.async.wait_group 0;"); __syncwarp();
        }
    } else if (F == F_Q5K) {   // 176-byte blocks: d, dmin, 12 scales, 32 qh, 128 qs; cp.async double buffer
        const int NBLK = K / QK, b0 = ks * NBLK / KS, b1 = (ks + 1) * NBLK / KS; uint8_t * stg2 = smem + warp * (2 * 16 * BLK_Q5);
        auto stage = [&](int blk, uint8_t * dst) { for (int i = lane; i < 16 * 11; i += 32) { const uint8_t * src = Wm + (size_t) (r0 + i / 11) * NBLK * BLK_Q5 + (size_t) blk * BLK_Q5 + (i % 11) * 16; const unsigned sd = (unsigned) __cvta_generic_to_shared(dst + i * 16); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sd), "l"(src)); } asm volatile("cp.async.commit_group;"); };
        if (b0 < b1) { stage(b0, stg2); asm volatile("cp.async.wait_group 0;"); } __syncwarp();
        for (int blk = b0; blk < b1; ++blk) {
            uint8_t * stg = stg2 + (blk & 1) * (16 * BLK_Q5); if (blk + 1 < b1) { stage(blk + 1, stg2 + ((blk + 1) & 1) * (16 * BLK_Q5)); }
            const uint8_t * ra = stg + g * BLK_Q5; const uint8_t * rb = stg + (g + 8) * BLK_Q5;
            const float da = __half2float(*(const half *) ra), dma = __half2float(*(const half *) (ra + 2)), db = __half2float(*(const half *) rb), dmb = __half2float(*(const half *) (rb + 2));
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int sa0, ma0, sa1, ma1, sb0, mb0, sb1, mb1; scale_min(ra + 4, 2 * c, sa0, ma0); scale_min(ra + 4, 2 * c + 1, sa1, ma1); scale_min(rb + 4, 2 * c, sb0, mb0); scale_min(rb + 4, 2 * c + 1, sb1, mb1);
                const int sub = blk * 8 + 2 * c;
                const unsigned qa0 = *(const unsigned *) (ra + 48 + c * 32 + t * 4), qa2 = *(const unsigned *) (ra + 48 + c * 32 + 16 + t * 4), ha0 = *(const unsigned *) (ra + 16 + t * 4), ha2 = *(const unsigned *) (ra + 32 + t * 4);
                const unsigned qb1 = *(const unsigned *) (rb + 48 + c * 32 + t * 4), qb3 = *(const unsigned *) (rb + 48 + c * 32 + 16 + t * 4), hb1 = *(const unsigned *) (rb + 16 + t * 4), hb3 = *(const unsigned *) (rb + 32 + t * 4);
                const unsigned ml = 0x01010101u << (2 * c), mh = 0x01010101u << (2 * c + 1);
                const unsigned al0 = (qa0 & 0x0F0F0F0Fu) | (((ha0 & ml) >> (2 * c)) << 4), al1 = (qb1 & 0x0F0F0F0Fu) | (((hb1 & ml) >> (2 * c)) << 4), al2 = (qa2 & 0x0F0F0F0Fu) | (((ha2 & ml) >> (2 * c)) << 4), al3 = (qb3 & 0x0F0F0F0Fu) | (((hb3 & ml) >> (2 * c)) << 4);
                const unsigned ah0 = ((qa0 >> 4) & 0x0F0F0F0Fu) | (((ha0 & mh) >> (2 * c + 1)) << 4), ah1 = ((qb1 >> 4) & 0x0F0F0F0Fu) | (((hb1 & mh) >> (2 * c + 1)) << 4), ah2 = ((qa2 >> 4) & 0x0F0F0F0Fu) | (((ha2 & mh) >> (2 * c + 1)) << 4), ah3 = ((qb3 >> 4) & 0x0F0F0F0Fu) | (((hb3 & mh) >> (2 * c + 1)) << 4);
                #pragma unroll
                for (int rbk = 0; rbk < G; ++rbk) {
                    const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP; const float * sxr = sx + rbk * DSTEP; int cl[4] = { 0, 0, 0, 0 }, ch[4] = { 0, 0, 0, 0 };
                    const unsigned b0l = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + t * 4)), b1l = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + 16 + t * 4));
                    const unsigned b0h = __ldg((const unsigned *) (xr + (size_t) g * XS + (sub + 1) * 32 + t * 4)), b1h = __ldg((const unsigned *) (xr + (size_t) g * XS + (sub + 1) * 32 + 16 + t * 4));
                    mma16832(cl[0], cl[1], cl[2], cl[3], al0, al1, al2, al3, b0l, b1l);
                    mma16832(ch[0], ch[1], ch[2], ch[3], ah0, ah1, ah2, ah3, b0h, b1h);
                    const float dxl0 = dxr[(2 * t) * NSUB + sub], dxl1 = dxr[(2 * t + 1) * NSUB + sub], dxh0 = dxr[(2 * t) * NSUB + sub + 1], dxh1 = dxr[(2 * t + 1) * NSUB + sub + 1];
                    const float sxl0 = sxr[(2 * t) * NSUB + sub], sxl1 = sxr[(2 * t + 1) * NSUB + sub], sxh0 = sxr[(2 * t) * NSUB + sub + 1], sxh1 = sxr[(2 * t + 1) * NSUB + sub + 1];
                    acc[rbk][0] += da * ((float) sa0 * dxl0 * (float) cl[0] + (float) sa1 * dxh0 * (float) ch[0]) - dma * ((float) ma0 * sxl0 + (float) ma1 * sxh0);
                    acc[rbk][1] += da * ((float) sa0 * dxl1 * (float) cl[1] + (float) sa1 * dxh1 * (float) ch[1]) - dma * ((float) ma0 * sxl1 + (float) ma1 * sxh1);
                    acc[rbk][2] += db * ((float) sb0 * dxl0 * (float) cl[2] + (float) sb1 * dxh0 * (float) ch[2]) - dmb * ((float) mb0 * sxl0 + (float) mb1 * sxh0);
                    acc[rbk][3] += db * ((float) sb0 * dxl1 * (float) cl[3] + (float) sb1 * dxh1 * (float) ch[3]) - dmb * ((float) mb0 * sxl1 + (float) mb1 * sxh1);
                }
            }
            asm volatile("cp.async.wait_group 0;"); __syncwarp();
        }
    } else if (F == F_Q6K) {   // 210-byte blocks, 16-bit staging at stride 212
        const int NBLK = K / QK, b0 = ks * NBLK / KS, b1 = (ks + 1) * NBLK / KS; uint8_t * stg = smem + warp * (16 * SROW_Q6);
        for (int blk = b0; blk < b1; ++blk) {
            __syncwarp();
            for (int i = lane; i < 16 * 105; i += 32) { const int r = i / 105, h = i % 105; *(uint16_t *) (stg + r * SROW_Q6 + h * 2) = *(const uint16_t *) (Wm + (size_t) (r0 + r) * NBLK * BLK_Q6 + (size_t) blk * BLK_Q6 + h * 2); }
            __syncwarp();
            const uint8_t * ra = stg + g * SROW_Q6; const uint8_t * rb = stg + (g + 8) * SROW_Q6;
            const float da = __half2float(*(const half *) (ra + 208)), db = __half2float(*(const half *) (rb + 208)); const int8_t * sca = (const int8_t *) (ra + 192); const int8_t * scb = (const int8_t *) (rb + 192);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const int q = gg >> 1, l0 = (gg & 1) * 16 + t * 4; const int e0 = blk * 256 + h * 128 + gg * 16; const int sub = e0 >> 5;
                    const unsigned a0 = q6_unpack4(ra, h, q, l0), a1 = q6_unpack4(rb, h, q, l0); const float fa = da * (float) sca[h * 8 + gg], fb = db * (float) scb[h * 8 + gg];
                    #pragma unroll
                    for (int rbk = 0; rbk < G; ++rbk) {
                        const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP; const unsigned xb0 = __ldg((const unsigned *) (xr + (size_t) g * XS + e0 + t * 4));
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0; mma16816(c0, c1, c2, c3, a0, a1, xb0); const float dx0 = dxr[(2 * t) * NSUB + sub], dx1 = dxr[(2 * t + 1) * NSUB + sub];
                        acc[rbk][0] += fa * dx0 * (float) c0; acc[rbk][1] += fa * dx1 * (float) c1; acc[rbk][2] += fb * dx0 * (float) c2; acc[rbk][3] += fb * dx1 * (float) c3;
                    }
                }
            }
        }
    } else if (F == F_Q80) {   // 34-byte blocks of 32 (half d + 32 int8), staged shifted by 2 so the int8 sit 4-byte aligned
        const int NB32 = K / 32, b0 = ks * NB32 / KS, b1 = (ks + 1) * NB32 / KS; uint8_t * stg = smem + warp * (16 * 36);
        for (int blk = b0; blk < b1; ++blk) {
            __syncwarp();
            for (int i = lane; i < 16 * 17; i += 32) { const int r = i / 17, h = i % 17; *(uint16_t *) (stg + r * 36 + 2 + h * 2) = *(const uint16_t *) (Wm + (size_t) (r0 + r) * NB32 * 34 + (size_t) blk * 34 + h * 2); }
            __syncwarp();
            const uint8_t * ra = stg + g * 36; const uint8_t * rb = stg + (g + 8) * 36; const float da = __half2float(*(const half *) (ra + 2)), db = __half2float(*(const half *) (rb + 2));
            #pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                const int e0 = blk * 32 + gg * 16; const unsigned a0 = *(const unsigned *) (ra + 4 + gg * 16 + t * 4), a1 = *(const unsigned *) (rb + 4 + gg * 16 + t * 4);
                #pragma unroll
                for (int rbk = 0; rbk < G; ++rbk) {
                    const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP;
                    const unsigned xb0 = __ldg((const unsigned *) (xr + (size_t) g * XS + e0 + t * 4)); int c0 = 0, c1 = 0, c2 = 0, c3 = 0; mma16816(c0, c1, c2, c3, a0, a1, xb0);
                    const float dx0 = dxr[(2 * t) * NSUB + blk], dx1 = dxr[(2 * t + 1) * NSUB + blk];
                    acc[rbk][0] += da * dx0 * (float) c0; acc[rbk][1] += da * dx1 * (float) c1; acc[rbk][2] += db * dx0 * (float) c2; acc[rbk][3] += db * dx1 * (float) c3;
                }
            }
        }
    } else if (F == F_IQ2) {   // 66-byte blocks, staged four at a time with 8-byte loads (K split in groups of 4 blocks)
        const int NBLK = K / QK, NG = NBLK / 4, g0 = ks * NG / KS, g1 = (ks + 1) * NG / KS; uint8_t * stg = smem + 2176 + warp * (16 * 264);
        for (int bg = g0 * 4; bg < g1 * 4; bg += 4) {
            __syncwarp();
            for (int i = lane; i < 16 * 33; i += 32) { const int r = i / 33, h = i % 33; *(uint2 *) (stg + r * 264 + h * 8) = __ldg((const uint2 *) (Wm + (size_t) (r0 + r) * NBLK * BLK_IQ2 + (size_t) bg * BLK_IQ2 + h * 8)); }
            __syncwarp();
            for (int j = 0; j < 4; ++j) {
                const int blk = bg + j; const uint8_t * ra = stg + g * 264 + j * 66; const uint8_t * rb = stg + (g + 8) * 264 + j * 66; const float da = __half2float(*(const half *) ra), db = __half2float(*(const half *) rb);
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const uint8_t * sa = ra + 2 + s * 8; const uint8_t * sb = rb + 2 + s * 8; const int sub = blk * 8 + s; const int l0 = t >> 1, j0 = (t & 1) * 4;
                    const unsigned a0 = iq2_unpack4(sa, l0, j0, grid_s, ksig_s), a1 = iq2_unpack4(sb, l0, j0, grid_s, ksig_s), a2 = iq2_unpack4(sa, l0 + 2, j0, grid_s, ksig_s), a3 = iq2_unpack4(sb, l0 + 2, j0, grid_s, ksig_s);
                    const unsigned auxa = *(const uint16_t *) (sa + 4) | ((unsigned) *(const uint16_t *) (sa + 6) << 16), auxb = *(const uint16_t *) (sb + 4) | ((unsigned) *(const uint16_t *) (sb + 6) << 16);
                    const float fa = da * (0.5f + (float) (auxa >> 28)) * 0.25f, fb = db * (0.5f + (float) (auxb >> 28)) * 0.25f;
                    #pragma unroll
                    for (int rbk = 0; rbk < G; ++rbk) {
                        const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP;
                        const unsigned xb0 = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + t * 4)), xb1 = __ldg((const unsigned *) (xr + (size_t) g * XS + sub * 32 + 16 + t * 4));
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0; mma16832(c0, c1, c2, c3, a0, a1, a2, a3, xb0, xb1);
                        const float dx0 = dxr[(2 * t) * NSUB + sub], dx1 = dxr[(2 * t + 1) * NSUB + sub];
                        acc[rbk][0] += fa * dx0 * (float) c0; acc[rbk][1] += fa * dx1 * (float) c1; acc[rbk][2] += fb * dx0 * (float) c2; acc[rbk][3] += fb * dx1 * (float) c3;
                    }
                }
            }
        }
    } else if (F == F_Q2K) {   // 84-byte blocks (16 scale/min bytes, 64 qs, d, dmin), 4-byte staging
        const int NBLK = K / QK, b0 = ks * NBLK / KS, b1 = (ks + 1) * NBLK / KS; uint8_t * stg = smem + warp * (16 * BLK_Q2);
        for (int blk = b0; blk < b1; ++blk) {
            __syncwarp();
            for (int i = lane; i < 16 * 21; i += 32) { const int r = i / 21, h = i % 21; *(uint32_t *) (stg + r * BLK_Q2 + h * 4) = __ldg((const uint32_t *) (Wm + (size_t) (r0 + r) * NBLK * BLK_Q2 + (size_t) blk * BLK_Q2 + h * 4)); }
            __syncwarp();
            const uint8_t * ra = stg + g * BLK_Q2; const uint8_t * rb = stg + (g + 8) * BLK_Q2;
            const float da = __half2float(*(const half *) (ra + 80)), dma = __half2float(*(const half *) (ra + 82)), db = __half2float(*(const half *) (rb + 80)), dmb = __half2float(*(const half *) (rb + 82));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const int shift = (gg >> 1) * 2, bo = 16 + h * 32 + (gg & 1) * 16 + t * 4; const int e0 = blk * 256 + h * 128 + gg * 16; const int sub = e0 >> 5, g16 = e0 >> 4;
                    const unsigned a0 = (*(const unsigned *) (ra + bo) >> shift) & 0x03030303u, a1 = (*(const unsigned *) (rb + bo) >> shift) & 0x03030303u;
                    const int sca = ra[h * 8 + gg], scb = rb[h * 8 + gg]; const float fa = da * (float) (sca & 0xF), fb = db * (float) (scb & 0xF), mla = dma * (float) (sca >> 4), mlb = dmb * (float) (scb >> 4);
                    #pragma unroll
                    for (int rbk = 0; rbk < G; ++rbk) {
                        const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP; const float * s16r = sx16 + rbk * S16STEP;
                        const unsigned xb0 = __ldg((const unsigned *) (xr + (size_t) g * XS + e0 + t * 4)); int c0 = 0, c1 = 0, c2 = 0, c3 = 0; mma16816(c0, c1, c2, c3, a0, a1, xb0);
                        const float dx0 = dxr[(2 * t) * NSUB + sub], dx1 = dxr[(2 * t + 1) * NSUB + sub], sx0 = s16r[(2 * t) * (K / 16) + g16], sx1 = s16r[(2 * t + 1) * (K / 16) + g16];
                        acc[rbk][0] += fa * dx0 * (float) c0 - mla * sx0; acc[rbk][1] += fa * dx1 * (float) c1 - mla * sx1; acc[rbk][2] += fb * dx0 * (float) c2 - mlb * sx0; acc[rbk][3] += fb * dx1 * (float) c3 - mlb * sx1;
                    }
                }
            }
        }
    } else {   // F_Q3K: 110-byte blocks (32 hmask, 64 qs, 12 scales, d), 16-bit staging at stride 112; value = low2 - 4 where the high bit is clear
        const int NBLK = K / QK, b0 = ks * NBLK / KS, b1 = (ks + 1) * NBLK / KS; uint8_t * stg = smem + warp * (16 * SROW_Q3);
        for (int blk = b0; blk < b1; ++blk) {
            __syncwarp();
            for (int i = lane; i < 16 * 55; i += 32) { const int r = i / 55, h = i % 55; *(uint16_t *) (stg + r * SROW_Q3 + h * 2) = *(const uint16_t *) (Wm + (size_t) (r0 + r) * NBLK * BLK_Q3 + (size_t) blk * BLK_Q3 + h * 2); }
            __syncwarp();
            const uint8_t * ra = stg + g * SROW_Q3; const uint8_t * rb = stg + (g + 8) * SROW_Q3;
            const float da = __half2float(*(const half *) (ra + 108)), db = __half2float(*(const half *) (rb + 108)); unsigned sa[4], sb[4]; q3_scales(ra + 96, sa); q3_scales(rb + 96, sb);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const int j = gg >> 1, s16 = gg & 1; const int qo = 32 + h * 32 + s16 * 16 + t * 4, ho = s16 * 16 + t * 4, bit = h * 4 + j; const int e0 = blk * 256 + h * 128 + gg * 16; const int sub = e0 >> 5;
                    const unsigned qa = (*(const unsigned *) (ra + qo) >> (2 * j)) & 0x03030303u, qb = (*(const unsigned *) (rb + qo) >> (2 * j)) & 0x03030303u;
                    const unsigned ha = ((*(const unsigned *) (ra + ho) >> bit) & 0x01010101u) ^ 0x01010101u, hb = ((*(const unsigned *) (rb + ho) >> bit) & 0x01010101u) ^ 0x01010101u;
                    const unsigned a0 = __vsub4(qa, ha << 2), a1 = __vsub4(qb, hb << 2);
                    const int is = h * 8 + gg; const float fa = da * (float) ((int) ((sa[is >> 2] >> (8 * (is & 3))) & 0xFFu) - 32), fb = db * (float) ((int) ((sb[is >> 2] >> (8 * (is & 3))) & 0xFFu) - 32);
                    #pragma unroll
                    for (int rbk = 0; rbk < G; ++rbk) {
                        const int8_t * xr = xq + rbk * XSTEP; const float * dxr = dx + rbk * DSTEP; const unsigned xb0 = __ldg((const unsigned *) (xr + (size_t) g * XS + e0 + t * 4));
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0; mma16816(c0, c1, c2, c3, a0, a1, xb0); const float dx0 = dxr[(2 * t) * NSUB + sub], dx1 = dxr[(2 * t + 1) * NSUB + sub];
                        acc[rbk][0] += fa * dx0 * (float) c0; acc[rbk][1] += fa * dx1 * (float) c1; acc[rbk][2] += fb * dx0 * (float) c2; acc[rbk][3] += fb * dx1 * (float) c3;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int rbk = 0; rbk < G; ++rbk) {
        float * yp = part + (((size_t) ks * G + rbk) * nmat + m) * NB * rows;
        yp[(size_t) (2 * t) * rows + r0 + g] = acc[rbk][0]; yp[(size_t) (2 * t) * rows + r0 + g + 8] = acc[rbk][2]; yp[(size_t) (2 * t + 1) * rows + r0 + g] = acc[rbk][1]; yp[(size_t) (2 * t + 1) * rows + r0 + g + 8] = acc[rbk][3];
    }
}
__global__ void reduce_g(const float * __restrict__ part, float * __restrict__ y, int rows, int nmat, int KS, int G) {   // G groups of NB rows: y[(g*NB+b)][m*rows + r], part as [k][g*nmat+m][b][rows]; G = 1 is reduce_parts
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= G * NB * nmat * rows) return; const int row = i / (nmat * rows), mr = i % (nmat * rows), m = mr / rows, r = mr % rows, g = row / NB, b = row % NB;
    float v = 0.f; for (int k = 0; k < KS; ++k) v += part[(((size_t) k * G * nmat + (size_t) g * nmat + m) * NB + b) * rows + r]; y[(size_t) row * nmat * rows + mr] = v;
}
__global__ void reduce_parts(const float * __restrict__ part, float * __restrict__ y, int rows, int nmat, int KS) {   // y[b][m*rows + r]
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= NB * nmat * rows) return; const int b = i / (nmat * rows), mr = i % (nmat * rows), m = mr / rows, r = mr % rows;
    float v = 0.f; for (int k = 0; k < KS; ++k) v += part[((size_t) k * nmat + m) * NB * rows + (size_t) b * rows + r]; y[(size_t) b * nmat * rows + mr] = v;
}
__global__ void rmsnorm_rows(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y, int D, int xstride, int ystride) {   // one block per row
    extern __shared__ float red[]; const int b = blockIdx.x; const float * xr = x + (size_t) b * xstride; float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss; __syncthreads();
    if (threadIdx.x < 32) { float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f; for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); if (threadIdx.x == 0) red[0] = v; } __syncthreads();
    const float inv = rsqrtf(red[0] / (float) D + 1e-6f); for (int i = threadIdx.x; i < D; i += blockDim.x) y[(size_t) b * ystride + i] = xr[i] * inv * w[i];
}
// YaRN rope (ggml mode 0, adjacent pairs) in place on v[b][m][off .. off+64) with per-row positions
__global__ void rope64(float * __restrict__ v, int nmat, int vstride, int off, const int * __restrict__ pos, float freq_base, float freq_scale, float ext, float attn_factor, float lo, float hi, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * nmat * 32) return; const int b = i / (nmat * 32), m = (i / 32) % nmat, pr = i % 32;
    float * p = v + (size_t) b * (size_t) nmat * vstride + (size_t) m * vstride + off + pr * 2;
    const float theta_scale = powf(freq_base, -2.0f / 64.f), mscale = ext != 0.f ? attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale)) : attn_factor;
    const float theta_base = (float) pos[b] * powf(theta_scale, (float) pr); const float ramp = 1.0f - fminf(fmaxf(((float) pr - lo) / fmaxf(0.001f, hi - lo), 0.f), 1.f); const float mix = ramp * ext;
    const float theta = freq_scale * theta_base * (1.0f - mix) + theta_base * mix; const float c = cosf(theta) * mscale, s = sinf(theta) * mscale; const float x0 = p[0], x1 = p[1]; p[0] = x0 * c - x1 * s; p[1] = x0 * s + x1 * c;
}
// cache write for nrows new positions p0..: KC[p][576] f16 = (kvc normed, kpe roped), VC[p][512] f16 = kvc; ccat[b][576] f32 keeps the latents for the host
__global__ void cache_write(const float * __restrict__ kvc, const float * __restrict__ kvpe, half * __restrict__ KC, half * __restrict__ VC, int p0, int nrows, float * __restrict__ ccat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nrows * 576) return; const int b = i / 576, d = i % 576;
    const float v = d < 512 ? kvc[(size_t) b * 512 + d] : kvpe[(size_t) b * 576 + d]; KC[(size_t) (p0 + b) * 576 + d] = __float2half(v); if (d < 512) VC[(size_t) (p0 + b) * 512 + d] = __float2half(v); ccat[(size_t) b * 576 + d] = v;
}
// attention core: block = (row b, 16 heads); scores over the cache up to the row's own position, softmax, o = P x VC
__global__ void __launch_bounds__(512) attn_core(const float * __restrict__ qlat, const float * __restrict__ q, const half * __restrict__ KC, const half * __restrict__ VC, float * __restrict__ o, int KVP, int p0, float kq_scale, float * __restrict__ scg) {
    extern __shared__ float sm[]; float * krow = sm;
    const int b = blockIdx.x, hg = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = hg * 16 + warp; const int plim = p0 + b + 1; float * sc = scg + ((size_t) b * HEADS + h) * KVP;   /* this warp's score row */
    float qr[18]; for (int j = 0; j < 18; ++j) { const int d = lane + 32 * j; qr[j] = d < 512 ? qlat[((size_t) b * HEADS + h) * 512 + d] : q[((size_t) b * HEADS + h) * QD + NOPE + (d - 512)]; }
    for (int p = 0; p < plim; ++p) {
        __syncthreads(); for (int d = threadIdx.x; d < 576; d += blockDim.x) krow[d] = __half2float(KC[(size_t) p * 576 + d]); __syncthreads();
        float dot = 0.f; for (int j = 0; j < 18; ++j) dot += qr[j] * krow[lane + 32 * j];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off); if (lane == 0) sc[p] = dot * kq_scale;
    }
    __syncthreads();
    __syncwarp(); float mx = -1e30f; for (int p = lane; p < plim; p += 32) mx = fmaxf(mx, sc[p]); for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.f; for (int p = lane; p < plim; p += 32) { const float e = __expf(sc[p] - mx); sc[p] = e; sum += e; } for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    __syncwarp(); const float inv = 1.f / sum; float acc[16]; for (int j = 0; j < 16; ++j) acc[j] = 0.f;
    for (int p = 0; p < plim; ++p) {
        __syncthreads(); for (int d = threadIdx.x; d < 512; d += blockDim.x) krow[d] = __half2float(VC[(size_t) p * 512 + d]); __syncthreads();
        const float pw = sc[p] * inv; for (int j = 0; j < 16; ++j) acc[j] += pw * krow[lane + 32 * j];
    }
    for (int j = 0; j < 16; ++j) o[((size_t) b * HEADS + h) * 512 + lane + 32 * j] = acc[j];
}
__global__ void add_rows(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ y, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a[i] + b[i]; }
// byte movers on the compute engine (mapped pinned host memory or device memory on either side): 16-byte lanes, grid-stride; copy1 for odd tails
__global__ void copy16(const uint4 * __restrict__ s, uint4 * __restrict__ d, size_t n) { for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) d[i] = s[i]; }
__global__ void copy1(const uint8_t * __restrict__ s, uint8_t * __restrict__ d, size_t n) { const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (i < n) d[i] = s[i]; }
// router: probs = sigmoid(gate_inp[e] . ncur[b]) one warp per (row, expert); then per row top-8 of probs + bias, weights normalised x2.5
__global__ void router_probs(const float * __restrict__ ncur, const void * __restrict__ gate_inp, int f16, float * __restrict__ probs, int rows) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (wid >= rows * 256) return; const int b = wid / 256, e = wid % 256;
    const float4 * x4 = (const float4 *) (ncur + (size_t) b * E); float acc = 0.f;
    if (f16) { const uint4 * w8 = (const uint4 *) ((const half *) gate_inp + (size_t) e * E);
        for (int i = lane; i < E / 8; i += 32) { const uint4 wv = __ldg(w8 + i); const half2 * h2 = (const half2 *) &wv; const float4 a = x4[2 * i], c2 = x4[2 * i + 1]; const float2 w0 = __half22float2(h2[0]), w1 = __half22float2(h2[1]), w2 = __half22float2(h2[2]), w3 = __half22float2(h2[3]);
            acc += a.x * w0.x + a.y * w0.y + a.z * w1.x + a.w * w1.y + c2.x * w2.x + c2.y * w2.y + c2.z * w3.x + c2.w * w3.y; } }
    else { const float4 * w4 = (const float4 *) ((const float *) gate_inp + (size_t) e * E); for (int i = lane; i < E / 4; i += 32) { const float4 a = x4[i], w = __ldg(w4 + i); acc += a.x * w.x + a.y * w.y + a.z * w.z + a.w * w.w; } }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o); if (lane == 0) probs[b * 256 + e] = 1.f / (1.f + __expf(-acc));
}
__global__ void router_topk(const float * __restrict__ probs, const float * __restrict__ bias, int * __restrict__ sel, float * __restrict__ wts) {   // one block of 256 threads per row
    __shared__ float sc[256]; __shared__ int taken[8]; __shared__ float wv[8]; __shared__ int wi[8]; const int b = blockIdx.x, e = threadIdx.x; sc[e] = probs[b * 256 + e] + bias[e]; __syncthreads();
    for (int k = 0; k < 8; ++k) {
        float v = sc[e]; int idx = e; for (int j = 0; j < k; ++j) if (taken[j] == e) v = -1e30f;
        for (int o = 16; o > 0; o >>= 1) { const float ov = __shfl_xor_sync(0xffffffffu, v, o); const int oi = __shfl_xor_sync(0xffffffffu, idx, o); if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; } }
        if ((e & 31) == 0) { wv[e >> 5] = v; wi[e >> 5] = idx; } __syncthreads();
        if (e == 0) { float bv = wv[0]; int bi = wi[0]; for (int w = 1; w < 8; ++w) if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) { bv = wv[w]; bi = wi[w]; } taken[k] = bi; } __syncthreads();
    }
    if (e < 8) { float sum = 0.f; for (int k = 0; k < 8; ++k) sum += probs[b * 256 + taken[k]]; sum = fmaxf(sum, 6.103515625e-5f); sel[b * 8 + e] = taken[e]; wts[b * 8 + e] = probs[b * 256 + taken[e]] / sum * 2.5f; }
}
// h = silu(sum_ks gate) * sum_ks up per (slot, row), quantised to q8 per 32 with per-32 and per-16 sums
__global__ void swiglu_q8(const float * __restrict__ gp, const float * __restrict__ up, int KS, int nslot, int H, QX o) {
    const int NSUB = H / 32; const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nslot * NB * NSUB) return; const int m = i / (NB * NSUB), b = (i / NSUB) % NB, s = i % NSUB;
    float hv[32]; float amax = 0.f;
    for (int j = 0; j < 32; ++j) { const int e = s * 32 + j; float gs = 0.f, us = 0.f; for (int k = 0; k < KS; ++k) { const size_t idx = (((size_t) k * nslot + m) * NB + b) * H + e; gs += gp[idx]; us += up[idx]; } const float v = gs / (1.f + __expf(-gs)) * us; hv[j] = v; amax = fmaxf(amax, fabsf(v)); }
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f; int s0 = 0, s1 = 0; const size_t r = (size_t) m * NB + b;
    for (int j = 0; j < 32; ++j) { const int q = __float2int_rn(hv[j] * id); o.q[r * (H + 16) + s * 32 + j] = (int8_t) q; if (j < 16) s0 += q; else s1 += q; }
    o.d[r * NSUB + s] = d; if (o.s) o.s[r * NSUB + s] = d * (float) (s0 + s1); if (o.s16) { o.s16[r * (H / 16) + 2 * s] = d * (float) s0; o.s16[r * (H / 16) + 2 * s + 1] = d * (float) s1; }
}
// hspec = fi + sum_ks shared-down partials (the speculative input of layer L+1)
// ---- wide pass: rows gathered by index into q8 slot sets; residual + shared expert per row; per-(row, expert) outputs; deterministic combine
__global__ void quant_rows(const float * __restrict__ X, const int * __restrict__ rowidx, int nsl, QX o) {   // slot set s, member i <- row rowidx[s*8+i] of X (E wide), -1 = zero row
    const int NSUB = E / 32; const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= nsl * NB * NSUB) return; const int si = t / NSUB, s = t % NSUB; const int r = rowidx[si];
    float xs[32]; if (r >= 0) { const float * src = X + (size_t) r * E + s * 32; for (int j = 0; j < 32; ++j) xs[j] = src[j]; } else { for (int j = 0; j < 32; ++j) xs[j] = 0.f; }
    float amax = 0.f; for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xs[j])); const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f; int s0 = 0, s1 = 0;
    for (int j = 0; j < 32; ++j) { const int q = __float2int_rn(xs[j] * id); o.q[(size_t) si * (E + 16) + s * 32 + j] = (int8_t) q; if (j < 16) s0 += q; else s1 += q; }
    o.d[(size_t) si * NSUB + s] = d; if (o.s) o.s[(size_t) si * NSUB + s] = d * (float) (s0 + s1); if (o.s16) { o.s16[(size_t) si * (E / 16) + 2 * s] = d * (float) s0; o.s16[(size_t) si * (E / 16) + 2 * s + 1] = d * (float) s1; }
}
__global__ void base_rows(const float * __restrict__ shp, int KS, const float * __restrict__ fi, float * __restrict__ base_all, int row0, int n, int R) {   // base[row0 + i] = ffn_inp[i] + shared expert output[i]; shp as [k][R rows][E]
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= n * E) return; const int i = t / E, e = t % E; float v = fi[t]; for (int k = 0; k < KS; ++k) v += shp[((size_t) k * R + i) * E + e]; base_all[(size_t) (row0 + i) * E + e] = v;
}
__global__ void scatter_out(const float * __restrict__ part, int KS, int nsl, const int * __restrict__ rowidx, const int * __restrict__ kidx, const float * __restrict__ wk, int KE, float * __restrict__ out_all) {   // out[(row, k)] = w * expert output
    const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (t >= (size_t) nsl * NB * E) return; const int si = (int) (t / E), e = (int) (t % E); const int r = rowidx[si]; if (r < 0) return;
    float v = 0.f; for (int k = 0; k < KS; ++k) v += part[((size_t) k * nsl * NB + si) * E + e]; out_all[((size_t) r * KE + kidx[si]) * E + e] = wk[si] * v;
}
__global__ void wide_finish(const float * __restrict__ base_all, const float * __restrict__ out_all, int KE, int n, float * __restrict__ h, float * __restrict__ moe) {   // h = base + sum_k out, moe = sum_k out (fixed order)
    const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (t >= (size_t) n * E) return; const int r = (int) (t / E), e = (int) (t % E); float m = 0.f; for (int k = 0; k < KE; ++k) m += out_all[((size_t) r * KE + k) * E + e]; moe[t] = m; h[t] = base_all[t] + m;
}
__global__ void wide_skip(const float * __restrict__ base_all, const float * __restrict__ mo, int n, float * __restrict__ h) { const size_t t = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (t < (size_t) n * E) h[t] = base_all[t] + mo[t]; }
__global__ void hspec_add(const float * __restrict__ shp, int KS, const float * __restrict__ fi, float * __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= NB * E) return; const int b = i / E, e = i % E; float v = fi[i]; for (int k = 0; k < KS; ++k) v += shp[((size_t) k * NB + b) * E + e]; y[i] = v;
}
// combine: routed = sum over slots of (row weight of that expert) x sum_ks part; lo2 = routed + mo + shared + fi; moe = routed
__global__ void moe_combine(const float * __restrict__ part, int KS, int nslot, const int * __restrict__ ids, const float * __restrict__ w, int KE, const float * __restrict__ shp, int KSsh, const float * __restrict__ mo, const float * __restrict__ fi, float * __restrict__ lo2, float * __restrict__ moe) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= NB * E) return; const int b = i / E, e = i % E; float v = 0.f;
    if (part) { for (int m = 0; m < nslot; ++m) { float wt = 0.f; for (int k = 0; k < KE; ++k) if (ids[b * KE + k] == m) wt += w[b * KE + k]; if (wt == 0.f) continue; float p = 0.f; for (int k = 0; k < KS; ++k) p += part[(((size_t) k * nslot + m) * NB + b) * E + e]; v += wt * p; } }
    if (moe) moe[i] = v; if (mo) v += mo[i];
    float sh = 0.f; for (int k = 0; k < KSsh; ++k) sh += shp[((size_t) k * NB + b) * E + e];
    lo2[i] = v + sh + fi[i];
}
// geometry sketch: y[b][j] = P[j] . x[b] (F32 projection, warp per (row, dim)); z = rms_norm(y, 1e-20) / 16; sc[b][k] = T[k] . z[b] (F16 table)
__global__ void sketch_proj(const float * __restrict__ P, const float * __restrict__ x, float * __restrict__ y, int rows) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (wid >= rows * 256) return; const int b = wid / 256, j = wid % 256;
    const float4 * x4 = (const float4 *) (x + (size_t) b * E); const float4 * p4 = (const float4 *) (P + (size_t) j * E); float acc = 0.f;
    for (int i = lane; i < E / 4; i += 32) { const float4 a = x4[i], p = __ldg(p4 + i); acc += a.x * p.x + a.y * p.y + a.z * p.z + a.w * p.w; }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o); if (lane == 0) y[b * 256 + j] = acc;
}
__global__ void sketch_norm(const float * __restrict__ y, float * __restrict__ z) {
    __shared__ float red[8]; const int b = blockIdx.x, q = threadIdx.x; const float v = y[b * 256 + q]; float ss = v * v;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o); if ((q & 31) == 0) red[q >> 5] = ss; __syncthreads();
    float tot = 0.f; for (int k = 0; k < 8; ++k) tot += red[k]; z[b * 256 + q] = v * rsqrtf(tot / 256.f + 1e-20f) * 0.0625f;
}
__global__ void sketch_scan(const half * __restrict__ T, int K, const float * __restrict__ z, float * __restrict__ sc, int rows) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31; if (wid >= rows * K) return; const int b = wid / K, k = wid % K;
    const half2 * t2 = (const half2 *) (T + (size_t) k * 256); const float2 * z2 = (const float2 *) (z + (size_t) b * 256); float acc = 0.f;
    for (int i = lane; i < 128; i += 32) { const float2 t = __half22float2(t2[i]), v = z2[i]; acc += t.x * v.x + t.y * v.y; }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o); if (lane == 0) sc[(size_t) b * K + k] = acc;
}
// ================================================================ host side
struct FLCtx {
    cudaStream_t st; cudaEvent_t ea0, ea1, eaM, eaS, em0, em1; int kvp, nsm, sccap, N, scK, spec_on; float rlo, rhi, freq_base, freq_scale, attn_factor, kq_scale; bool pinned; double ms_main, ms_spec;
    float * hin, * cur, * qa, * qan, * q, * kvpe, * kvc, * qlat, * o, * v, * attn, * ffn_inp, * ncur, * probs, * wts, * ccat, * ccat2, * hspec, * fi2, * ncur2, * probs2, * wts2, * skt, * sxn, * sc, * lo2, * moe, * mo, * w;
    float * part, * partH, * partA, * partB, * partS, * partSD, * scg; int * sel, * sel2, * pos, * ids; QX xcur, xqa, xqn, xo, xv, xh;
    int * h_sel, * h_sel2, * h_ids, * h_pos; float * h_wts, * h_wts2, * h_ncur, * h_fi, * h_skt, * h_sc, * h_ccat, * h_h, * h_moe, * h_mo, * h_hin, * h_w;
    size_t smQ4, smQ5, smQ6, smQ8, smI2, smQ2, smQ3, smCore, smTC; int attn_tc; double ms_attn, ms_moe; size_t n_zc, n_ce; int said;
    int wmax, wke; float * ncur_all, * base_all, * out_all; int * wrows[2], * wkidx[2], * wmidx[2]; float * wkw[2]; int * h_wrows[2], * h_wkidx[2], * h_wmidx[2]; float * h_wkw[2], * h_wh, * h_wmoe, * h_wmo; QX xw; cudaEvent_t wev[2]; int * mz, * mh;   // wide pass: slot tables and a done-event per batch parity; mz/mh: matrix index tables for G-group launches
    cudaEvent_t eP[3]; int dbg; double ms_p[4]; int sc_rows; int last_ksd, last_nexp;   // last_ksd/last_nexp: the narrow MoE's partial layout, for fl_moe_outs (attribution)   // sc_rows: rows per attention-core launch (score buffer sc_rows x HEADS x ctx floats, <= 64 MB)   // FUSED_WIDE_DEBUG: attention phases (projections | attention core | o-proj+norm | router/dense) in ms
};
static const size_t PARTA_F = (size_t) 2 * 72 * NB * HSH, PARTB_F = (size_t) 72 * NB * E, XH_SLOTS = 72;
static const int GMAX = 8, RMAX = GMAX * NB;   // rows per attention launch set: up to 8 groups of 8 (wide passes); a narrow pass is one group
template <class T> static T * dmalloc(size_t n) { T * p; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> static T * hmalloc(FLCtx * c, size_t n) { T * p = nullptr; if (c->pinned && cudaHostAlloc((void **) &p, n * sizeof(T), cudaHostAllocMapped) == cudaSuccess) { return p; } cudaGetLastError(); c->pinned = false; return (T *) std::calloc(n, sizeof(T)); }
static QX qx_alloc(int K, int nmat, bool s16) { QX q; const size_t r = (size_t) nmat * NB; q.q = dmalloc<int8_t>(r * (K + 16)); q.d = dmalloc<float>(r * (K / 32)); q.s = dmalloc<float>(r * (K / 32)); q.s16 = s16 ? dmalloc<float>(r * (K / 16)) : nullptr; return q; }
static inline int blocks_for(size_t n) { return (int) ((n + 255) / 256); }
static inline int round_g(int g) { return g <= 1 ? 1 : g <= 2 ? 2 : g <= 4 ? 4 : 8; }   // row-block count of a launch set -> kernel instantiation (pad row-blocks are computed and dropped)
static int pick_ks(const FLCtx * c, int tiles, int nmat, int max_ks, int cap) {   // enough warps to cover the SMs; KS x nmat bounded by the partial buffer
    const int target = c->nsm * 24; int ks = (target + tiles * nmat - 1) / (tiles * nmat); ks = std::max(1, std::min(max_ks, ks)); while (ks > 1 && ks * nmat > cap) { --ks; } return ks; }
static void dcopy(FLCtx * c, void * dst, const void * src, size_t bytes) {   // any-to-any copy by kernel on the layer stream
    if (bytes == 0) return; const bool al = (((uintptr_t) dst | (uintptr_t) src | bytes) & 15) == 0;
    if (al) { const size_t n = bytes / 16; const int blocks = (int) std::min<size_t>((n + 255) / 256, 2048); copy16<<<blocks, 256, 0, c->st>>>((const uint4 *) src, (uint4 *) dst, n); }
    else { const int blocks = (int) std::min<size_t>((bytes + 255) / 256, 65535); copy1<<<blocks, 256, 0, c->st>>>((const uint8_t *) src, (uint8_t *) dst, bytes); }
}
static void * mapped(FLCtx * c, const void * host) { void * dp = nullptr; if (cudaHostGetDevicePointer(&dp, (void *) host, 0) != cudaSuccess) { cudaGetLastError(); return nullptr; } return dp; }
extern "C" void fl_h2d(FLCtx * c, void * dst, const void * src, size_t bytes) {
    void * dp = mapped(c, src);
    if (dp) { dcopy(c, dst, dp, bytes); c->n_zc += bytes; } else { CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->st)); c->n_ce += bytes; }
    if (!c->said && (c->n_zc + c->n_ce) > ((size_t) 64 << 20)) { c->said = 1; std::fprintf(stderr, "[fused] host->device by kernel (mapped pinned) %.0f MB, by copy engine %.0f MB so far\n", c->n_zc / 1048576.0, c->n_ce / 1048576.0); }
}
static void d2h(FLCtx * c, void * host_dst, const void * dev_src, size_t bytes) { void * dp = mapped(c, host_dst); if (dp) { dcopy(c, dp, dev_src, bytes); } else { CK(cudaMemcpyAsync(host_dst, dev_src, bytes, cudaMemcpyDeviceToHost, c->st)); } }
extern "C" void fl_wait_event(FLCtx * c, cudaEvent_t ev) { CK(cudaStreamWaitEvent(c->st, ev, 0)); }
static const int * MIDX = nullptr;   // set around the wide expert launches: slot -> expert index within the arena batch
template <int F> static void mv_t(FLCtx * c, const void * W, size_t wstride, int rows, int K, int nmat, const QX & x, int xm, int KS, float * part, size_t sm, int G) {
    const int tiles = rows / 16; const int blocks = (tiles * KS * nmat + 3) / 4;
    switch (G) {
        case 1: tc_matvec<F, 4, 1><<<blocks, 128, sm, c->st>>>((const uint8_t *) W, wstride, rows, K, nmat, x, xm, KS, MIDX, part); break;
        case 2: tc_matvec<F, 4, 2><<<blocks, 128, sm, c->st>>>((const uint8_t *) W, wstride, rows, K, nmat, x, xm, KS, MIDX, part); break;
        case 4: tc_matvec<F, 4, 4><<<blocks, 128, sm, c->st>>>((const uint8_t *) W, wstride, rows, K, nmat, x, xm, KS, MIDX, part); break;
        default: tc_matvec<F, 4, 8><<<blocks, 128, sm, c->st>>>((const uint8_t *) W, wstride, rows, K, nmat, x, xm, KS, MIDX, part); break; } }
// xm = 1: matrix m takes activation set m (per-head projections, per-expert down); xm = 0: every matrix takes set 0 (routed gate/up on the shared input)
// G: row-blocks per matrix (1, 2, 4 or 8), each with its own activation set rb*nmat + m and output slot (ks*G + rb)*nmat + m
static void mv(FLCtx * c, int F, const void * W, size_t wstride, int rows, int K, int nmat, const QX & x, int xm, int KS, float * part, int G = 1) {
    switch (F) {
        case F_Q4K: mv_t<F_Q4K>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ4, G); break; case F_Q5K: mv_t<F_Q5K>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ5, G); break;
        case F_Q6K: mv_t<F_Q6K>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ6, G); break; case F_Q80: mv_t<F_Q80>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ8, G); break;
        case F_IQ2: mv_t<F_IQ2>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smI2, G); break; case F_Q2K: mv_t<F_Q2K>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ2, G); break;
        default: mv_t<F_Q3K>(c, W, wstride, rows, K, nmat, x, xm, KS, part, c->smQ3, G); break; }
}
static float fbits(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
extern "C" FLCtx * fl_init(int kvp, int sc_cap) {
    FLCtx * c = new FLCtx(); std::memset(c, 0, sizeof(FLCtx)); c->kvp = kvp; c->sccap = std::max(64, sc_cap); c->pinned = true;
    CK(cudaDeviceGetAttribute(&c->nsm, cudaDevAttrMultiProcessorCount, 0)); CK(cudaStreamCreateWithFlags(&c->st, cudaStreamNonBlocking));
    CK(cudaEventCreate(&c->ea0)); CK(cudaEventCreate(&c->ea1)); CK(cudaEventCreate(&c->eaM)); CK(cudaEventCreate(&c->eaS)); CK(cudaEventCreate(&c->em0)); CK(cudaEventCreate(&c->em1));
    c->freq_base = fbits(0x461c4000); c->freq_scale = fbits(0x3ccccccd); c->attn_factor = fbits(0x3f3b035c); c->kq_scale = fbits(0x3e0a7ab9);
    c->rlo = floorf(64.f * logf(4096.f / (32.f * 2.f * 3.14159265f)) / (2.f * logf(c->freq_base))); c->rhi = fminf(63.f, ceilf(64.f * logf(4096.f / (1.f * 2.f * 3.14159265f)) / (2.f * logf(c->freq_base))));
    c->smQ4 = 4 * 4608; c->smQ5 = 4 * 2 * 16 * BLK_Q5; c->smQ6 = 4 * 16 * SROW_Q6; c->smQ8 = 4 * 16 * 36; c->smI2 = 2176 + 4 * 16 * 264; c->smQ2 = 4 * 16 * BLK_Q2; c->smQ3 = 4 * 16 * SROW_Q3; c->smCore = (size_t) 640 * 4;   /* scores live in global memory (scg): any context length */
    { int r = (int) (((size_t) 64 << 20) / ((size_t) HEADS * kvp * 4)); r = (r / NB) * NB; c->sc_rows = std::max(NB, std::min(RMAX, r)); } c->scg = dmalloc<float>((size_t) c->sc_rows * HEADS * kvp); c->smTC = (size_t) 48 * TC_KS * 2; CK(cudaFuncSetAttribute(attn_core_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) c->smTC));
    { const char * e = std::getenv("FUSED_ATTN_TC"); c->attn_tc = (e && e[0] == '0') ? 0 : 1; } c->dbg = std::getenv("FUSED_WIDE_DEBUG") ? 1 : 0; for (int i = 0; i < 3; ++i) { CK(cudaEventCreate(&c->eP[i])); }
    const size_t R = (size_t) RMAX * E;   /* row buffers hold up to RMAX rows (wide passes); a narrow pass uses the first NB */
    c->hin = dmalloc<float>(R); c->cur = dmalloc<float>(R); c->qa = dmalloc<float>(RMAX * 1536); c->qan = dmalloc<float>(RMAX * 1536); c->q = dmalloc<float>((size_t) RMAX * HEADS * QD); c->kvpe = dmalloc<float>(RMAX * 576); c->kvc = dmalloc<float>(RMAX * 512);
    c->qlat = dmalloc<float>((size_t) RMAX * HEADS * 512); c->o = dmalloc<float>((size_t) RMAX * HEADS * 512); c->v = dmalloc<float>((size_t) RMAX * HEADS * VD); c->attn = dmalloc<float>(R); c->ffn_inp = dmalloc<float>(R); c->ncur = dmalloc<float>(R);
    c->probs = dmalloc<float>(RMAX * 256); c->wts = dmalloc<float>(RMAX * 8); c->sel = dmalloc<int>(RMAX * 8); c->ccat = dmalloc<float>(RMAX * 576); c->ccat2 = dmalloc<float>(NB * 576); c->hspec = dmalloc<float>(R); c->fi2 = dmalloc<float>(R); c->ncur2 = dmalloc<float>(R);
    c->probs2 = dmalloc<float>(NB * 256); c->wts2 = dmalloc<float>(NB * 8); c->sel2 = dmalloc<int>(NB * 8); c->skt = dmalloc<float>(RMAX * 256); c->sxn = dmalloc<float>(RMAX * 256); c->sc = dmalloc<float>((size_t) RMAX * c->sccap);
    c->lo2 = dmalloc<float>(R); c->moe = dmalloc<float>(R); c->mo = dmalloc<float>(R); c->pos = dmalloc<int>(RMAX); c->ids = dmalloc<int>(NB * 8); c->w = dmalloc<float>(NB * 8);
    c->part = dmalloc<float>(std::max((size_t) 8 * RMAX * E, (size_t) RMAX * HEADS * QD)); c->partH = dmalloc<float>((size_t) HEADS * RMAX * 512); c->partA = dmalloc<float>(PARTA_F); c->partB = dmalloc<float>(PARTB_F); c->partS = dmalloc<float>((size_t) 2 * 7 * RMAX * HSH); c->partSD = dmalloc<float>((size_t) 2 * RMAX * E);
    c->xcur = qx_alloc(E, GMAX, false); c->xqa = qx_alloc(1536, GMAX, false); c->xqn = qx_alloc(128, GMAX * HEADS, false); c->xo = qx_alloc(512, GMAX * HEADS, false); c->xv = qx_alloc(HEADS * VD, GMAX, false);
    { QX q; q.q = dmalloc<int8_t>(std::max((size_t) XH_SLOTS * NB * (HSH + 16), (size_t) RMAX * (HD + 16))); q.d = dmalloc<float>(std::max((size_t) XH_SLOTS * NB * (HSH / 32), (size_t) RMAX * (HD / 32))); q.s = dmalloc<float>(std::max((size_t) XH_SLOTS * NB * (HSH / 32), (size_t) RMAX * (HD / 32))); q.s16 = dmalloc<float>(std::max((size_t) XH_SLOTS * NB * (HSH / 16), (size_t) RMAX * (HD / 16))); c->xh = q; }
    c->h_sel = hmalloc<int>(c, RMAX * 8); c->h_sel2 = hmalloc<int>(c, NB * 8); c->h_ids = hmalloc<int>(c, NB * 8); c->h_pos = hmalloc<int>(c, RMAX); c->h_wts = hmalloc<float>(c, RMAX * 8); c->h_wts2 = hmalloc<float>(c, NB * 8); c->h_w = hmalloc<float>(c, NB * 8);
    c->h_ncur = hmalloc<float>(c, R); c->h_fi = hmalloc<float>(c, R); c->h_skt = hmalloc<float>(c, RMAX * 256); c->h_sc = hmalloc<float>(c, (size_t) RMAX * c->sccap); c->h_ccat = hmalloc<float>(c, RMAX * 576);
    { std::vector<int> z(GMAX, 0), hh(GMAX * HEADS); for (int i = 0; i < GMAX * HEADS; ++i) { hh[i] = i % HEADS; } c->mz = dmalloc<int>(GMAX); c->mh = dmalloc<int>(GMAX * HEADS); CK(cudaMemcpy(c->mz, z.data(), GMAX * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(c->mh, hh.data(), GMAX * HEADS * 4, cudaMemcpyHostToDevice)); } c->h_h = hmalloc<float>(c, R); c->h_moe = hmalloc<float>(c, R); c->h_mo = hmalloc<float>(c, R); c->h_hin = hmalloc<float>(c, R);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    std::fprintf(stderr, "[fused] context: %d SMs, ctx %d, attention core %s, host mirrors %s, VRAM free %.0f MB after the activation buffers\n", c->nsm, kvp, c->attn_tc ? "tensor-core (32-position tiles)" : "per-position", c->pinned ? "pinned" : "pageable (pinned pool exhausted)", fr / 1048576.0);
    return c;
}
// stage 1-3 on x (up to RMAX rows = G groups of NB) with weights W; outputs fi, nc, probs/sel/wts (MoE) or lo2 (dense); writes the cache at pos0.. for N rows.
// Every weight matrix is launched once over all G row-blocks (matrix index tables mz/mh), so it is read from VRAM once per launch set and the
// other row-blocks find it in L2; row buffers are row-major over R = G*NB rows, per-head activation sets are [g][h][b]. G = 1 is the old path.
static void mvg(FLCtx * c, int F, const void * W, size_t wstride, int rows, int K, int nmat, const QX & x, int KS, float * part, int G) {
    mv(c, F, W, wstride, rows, K, nmat, x, 1, KS, part, G);   // one launch: every weight fragment dequantised once for the G row-blocks
}
static QX qx_at(const QX & x, int K, size_t blocks) { QX q = x; q.q += blocks * NB * (K + 16); q.d += blocks * NB * (K / 32); if (q.s) { q.s += blocks * NB * (K / 32); } if (q.s16) { q.s16 += blocks * NB * (K / 16); } return q; }
static void attn_block(FLCtx * c, const FLW * W, const float * x, int N, int pos0, void * KC, void * VC, float * ccat, float * fi, float * nc, float * probs, int * sel, float * wts, int dense) {
    cudaStream_t st = c->st; const int G = round_g((N + NB - 1) / NB), R = G * NB;
    rmsnorm_rows<<<R, 256, 128, st>>>(x, (const float *) W->attn_norm, c->cur, E, E, E); quant_x<<<blocks_for((size_t) R * (E / 32)), 256, 0, st>>>(c->cur, E, G, E, NB * E, c->xcur);
    mvg(c, F_Q4K, W->q_a, 0, 1536, E, 1, c->xcur, 7, c->part, G); reduce_g<<<blocks_for((size_t) R * 1536), 256, 0, st>>>(c->part, c->qa, 1536, 1, 7, G);
    rmsnorm_rows<<<R, 256, 128, st>>>(c->qa, (const float *) W->q_a_norm, c->qan, 1536, 1536, 1536); quant_x<<<blocks_for((size_t) R * (1536 / 32)), 256, 0, st>>>(c->qan, 1536, G, 1536, NB * 1536, c->xqa);
    mvg(c, F_Q4K, W->q_b, 0, HEADS * QD, 1536, 1, c->xqa, 1, c->part, G); reduce_g<<<blocks_for((size_t) R * HEADS * QD), 256, 0, st>>>(c->part, c->q, HEADS * QD, 1, 1, G);
    rope64<<<blocks_for((size_t) R * HEADS * 32), 256, 0, st>>>(c->q, HEADS, QD, NOPE, c->pos, c->freq_base, c->freq_scale, 1.0f, c->attn_factor, c->rlo, c->rhi, R);
    mvg(c, F_Q6K, W->kv_a, 0, 576, E, 1, c->xcur, 7, c->part, G); reduce_g<<<blocks_for((size_t) R * 576), 256, 0, st>>>(c->part, c->kvpe, 576, 1, 7, G);
    rmsnorm_rows<<<R, 256, 128, st>>>(c->kvpe, (const float *) W->kv_a_norm, c->kvc, 512, 576, 512);
    rope64<<<blocks_for((size_t) R * 32), 256, 0, st>>>(c->kvpe, 1, 576, 512, c->pos, c->freq_base, c->freq_scale, 1.0f, c->attn_factor, c->rlo, c->rhi, R);
    quant_xg<<<blocks_for((size_t) G * HEADS * NB * 4), 256, 0, st>>>(c->q, 128, HEADS, G, HEADS * QD, QD, (size_t) NB * HEADS * QD, c->xqn);
    mvg(c, F_Q80, W->k_bT, (size_t) 512 * 4 * 34, 512, 128, HEADS, c->xqn, 1, c->partH, G); reduce_g<<<blocks_for((size_t) R * HEADS * 512), 256, 0, st>>>(c->partH, c->qlat, 512, HEADS, 1, G);
    if (c->dbg) { CK(cudaEventRecord(c->eP[0], st)); }
    cache_write<<<blocks_for((size_t) N * 576), 256, 0, st>>>(c->kvc, c->kvpe, (half *) KC, (half *) VC, pos0, N, ccat);
    for (int r0 = 0; r0 < N; r0 += c->sc_rows) {   // the attention core over the set's rows at once (score buffer: sc_rows x HEADS x ctx)
        const int Nr = std::min(c->sc_rows, N - r0); const float * ql = c->qlat + (size_t) r0 * HEADS * 512; const float * qg = c->q + (size_t) r0 * HEADS * QD; float * og = c->o + (size_t) r0 * HEADS * 512;
        if (c->attn_tc) { attn_core_tc<<<dim3(Nr, HEADS / 16), 128, c->smTC, st>>>(ql, qg, (const half *) KC, (const half *) VC, og, c->kvp, pos0 + r0, c->kq_scale, c->scg); }
        else { attn_core<<<dim3(Nr, HEADS / 16), 512, c->smCore, st>>>(ql, qg, (const half *) KC, (const half *) VC, og, c->kvp, pos0 + r0, c->kq_scale, c->scg); }
    }
    quant_xg<<<blocks_for((size_t) G * HEADS * NB * 16), 256, 0, st>>>(c->o, 512, HEADS, G, HEADS * 512, 512, (size_t) NB * HEADS * 512, c->xo);
    if (c->dbg) { CK(cudaEventRecord(c->eP[1], st)); }
    mvg(c, F_Q6K, W->kv_b_v ? (const uint8_t *) W->kv_b : (const uint8_t *) W->kv_b + (size_t) 128 * 420, W->kv_b_v ? (size_t) 128 * 420 : (size_t) 256 * 420, 128, 512, HEADS, c->xo, 1, c->partH, G); reduce_g<<<blocks_for((size_t) R * HEADS * 128), 256, 0, st>>>(c->partH, c->v, 128, HEADS, 1, G);
    quant_x<<<blocks_for((size_t) R * (HEADS * VD / 32)), 256, 0, st>>>(c->v, HEADS * VD, G, HEADS * VD, NB * HEADS * VD, c->xv);
    mvg(c, F_Q4K, W->attn_out, 0, E, HEADS * VD, 1, c->xv, 8, c->part, G); reduce_g<<<blocks_for((size_t) R * E), 256, 0, st>>>(c->part, c->attn, E, 1, 8, G);
    if (c->dbg) { CK(cudaEventRecord(c->eP[2], st)); }
    add_rows<<<blocks_for((size_t) R * E), 256, 0, st>>>(c->attn, x, fi, R * E); rmsnorm_rows<<<R, 256, 128, st>>>(fi, (const float *) W->ffn_norm, nc, E, E, E);
    if (dense) {
        quant_x<<<blocks_for((size_t) R * (E / 32)), 256, 0, st>>>(nc, E, G, E, NB * E, c->xcur);
        mvg(c, F_Q4K, W->d_gate, 0, HD, E, 1, c->xcur, 1, c->partA, G); mvg(c, F_Q4K, W->d_up, 0, HD, E, 1, c->xcur, 1, c->partA + (size_t) R * HD, G);
        swiglu_q8<<<blocks_for((size_t) R * (HD / 32)), 256, 0, st>>>(c->partA, c->partA + (size_t) R * HD, 1, G, HD, c->xh);
        mvg(c, F_Q6K, W->d_down, 0, E, HD, 1, c->xh, 2, c->partB, G); reduce_g<<<blocks_for((size_t) R * E), 256, 0, st>>>(c->partB, c->attn, E, 1, 2, G);
        add_rows<<<blocks_for((size_t) R * E), 256, 0, st>>>(c->attn, fi, c->lo2, R * E);
    } else {
        router_probs<<<blocks_for((size_t) R * 256 * 32), 256, 0, st>>>(nc, W->gate_inp, W->gate_f16, probs, R); router_topk<<<R, 256, 0, st>>>(probs, (const float *) W->probs_b, sel, wts);
    }
}
extern "C" void fl_attn(FLCtx * c, const FLW * W, int N, int pos0, const int * pos, void * KC, void * VC, const float * hin, int dense, int last_row, int wrow) {
    cudaStream_t st = c->st; c->N = N; c->scK = 0; c->spec_on = 0; CK(cudaEventRecord(c->ea0, st)); const int G = round_g((N + NB - 1) / NB), R = G * NB;
    if (hin) { std::memcpy(c->h_hin, hin, (size_t) N * E * 4); fl_h2d(c, c->hin, c->h_hin, (size_t) N * E * 4); if (N < R) { CK(cudaMemsetAsync(c->hin + (size_t) N * E, 0, (size_t) (R - N) * E * 4, st)); } }
    else { dcopy(c, c->hin, c->lo2, (size_t) NB * E * 4); }   /* the previous layer's output is still on the device */
    for (int i = 0; i < R; ++i) { c->h_pos[i] = i < N ? pos[i] : pos0 + i; } fl_h2d(c, c->pos, c->h_pos, R * 4);
    attn_block(c, W, c->hin, N, pos0, KC, VC, c->ccat, c->ffn_inp, c->ncur, c->probs, c->sel, c->wts, dense);
    if (dense) { CK(cudaEventRecord(c->eaM, st)); return; }
    if (last_row > 0) {   // layer 60: only the last real row goes on, moved to row 0
        dcopy(c, c->ncur, c->ncur + (size_t) last_row * E, E * 4); dcopy(c, c->ffn_inp, c->ffn_inp + (size_t) last_row * E, E * 4);
        dcopy(c, c->sel, c->sel + last_row * 8, 32); dcopy(c, c->wts, c->wts + last_row * 8, 32); dcopy(c, c->probs, c->probs + last_row * 256, 1024);
    }
    // shared expert on ncur (all rows): partials kept for the spec input and the MoE combine
    quant_x<<<blocks_for((size_t) R * (E / 32)), 256, 0, st>>>(c->ncur, E, G, E, NB * E, c->xcur);
    mvg(c, F_Q5K, W->sh_gate, 0, HSH, E, 1, c->xcur, 7, c->partS, G); mvg(c, F_Q5K, W->sh_up, 0, HSH, E, 1, c->xcur, 7, c->partS + (size_t) 7 * R * HSH, G);
    swiglu_q8<<<blocks_for((size_t) R * (HSH / 32)), 256, 0, st>>>(c->partS, c->partS + (size_t) 7 * R * HSH, 7, G, HSH, c->xh);
    mvg(c, F_Q6K, W->sh_down, 0, E, HSH, 1, c->xh, 2, c->partSD, G);
    if (wrow >= 0) { dcopy(c, c->ncur_all + (size_t) wrow * E, c->ncur, (size_t) N * E * 4); base_rows<<<blocks_for((size_t) N * E), 256, 0, st>>>(c->partSD, 2, c->ffn_inp, c->base_all, wrow, N, R); }   /* wide pass: this group's rows into the pass-wide buffers */
    CK(cudaEventRecord(c->eaM, st));
}
extern "C" int fl_wide_init(FLCtx * c, int wmax, int KE) {   // buffers for a pass of up to wmax rows with KE experts per row
    c->wmax = wmax; c->wke = KE; const size_t R = (size_t) wmax * E;
    c->ncur_all = dmalloc<float>(R); c->base_all = dmalloc<float>(R); c->out_all = dmalloc<float>(R * KE);
    for (int p = 0; p < 2; ++p) {   // slot tables per batch parity: batch b+1's host writes never touch what batch b's queued copies still read
        c->wrows[p] = dmalloc<int>(72 * NB); c->wkidx[p] = dmalloc<int>(72 * NB); c->wmidx[p] = dmalloc<int>(72); c->wkw[p] = dmalloc<float>(72 * NB);
        c->h_wrows[p] = hmalloc<int>(c, 72 * NB); c->h_wkidx[p] = hmalloc<int>(c, 72 * NB); c->h_wmidx[p] = hmalloc<int>(c, 72); c->h_wkw[p] = hmalloc<float>(c, 72 * NB); CK(cudaEventCreateWithFlags(&c->wev[p], cudaEventDisableTiming));
    }
    c->h_wh = hmalloc<float>(c, R); c->h_wmoe = hmalloc<float>(c, R); c->h_wmo = hmalloc<float>(c, R);
    c->xw = qx_alloc(E, 72, false); size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    std::fprintf(stderr, "[fused] wide pass: up to %d rows x %d experts per row (%.0f MB of VRAM), VRAM free %.0f MB\n", wmax, KE, (R * (2 + KE) * 4 + (size_t) 72 * NB * (E + 16)) / 1048576.0, fr / 1048576.0); return 1;
}
// routed experts of one arena batch applied to the rows that use them: slot s = expert midx[s] of the batch x up to 8 (row, k) pairs
extern "C" void fl_wide_moe(FLCtx * c, const void * gate, const void * up, const void * down, int down_fmt, int nexp, const int * midx, const int * rows, const int * kidx, const float * wk, int nslots, int KE, int par) {
    cudaStream_t st = c->st; const size_t wsg = (size_t) HSH * (E / QK) * BLK_IQ2, wsd = (size_t) E * (HSH / QK) * (down_fmt == FL_Q3K ? BLK_Q3 : BLK_Q2); (void) nexp; par &= 1;
    int * wr = c->wrows[par], * wki = c->wkidx[par], * wmi = c->wmidx[par]; float * wkw = c->wkw[par]; int * hr = c->h_wrows[par], * hki = c->h_wkidx[par], * hmi = c->h_wmidx[par]; float * hkw = c->h_wkw[par];
    for (int s0 = 0; s0 < nslots; s0 += 72) {
        const int ns = std::min(72, nslots - s0); if (s0 > 0) { CK(cudaStreamSynchronize(st)); }   // a further chunk reuses the tables: the first must be done
        std::memcpy(hr, rows + (size_t) s0 * NB, (size_t) ns * NB * 4); std::memcpy(hki, kidx + (size_t) s0 * NB, (size_t) ns * NB * 4); std::memcpy(hkw, wk + (size_t) s0 * NB, (size_t) ns * NB * 4); std::memcpy(hmi, midx + s0, (size_t) ns * 4);
        fl_h2d(c, wr, hr, (size_t) ns * NB * 4); fl_h2d(c, wki, hki, (size_t) ns * NB * 4); fl_h2d(c, wkw, hkw, (size_t) ns * NB * 4); fl_h2d(c, wmi, hmi, (size_t) ns * 4);
        quant_rows<<<blocks_for((size_t) ns * NB * (E / 32)), 256, 0, st>>>(c->ncur_all, wr, ns, c->xw);
        MIDX = wmi; mv(c, F_IQ2, gate, wsg, HSH, E, ns, c->xw, 1, 1, c->partA); mv(c, F_IQ2, up, wsg, HSH, E, ns, c->xw, 1, 1, c->partA + (size_t) ns * NB * HSH);
        swiglu_q8<<<blocks_for((size_t) ns * NB * (HSH / 32)), 256, 0, st>>>(c->partA, c->partA + (size_t) ns * NB * HSH, 1, ns, HSH, c->xh);
        mv(c, down_fmt == FL_Q3K ? F_Q3K : F_Q2K, down, wsd, E, HSH, ns, c->xh, 1, 1, c->partB); MIDX = nullptr;
        scatter_out<<<blocks_for((size_t) ns * NB * E), 256, 0, st>>>(c->partB, 1, ns, wr, wki, wkw, KE, c->out_all);
    }
}
extern "C" void fl_wide_mark(FLCtx * c, int par) { CK(cudaEventRecord(c->wev[par & 1], c->st)); }   // this parity's staging copies and kernels are queued
extern "C" void fl_wide_wait(FLCtx * c, int par) { CK(cudaEventSynchronize(c->wev[par & 1])); }   // the host may rewrite that parity's staging half and tables
extern "C" void fl_wide_finish(FLCtx * c, int n, int KE, FLOut * out) {   // h and moe of the whole pass into the host mirrors, waited
    std::memset(out, 0, sizeof(FLOut)); float * hd = (float *) mapped(c, c->h_wh), * md = (float *) mapped(c, c->h_wmoe);
    if (hd && md) { wide_finish<<<blocks_for((size_t) n * E), 256, 0, c->st>>>(c->base_all, c->out_all, KE, n, hd, md); }
    else { wide_finish<<<blocks_for((size_t) n * E), 256, 0, c->st>>>(c->base_all, c->out_all, KE, n, c->ncur_all, c->base_all); d2h(c, c->h_wh, c->ncur_all, (size_t) n * E * 4); d2h(c, c->h_wmoe, c->base_all, (size_t) n * E * 4); }
    CK(cudaStreamSynchronize(c->st)); out->h = c->h_wh; out->moe = c->h_wmoe;
}
extern "C" void fl_wide_skip(FLCtx * c, const float * mo, int n, FLOut * out) {   // every row served from memory: h = base + stored output
    std::memset(out, 0, sizeof(FLOut)); std::memcpy(c->h_wmo, mo, (size_t) n * E * 4); const float * mod = (const float *) mapped(c, c->h_wmo); float * hd = (float *) mapped(c, c->h_wh);
    if (mod && hd) { wide_skip<<<blocks_for((size_t) n * E), 256, 0, c->st>>>(c->base_all, mod, n, hd); }
    else { fl_h2d(c, c->ncur_all, c->h_wmo, (size_t) n * E * 4); wide_skip<<<blocks_for((size_t) n * E), 256, 0, c->st>>>(c->base_all, c->ncur_all, n, c->out_all); d2h(c, c->h_wh, c->out_all, (size_t) n * E * 4); }
    CK(cudaStreamSynchronize(c->st)); out->h = c->h_wh;
}
// attribution: the routed experts' weighted pushes for one row, as the combine added them (narrow MoE from partB, wide pass from out_all)
extern "C" void fl_moe_outs(FLCtx * c, int b, const int * ids, const float * w, int KE, float * out) {
    const int KS = c->last_ksd, ns = c->last_nexp; std::vector<float> part((size_t) KS * ns * NB * E); CK(cudaStreamSynchronize(c->st));
    CK(cudaMemcpy(part.data(), c->partB, part.size() * 4, cudaMemcpyDeviceToHost));
    for (int k = 0; k < KE; ++k) { const int m = ids[k]; for (int e = 0; e < E; ++e) { float v = 0.f; for (int ks = 0; ks < KS; ++ks) { v += part[(((size_t) ks * ns + m) * NB + b) * E + e]; } out[(size_t) k * E + e] = w[k] * v; } }
}
extern "C" void fl_wide_outs(FLCtx * c, int row, int KE, float * out) {
    CK(cudaStreamSynchronize(c->st)); CK(cudaMemcpy(out, c->out_all + (size_t) row * KE * E, (size_t) KE * E * 4, cudaMemcpyDeviceToHost));
}
extern "C" void fl_spec(FLCtx * c, const FLW * W1, void * KC1, void * VC1) {
    CK(cudaEventRecord(c->eaS, c->st)); c->spec_on = 1;
    hspec_add<<<blocks_for((size_t) NB * E), 256, 0, c->st>>>(c->partSD, 2, c->ffn_inp, c->hspec);
    attn_block(c, W1, c->hspec, c->N, c->h_pos[0], KC1, VC1, c->ccat2, c->fi2, c->ncur2, c->probs2, c->sel2, c->wts2, 0);
}
extern "C" void fl_sketch(FLCtx * c, const float * TPROJ, const void * GSKT, int K) {
    const int R = round_g((c->N + NB - 1) / NB) * NB;
    sketch_proj<<<blocks_for((size_t) R * 256 * 32), 256, 0, c->st>>>(TPROJ, c->ncur, c->skt, R);
    if (GSKT && K >= 64 && K <= c->sccap) { sketch_norm<<<R, 256, 0, c->st>>>(c->skt, c->sxn); sketch_scan<<<blocks_for((size_t) R * K * 32), 256, 0, c->st>>>((const half *) GSKT, K, c->sxn, c->sc, R); c->scK = K; }
}
extern "C" void fl_fetch(FLCtx * c, int n, int what, FLOut * out) {
    cudaStream_t st = c->st; std::memset(out, 0, sizeof(FLOut)); CK(cudaEventRecord(c->ea1, st));
    if (what & 1) { d2h(c, c->h_sel, c->sel, (size_t) n * 32); d2h(c, c->h_wts, c->wts, (size_t) n * 32); d2h(c, c->h_ncur, c->ncur, (size_t) n * E * 4); d2h(c, c->h_fi, c->ffn_inp, (size_t) n * E * 4); out->sel = c->h_sel; out->wts = c->h_wts; out->ncur = c->h_ncur; out->fi = c->h_fi; }
    if (what & 2) { d2h(c, c->h_skt, c->skt, (size_t) n * 256 * 4); out->skt = c->h_skt; }
    if (what & 4) { d2h(c, c->h_sel2, c->sel2, NB * 32); d2h(c, c->h_wts2, c->wts2, NB * 32); out->sel_s = c->h_sel2; out->wts_s = c->h_wts2; }
    if ((what & 8) && c->scK > 0) { d2h(c, c->h_sc, c->sc, (size_t) n * c->scK * 4); out->sc = c->h_sc; }
    if (what & 16) { d2h(c, c->h_ccat, c->ccat, (size_t) c->N * 576 * 4); out->ccat = c->h_ccat; }
    if (what & 32) { d2h(c, c->h_h, c->lo2, (size_t) n * E * 4); out->h = c->h_h; }
    CK(cudaStreamSynchronize(st)); float ms = 0.f; cudaEventElapsedTime(&ms, c->ea0, c->ea1); c->ms_attn += ms;
    if (c->dbg && !c->spec_on) { float p = 0.f; cudaEventElapsedTime(&p, c->ea0, c->eP[0]); c->ms_p[0] += p; cudaEventElapsedTime(&p, c->eP[0], c->eP[1]); c->ms_p[1] += p; cudaEventElapsedTime(&p, c->eP[1], c->eP[2]); c->ms_p[2] += p; cudaEventElapsedTime(&p, c->eP[2], c->eaM); c->ms_p[3] += p; }
    ms = 0.f; cudaEventElapsedTime(&ms, c->ea0, c->eaM); c->ms_main += ms; if (c->spec_on) { ms = 0.f; cudaEventElapsedTime(&ms, c->eaS, c->ea1); c->ms_spec += ms; }
}
extern "C" void fl_moe(FLCtx * c, const void * gate, const void * up, const void * down, int nexp, int down_fmt, const int * ids, const float * w, int KE, int n, FLOut * out) {
    cudaStream_t st = c->st; std::memset(out, 0, sizeof(FLOut)); CK(cudaEventRecord(c->em0, st));
    for (int b = 0; b < NB; ++b) { for (int k = 0; k < KE; ++k) { c->h_ids[b * KE + k] = b < n ? ids[b * KE + k] : 0; c->h_w[b * KE + k] = b < n ? w[b * KE + k] : 0.f; } }
    fl_h2d(c, c->ids, c->h_ids, NB * KE * 4); fl_h2d(c, c->w, c->h_w, NB * KE * 4);
    quant_x<<<blocks_for((size_t) NB * (E / 32)), 256, 0, st>>>(c->ncur, E, 1, E, E, c->xcur);
    const int KSg = pick_ks(c, HSH / 16, nexp, 7, 72); const size_t wsg = (size_t) HSH * (E / QK) * BLK_IQ2;
    mv(c, F_IQ2, gate, wsg, HSH, E, nexp, c->xcur, 0, KSg, c->partA); mv(c, F_IQ2, up, wsg, HSH, E, nexp, c->xcur, 0, KSg, c->partA + (size_t) KSg * nexp * NB * HSH);
    swiglu_q8<<<blocks_for((size_t) nexp * NB * (HSH / 32)), 256, 0, st>>>(c->partA, c->partA + (size_t) KSg * nexp * NB * HSH, KSg, nexp, HSH, c->xh);
    const int KSd = pick_ks(c, E / 16, nexp, 2, 72); const size_t wsd = (size_t) E * (HSH / QK) * (down_fmt == FL_Q3K ? BLK_Q3 : BLK_Q2);
    mv(c, down_fmt == FL_Q3K ? F_Q3K : F_Q2K, down, wsd, E, HSH, nexp, c->xh, 1, KSd, c->partB); c->last_ksd = KSd; c->last_nexp = nexp;
    moe_combine<<<blocks_for((size_t) NB * E), 256, 0, st>>>(c->partB, KSd, nexp, c->ids, c->w, KE, c->partSD, 2, nullptr, c->ffn_inp, c->lo2, c->moe);
    d2h(c, c->h_h, c->lo2, (size_t) n * E * 4); d2h(c, c->h_moe, c->moe, (size_t) n * E * 4); CK(cudaEventRecord(c->em1, st));
    CK(cudaStreamSynchronize(st)); float ms = 0.f; cudaEventElapsedTime(&ms, c->em0, c->em1); c->ms_moe += ms; out->h = c->h_h; out->moe = c->h_moe;
}
extern "C" void fl_skip(FLCtx * c, const float * mo, int n, FLOut * out) {
    cudaStream_t st = c->st; std::memset(out, 0, sizeof(FLOut)); CK(cudaEventRecord(c->em0, st));
    std::memcpy(c->h_mo, mo, (size_t) n * E * 4); fl_h2d(c, c->mo, c->h_mo, (size_t) n * E * 4); if (n < NB) { CK(cudaMemsetAsync(c->mo + (size_t) n * E, 0, (size_t) (NB - n) * E * 4, st)); }
    moe_combine<<<blocks_for((size_t) NB * E), 256, 0, st>>>(nullptr, 0, 0, nullptr, nullptr, 0, c->partSD, 2, c->mo, c->ffn_inp, c->lo2, nullptr);
    d2h(c, c->h_h, c->lo2, (size_t) n * E * 4); CK(cudaEventRecord(c->em1, st));
    CK(cudaStreamSynchronize(st)); float ms = 0.f; cudaEventElapsedTime(&ms, c->em0, c->em1); c->ms_moe += ms; out->h = c->h_h;
}
extern "C" double fl_gpu_ms(FLCtx * c, int which) { return which == 0 ? c->ms_attn : which == 1 ? c->ms_moe : which == 2 ? c->ms_main : which >= 4 && which <= 7 ? c->ms_p[which - 4] : c->ms_spec; }   // 0 whole attention phase (incl. idle while the host waits), 1 experts, 2 attention+router+shared, 3 spec+scan
