/*
 * qwen_asr_kernels.cu - CUDA kernels for Qwen3-ASR decoder forward pass
 *
 * Compiled to CUBIN via nvcc, embedded as C byte-array, loaded via CUDA
 * Driver API (cuModuleLoadData). All functions are extern "C" for driver API
 * compatibility. Adapted from voxtral.c CUDA kernels where noted.
 *
 * Kernel inventory:
 *   qwen_rms_norm_f32        - RMSNorm (multi-row)
 *   qwen_rms_norm_per_head_f32 - Per-head RMSNorm (in-place, one block per head)
 *   qwen_apply_rope_neox_f32 - NeoX split-half RoPE
 *   qwen_add_inplace_f32     - Element-wise residual add
 *   qwen_swiglu_interleaved_f32 - SwiGLU on interleaved [g0,u0,g1,u1,...] layout
 *   qwen_kv_append_f32       - Append K,V vectors to cache at position
 *   qwen_attn_gqa2_f32       - GQA 2:1 causal attention (online softmax)
 *   qwen_attn_probe_f32      - Timestamp alignment probe (attention argmax)
 *   qwen_argmax_f32          - Single-block argmax reduction
 */

#include <stdint.h>
#include <math.h>

/* ========================================================================
 * Warp/block reduction primitives
 * ======================================================================== */

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

/* ========================================================================
 * qwen_rms_norm_f32 - RMSNorm (adapted from voxtral vox_rms_norm_f32)
 *
 * out[r,i] = x[r,i] * rsqrt(mean(x[r,:]^2) + eps) * weight[i]
 * Grid: rows, Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_rms_norm_f32(float *out,
                                              const float *x,
                                              const float *weight,
                                              int rows,
                                              int hidden,
                                              float eps) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;

    const float *x_row = x + (size_t)r * (size_t)hidden;
    float *o_row = out + (size_t)r * (size_t)hidden;

    __shared__ float sh[256];
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sh[0] / (float)hidden + eps);
    for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
        o_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

/* ========================================================================
 * qwen_rms_norm_per_head_f32 - Per-head RMSNorm (in-place)
 *
 * x: [n_heads, head_dim], weight: [head_dim]
 * Each block normalizes one head independently.
 * Grid: n_heads, Block: 128 (head_dim=128, one thread per element)
 * ======================================================================== */

extern "C" __global__ void qwen_rms_norm_per_head_f32(float *x,
                                                       const float *weight,
                                                       int n_heads,
                                                       int head_dim,
                                                       float eps) {
    int h = (int)blockIdx.x;
    if (h >= n_heads) return;
    int tid = (int)threadIdx.x;

    float *head = x + (size_t)h * (size_t)head_dim;

    /* Each thread handles one or more dimensions */
    __shared__ float sh[128];
    float sum = 0.0f;
    for (int d = tid; d < head_dim; d += (int)blockDim.x) {
        float v = head[d];
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sh[0] / (float)head_dim + eps);
    for (int d = tid; d < head_dim; d += (int)blockDim.x) {
        head[d] = head[d] * inv_rms * weight[d];
    }
}

/* ========================================================================
 * qwen_apply_rope_neox_f32 - NeoX split-half RoPE (in-place)
 *
 * x: [n_heads, head_dim], cos/sin: [head_dim] (for single position)
 * NeoX variant: x1 = x[...,:half], x2 = x[...,half:]
 *   new_x1 = x1 * cos - x2 * sin
 *   new_x2 = x2 * cos + x1 * sin
 * Grid: n_heads, Block: half_dim (=64 for head_dim=128)
 * ======================================================================== */

extern "C" __global__ void qwen_apply_rope_neox_f32(float *x,
                                                     const float *cos_vals,
                                                     const float *sin_vals,
                                                     int n_heads,
                                                     int head_dim) {
    int h = (int)blockIdx.x;
    if (h >= n_heads) return;
    int d = (int)threadIdx.x;
    int half = head_dim / 2;
    if (d >= half) return;

    float *vec = x + (size_t)h * (size_t)head_dim;
    float x1 = vec[d];
    float x2 = vec[half + d];
    float c = cos_vals[d];
    float s = sin_vals[d];

    vec[d]        = x1 * c - x2 * s;
    vec[half + d] = x2 * c + x1 * s;
}

/* ========================================================================
 * qwen_add_inplace_f32 - Element-wise add (adapted from voxtral)
 *
 * x[i] += y[i]
 * Grid: ceil(n/256), Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_add_inplace_f32(float *x,
                                                 const float *y,
                                                 int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] += y[idx];
}

/* ========================================================================
 * qwen_swiglu_interleaved_f32 - SwiGLU on interleaved layout
 *
 * gate_up: [2*intermediate] with layout [g0,u0,g1,u1,...]
 * out: [intermediate] with out[j] = SiLU(gate_up[2j]) * gate_up[2j+1]
 * Grid: ceil(intermediate/256), Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_swiglu_interleaved_f32(float *out,
                                                        const float *gate_up,
                                                        int intermediate) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= intermediate) return;

    float g = gate_up[2 * j];
    float u = gate_up[2 * j + 1];
    /* SiLU(g) = g / (1 + exp(-g)) = g * sigmoid(g) */
    float sg = 1.0f / (1.0f + __expf(-g));
    out[j] = g * sg * u;
}

/* ========================================================================
 * qwen_kv_append_f32 - Append K,V to cache (adapted from voxtral)
 *
 * k_base[pos * kv_dim + idx] = k[idx]
 * v_base[pos * kv_dim + idx] = v[idx]
 * Grid: ceil(kv_dim/256), Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_kv_append_f32(float *k_base,
                                               float *v_base,
                                               const float *k,
                                               const float *v,
                                               int pos,
                                               int kv_dim) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= kv_dim) return;
    int dst = pos * kv_dim + idx;
    k_base[dst] = k[idx];
    v_base[dst] = v[idx];
}

/* ========================================================================
 * qwen_attn_gqa2_f32 - GQA 2:1 causal attention (online softmax)
 *
 * Adapted from voxtral vox_attn_q4_kv8_f32 but with 2:1 GQA ratio
 * (2 query heads per KV head, i.e. kv_h = h >> 1).
 *
 * Q: [n_heads * head_dim] (single query position)
 * K cache: [total_seq * kv_heads * head_dim]
 * V cache: [total_seq * kv_heads * head_dim]
 * out: [n_heads * head_dim]
 *
 * Grid: n_heads (16), Block: 32 (one warp per head)
 * Each lane owns 4 dims of the 128-dim head.
 * ======================================================================== */

extern "C" __global__ void qwen_attn_gqa2_f32(float *out,
                                               const float *q,
                                               const float *k_cache,
                                               const float *v_cache,
                                               int total_seq,
                                               int n_kv_heads,
                                               int head_dim,
                                               float scale) {
    int h = (int)blockIdx.x;     /* query head index: 0..n_heads-1 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (lane >= 32) return;

    /* GQA 2:1 mapping: 2 query heads share 1 KV head */
    int kv_h = h >> 1;
    int kv_dim = n_kv_heads * head_dim;

    /* Load Q vector: 4 elements per lane (32 lanes x 4 = 128 dims) */
    float qv0 = q[h * head_dim + (lane + 0 * 32)];
    float qv1 = q[h * head_dim + (lane + 1 * 32)];
    float qv2 = q[h * head_dim + (lane + 2 * 32)];
    float qv3 = q[h * head_dim + (lane + 3 * 32)];

    /* Online softmax state */
    float max_score = -1.0e30f;
    float sum_exp = 0.0f;

    /* Accumulated output vector */
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    for (int j = 0; j < total_seq; j++) {
        const float *k_row = k_cache + (size_t)j * kv_dim + (size_t)kv_h * head_dim;
        float k0 = k_row[lane + 0 * 32];
        float k1 = k_row[lane + 1 * 32];
        float k2 = k_row[lane + 2 * 32];
        float k3 = k_row[lane + 3 * 32];

        float partial = qv0 * k0 + qv1 * k1 + qv2 * k2 + qv3 * k3;
        float dot = warp_reduce_sum(partial);
        dot = __shfl_sync(0xffffffff, dot, 0);

        float score = dot * scale;

        /* Online softmax (lane 0 tracks scalars) */
        float w = 0.0f;
        float corr = 1.0f;
        int new_max = 0;
        if (lane == 0) {
            if (score > max_score) {
                corr = __expf(max_score - score);
                sum_exp = sum_exp * corr + 1.0f;
                max_score = score;
                w = 1.0f;
                new_max = 1;
            } else {
                w = __expf(score - max_score);
                sum_exp += w;
                corr = 1.0f;
                new_max = 0;
            }
        }
        w = __shfl_sync(0xffffffff, w, 0);
        corr = __shfl_sync(0xffffffff, corr, 0);
        new_max = __shfl_sync(0xffffffff, new_max, 0);

        const float *v_row = v_cache + (size_t)j * kv_dim + (size_t)kv_h * head_dim;
        float v0 = v_row[lane + 0 * 32];
        float v1 = v_row[lane + 1 * 32];
        float v2 = v_row[lane + 2 * 32];
        float v3 = v_row[lane + 3 * 32];

        if (new_max) {
            out0 = out0 * corr + v0;
            out1 = out1 * corr + v1;
            out2 = out2 * corr + v2;
            out3 = out3 * corr + v3;
        } else {
            out0 += w * v0;
            out1 += w * v1;
            out2 += w * v2;
            out3 += w * v3;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    out[h * head_dim + (lane + 0 * 32)] = out0 * inv_sum;
    out[h * head_dim + (lane + 1 * 32)] = out1 * inv_sum;
    out[h * head_dim + (lane + 2 * 32)] = out2 * inv_sum;
    out[h * head_dim + (lane + 3 * 32)] = out3 * inv_sum;
}

/* ========================================================================
 * qwen_attn_probe_f32 - Timestamp alignment probe
 *
 * Finds which encoder KV position has the highest summed Q*K dot product
 * across all query heads. Used for audio timestamp alignment.
 *
 * q: [n_heads * head_dim]
 * k_cache: [total_seq * kv_heads * head_dim]
 * Returns: index (relative to enc_start) of peak position in out_pos[0]
 *
 * Grid: 1, Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_attn_probe_f32(int *out_pos,
                                                const float *q,
                                                const float *k_cache,
                                                int enc_start,
                                                int enc_end,
                                                int n_heads,
                                                int n_kv_heads,
                                                int head_dim,
                                                float scale) {
    int tid = (int)threadIdx.x;
    int enc_count = enc_end - enc_start;
    if (enc_count <= 0) {
        if (tid == 0) out_pos[0] = enc_start;
        return;
    }

    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    /* Each thread processes a subset of encoder positions */
    float best_score = -1.0e30f;
    int best_pos = enc_start;

    for (int p = enc_start + tid; p < enc_end; p += (int)blockDim.x) {
        float sum = 0.0f;
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;
            const float *q_head = q + h * head_dim;
            const float *k_at_p = k_cache + (size_t)p * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_head[d] * k_at_p[d];
            sum += dot;
        }
        if (sum > best_score) {
            best_score = sum;
            best_pos = p;
        }
    }

    /* Reduce across threads to find global best */
    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    sh_val[tid] = best_score;
    sh_idx[tid] = best_pos;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sh_val[tid + stride] > sh_val[tid]) {
                sh_val[tid] = sh_val[tid + stride];
                sh_idx[tid] = sh_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) out_pos[0] = sh_idx[0];
}

/* ========================================================================
 * qwen_argmax_f32 - Single-block argmax (adapted from voxtral)
 *
 * Finds index of maximum element in x[0..n-1].
 * Grid: 1, Block: 256
 * ======================================================================== */

extern "C" __global__ void qwen_argmax_f32(int *out_idx,
                                            const float *x,
                                            int n) {
    int tid = (int)threadIdx.x;
    float best = -1.0e30f;
    int best_i = 0;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        float v = x[i];
        if (v > best) { best = v; best_i = i; }
    }

    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    sh_val[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sh_val[tid + stride] > sh_val[tid]) {
                sh_val[tid] = sh_val[tid + stride];
                sh_idx[tid] = sh_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) out_idx[0] = sh_idx[0];
}
