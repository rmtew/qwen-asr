/*
 * qwen_asr_gpu.c - cuBLAS GPU acceleration for Qwen3-ASR GEMM operations
 *
 * Strategy: Keep all weight matrices on GPU (uploaded once at model load).
 * Per-GEMM: upload activation from CPU, cuBLAS sgemm, download result to CPU.
 * Non-GEMM ops (LayerNorm, RMSNorm, GELU, SwiGLU, RoPE, softmax, attention)
 * remain on CPU.
 *
 * Compiled with MSVC + CUDA headers (no nvcc needed). Links against
 * cublas.lib and cudart.lib from the CUDA Toolkit.
 */

#ifdef USE_CUBLAS

#include "qwen_asr_gpu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern int qwen_verbose;

#define MAX_GPU_WEIGHTS 512

typedef struct {
    const void *host_ptr;   /* CPU pointer (lookup key) */
    float *d_ptr;           /* GPU device pointer (f32) */
    int rows;
    int cols;
} gpu_weight_entry_t;

struct qwen_gpu_ctx {
    cublasHandle_t handle;

    gpu_weight_entry_t weights[MAX_GPU_WEIGHTS];
    int n_weights;

    /* Pre-allocated device buffers for activations and results.
     * Grown dynamically as needed. */
    float *d_A;
    float *d_C;
    size_t d_A_cap;     /* capacity in floats */
    size_t d_C_cap;

    /* Host buffer for argmax download */
    float *h_argmax_buf;
    size_t h_argmax_cap;

    /* VRAM tracking */
    size_t vram_weights;
    size_t vram_buffers;
};

/* Ensure a device buffer has at least 'need' floats. Grows by doubling. */
static int ensure_device_buf(float **d_buf, size_t *cap, size_t need,
                              size_t *vram_counter) {
    if (need <= *cap) return 0;
    size_t new_cap = *cap > 0 ? *cap : 4096;
    while (new_cap < need) new_cap *= 2;

    float *new_buf;
    cudaError_t err = cudaMalloc((void **)&new_buf, new_cap * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU: cudaMalloc failed for buffer (%.1f MB): %s\n",
                new_cap * sizeof(float) / (1024.0 * 1024.0),
                cudaGetErrorString(err));
        return -1;
    }

    if (*d_buf) {
        if (vram_counter) *vram_counter -= *cap * sizeof(float);
        cudaFree(*d_buf);
    }
    *d_buf = new_buf;
    *cap = new_cap;
    if (vram_counter) *vram_counter += new_cap * sizeof(float);
    return 0;
}

qwen_gpu_ctx_t *qwen_gpu_init(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "GPU: No CUDA devices available\n");
        return NULL;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU: %s (%.0f MB VRAM, compute %d.%d)\n",
                prop.name, prop.totalGlobalMem / (1024.0 * 1024.0),
                prop.major, prop.minor);

    qwen_gpu_ctx_t *gpu = (qwen_gpu_ctx_t *)calloc(1, sizeof(qwen_gpu_ctx_t));
    if (!gpu) return NULL;

    cublasStatus_t status = cublasCreate(&gpu->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GPU: cuBLAS init failed (status %d)\n", (int)status);
        free(gpu);
        return NULL;
    }

    /* Use tensor cores when available (Ampere+). cuBLAS will auto-select
     * the fastest algorithm. */
    cublasSetMathMode(gpu->handle, CUBLAS_DEFAULT_MATH);

    return gpu;
}

void qwen_gpu_free(qwen_gpu_ctx_t *gpu) {
    if (!gpu) return;

    for (int i = 0; i < gpu->n_weights; i++) {
        if (gpu->weights[i].d_ptr) cudaFree(gpu->weights[i].d_ptr);
    }

    if (gpu->d_A) cudaFree(gpu->d_A);
    if (gpu->d_C) cudaFree(gpu->d_C);
    free(gpu->h_argmax_buf);

    cublasDestroy(gpu->handle);
    free(gpu);
}

int qwen_gpu_upload_weight_f32(qwen_gpu_ctx_t *gpu, const float *host_ptr,
                                int rows, int cols) {
    if (!gpu || !host_ptr) return -1;
    if (gpu->n_weights >= MAX_GPU_WEIGHTS) {
        fprintf(stderr, "GPU: weight registry full (%d)\n", MAX_GPU_WEIGHTS);
        return -1;
    }

    size_t n = (size_t)rows * cols;
    size_t bytes = n * sizeof(float);
    float *d_ptr;
    cudaError_t err = cudaMalloc((void **)&d_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU: cudaMalloc failed for f32 weight %dx%d (%.1f MB): %s\n",
                rows, cols, bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU: cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return -1;
    }

    gpu_weight_entry_t *w = &gpu->weights[gpu->n_weights++];
    w->host_ptr = host_ptr;
    w->d_ptr = d_ptr;
    w->rows = rows;
    w->cols = cols;
    gpu->vram_weights += bytes;
    return 0;
}

int qwen_gpu_upload_weight_bf16(qwen_gpu_ctx_t *gpu, const uint16_t *host_ptr,
                                 int rows, int cols) {
    if (!gpu || !host_ptr) return -1;
    if (gpu->n_weights >= MAX_GPU_WEIGHTS) {
        fprintf(stderr, "GPU: weight registry full (%d)\n", MAX_GPU_WEIGHTS);
        return -1;
    }

    size_t n = (size_t)rows * cols;
    size_t bytes = n * sizeof(float);

    /* Convert bf16 -> f32 on CPU */
    float *h_f32 = (float *)malloc(bytes);
    if (!h_f32) return -1;

    uint32_t *h_u32 = (uint32_t *)(void *)h_f32;
    for (size_t i = 0; i < n; i++) {
        h_u32[i] = ((uint32_t)host_ptr[i]) << 16;
    }

    /* Upload f32 to GPU */
    float *d_ptr;
    cudaError_t err = cudaMalloc((void **)&d_ptr, bytes);
    if (err != cudaSuccess) {
        free(h_f32);
        fprintf(stderr, "GPU: cudaMalloc failed for bf16 weight %dx%d (%.1f MB): %s\n",
                rows, cols, bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_ptr, h_f32, bytes, cudaMemcpyHostToDevice);
    free(h_f32);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU: cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return -1;
    }

    gpu_weight_entry_t *w = &gpu->weights[gpu->n_weights++];
    w->host_ptr = host_ptr;
    w->d_ptr = d_ptr;
    w->rows = rows;
    w->cols = cols;
    gpu->vram_weights += bytes;
    return 0;
}

int qwen_gpu_find_weight(qwen_gpu_ctx_t *gpu, const void *host_ptr) {
    if (!gpu || !host_ptr) return -1;
    for (int i = 0; i < gpu->n_weights; i++) {
        if (gpu->weights[i].host_ptr == host_ptr) return i;
    }
    return -1;
}

void qwen_gpu_gemm(qwen_gpu_ctx_t *gpu, float *C_host,
                   const float *A_host, int weight_handle,
                   int M, int K, int N) {
    if (!gpu || weight_handle < 0 || weight_handle >= gpu->n_weights) return;

    gpu_weight_entry_t *w = &gpu->weights[weight_handle];
    size_t A_size = (size_t)M * K;
    size_t C_size = (size_t)M * N;

    /* Ensure activation buffers are large enough */
    if (ensure_device_buf(&gpu->d_A, &gpu->d_A_cap, A_size, &gpu->vram_buffers) != 0) return;
    if (ensure_device_buf(&gpu->d_C, &gpu->d_C_cap, C_size, &gpu->vram_buffers) != 0) return;

    /* Upload activation A[M,K] to GPU */
    cudaMemcpy(gpu->d_A, A_host, A_size * sizeof(float), cudaMemcpyHostToDevice);

    /* cuBLAS GEMM: row-major C[M,N] = A[M,K] @ W[N,K]^T
     *
     * cuBLAS is column-major. The standard trick:
     *   cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1, A, K, W, K, 0, C, N)
     * becomes:
     *   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, W, K, A, K, &b, C, N)
     */
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(gpu->handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha, w->d_ptr, K,
                gpu->d_A, K,
                &beta, gpu->d_C, N);

    /* Download result C[M,N] from GPU */
    cudaMemcpy(C_host, gpu->d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);
}

int qwen_gpu_argmax_matvec(qwen_gpu_ctx_t *gpu,
                            const float *x_host, int weight_handle,
                            int in_dim, int out_dim) {
    if (!gpu || weight_handle < 0 || weight_handle >= gpu->n_weights) return 0;

    /* GEMM with M=1: result[1, out_dim] = x[1, in_dim] @ W[out_dim, in_dim]^T */
    size_t A_size = (size_t)in_dim;
    size_t C_size = (size_t)out_dim;

    if (ensure_device_buf(&gpu->d_A, &gpu->d_A_cap, A_size, &gpu->vram_buffers) != 0) return 0;
    if (ensure_device_buf(&gpu->d_C, &gpu->d_C_cap, C_size, &gpu->vram_buffers) != 0) return 0;

    cudaMemcpy(gpu->d_A, x_host, A_size * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    gpu_weight_entry_t *w = &gpu->weights[weight_handle];
    cublasSgemm(gpu->handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_dim, 1, in_dim,
                &alpha, w->d_ptr, in_dim,
                gpu->d_A, in_dim,
                &beta, gpu->d_C, out_dim);

    /* Download logits and find argmax on CPU */
    if (gpu->h_argmax_cap < (size_t)out_dim) {
        free(gpu->h_argmax_buf);
        gpu->h_argmax_buf = (float *)malloc((size_t)out_dim * sizeof(float));
        gpu->h_argmax_cap = gpu->h_argmax_buf ? (size_t)out_dim : 0;
    }
    if (!gpu->h_argmax_buf) return 0;

    cudaMemcpy(gpu->h_argmax_buf, gpu->d_C,
               (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    int best = 0;
    float best_val = gpu->h_argmax_buf[0];
    for (int i = 1; i < out_dim; i++) {
        if (gpu->h_argmax_buf[i] > best_val) {
            best_val = gpu->h_argmax_buf[i];
            best = i;
        }
    }
    return best;
}

float *qwen_gpu_get_weight_ptr(qwen_gpu_ctx_t *gpu, int handle) {
    if (!gpu || handle < 0 || handle >= gpu->n_weights) return NULL;
    return gpu->weights[handle].d_ptr;
}

void *qwen_gpu_get_cublas_handle(qwen_gpu_ctx_t *gpu) {
    if (!gpu) return NULL;
    return (void *)gpu->handle;
}

void qwen_gpu_print_stats(qwen_gpu_ctx_t *gpu) {
    if (!gpu) return;
    fprintf(stderr, "GPU: %d weights uploaded (%.0f MB), buffers %.0f MB\n",
            gpu->n_weights,
            gpu->vram_weights / (1024.0 * 1024.0),
            gpu->vram_buffers / (1024.0 * 1024.0));
}

#endif /* USE_CUBLAS */

/* ========================================================================
 * Full GPU Decoder (USE_CUDA_KERNELS)
 *
 * All activations stay on device. Custom CUDA kernels for non-GEMM ops,
 * cuBLAS for GEMMs (device-to-device, no PCIe round-trips for activations).
 * Only transfers: input embedding H2D, argmax + probe result D2H.
 * ======================================================================== */

#ifdef USE_CUDA_KERNELS

#include "qwen_asr.h"
#include "qwen_asr_gpu.h"
#include "qwen_asr_kernels_cubin.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int qwen_verbose;

/* Number of CUDA kernel functions we load from the CUBIN */
#define NUM_KERNELS 9

/* Kernel function indices */
enum {
    KF_RMS_NORM = 0,
    KF_RMS_NORM_PER_HEAD,
    KF_APPLY_ROPE_NEOX,
    KF_ADD_INPLACE,
    KF_SWIGLU_INTERLEAVED,
    KF_KV_APPEND,
    KF_ATTN_GQA2,
    KF_ATTN_PROBE,
    KF_ARGMAX,
};

static const char *kernel_names[NUM_KERNELS] = {
    "qwen_rms_norm_f32",
    "qwen_rms_norm_per_head_f32",
    "qwen_apply_rope_neox_f32",
    "qwen_add_inplace_f32",
    "qwen_swiglu_interleaved_f32",
    "qwen_kv_append_f32",
    "qwen_attn_gqa2_f32",
    "qwen_attn_probe_f32",
    "qwen_argmax_f32",
};

struct qwen_gpu_dec_ctx {
    /* CUDA Driver API handles */
    CUmodule module;
    CUfunction kernels[NUM_KERNELS];

    /* Device activation buffers (single-token sizes) */
    float *d_x;          /* [hidden] */
    float *d_x_norm;     /* [hidden] */
    float *d_q;          /* [n_heads * head_dim] */
    float *d_k;          /* [n_kv_heads * head_dim] */
    float *d_v;          /* [n_kv_heads * head_dim] */
    float *d_attn_out;   /* [n_heads * head_dim] */
    float *d_proj_out;   /* [hidden] */
    float *d_gate_buf;   /* [2 * intermediate] */
    float *d_ffn_out;    /* [hidden] */
    float *d_logits;     /* [vocab_size] for final lm_head output */

    /* Device KV cache: contiguous [layers, max_seq, kv_dim] */
    float *d_kv_k;
    float *d_kv_v;
    int kv_cache_max;

    /* Device RoPE cache: [n_pos, head_dim] */
    float *d_rope_cos;
    float *d_rope_sin;
    int rope_cache_cap;

    /* Per-layer norm weights on GPU [layers][hidden] or [head_dim] */
    float **d_input_norm;     /* [layers] -> device [hidden] */
    float **d_post_attn_norm; /* [layers] -> device [hidden] */
    float **d_q_norm;         /* [layers] -> device [head_dim] */
    float **d_k_norm;         /* [layers] -> device [head_dim] */
    float *d_final_norm;      /* device [hidden] */
    int n_layers;

    /* KV cache sync tracking: positions [0, kv_synced_len) are on GPU.
     * When prefill runs on CPU, we need to upload before GPU decode. */
    int kv_synced_len;

    /* Pinned host memory for async result download */
    int *h_argmax_result;     /* pinned, 1 int */
    int *h_probe_result;      /* pinned, 1 int */

    /* Device scalars for kernel results */
    int *d_argmax_result;
    int *d_probe_result;

    /* VRAM tracking */
    size_t vram_activations;
    size_t vram_kv_cache;
    size_t vram_norms;
    size_t vram_rope;
};

/* ========================================================================
 * CUBIN Loading
 * ======================================================================== */

static int load_cubin_module(qwen_gpu_dec_ctx_t *dctx) {
    CUresult res;

    /* Initialize CUDA Driver API (safe to call multiple times) */
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "GPU dec: cuInit failed (%d)\n", (int)res);
        return -1;
    }

    /* Load embedded CUBIN */
    res = cuModuleLoadData(&dctx->module, qwen_asr_kernels_cubin);
    if (res != CUDA_SUCCESS) {
        const char *err_str = NULL;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "GPU dec: cuModuleLoadData failed (%d): %s\n",
                (int)res, err_str ? err_str : "unknown");
        fprintf(stderr, "GPU dec: This likely means the CUBIN was compiled for a different GPU architecture.\n");
        return -1;
    }

    /* Get function handles for all kernels */
    for (int i = 0; i < NUM_KERNELS; i++) {
        res = cuModuleGetFunction(&dctx->kernels[i], dctx->module, kernel_names[i]);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "GPU dec: cuModuleGetFunction failed for '%s' (%d)\n",
                    kernel_names[i], (int)res);
            cuModuleUnload(dctx->module);
            dctx->module = NULL;
            return -1;
        }
    }

    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU dec: loaded %d kernels from embedded CUBIN (%u bytes)\n",
                NUM_KERNELS, qwen_asr_kernels_cubin_len);
    return 0;
}

/* ========================================================================
 * Kernel Launch Helpers
 * ======================================================================== */

static void launch_kernel(CUfunction f, unsigned int gridX, unsigned int gridY,
                           unsigned int gridZ, unsigned int blockX,
                           unsigned int blockY, unsigned int blockZ,
                           unsigned int sharedMem, void **params) {
    CUresult res = cuLaunchKernel(f,
        gridX, gridY, gridZ, blockX, blockY, blockZ,
        sharedMem, NULL, params, NULL);
    if (res != CUDA_SUCCESS) {
        const char *err_str = NULL;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "GPU dec: kernel launch failed (%d): %s\n",
                (int)res, err_str ? err_str : "unknown");
    }
}

/* Device-to-device cuBLAS GEMM: C_d[M,N] = A_d[M,K] @ W_d[N,K]^T
 * All pointers are device memory. Uses same column-major trick as cpu-path. */
static void gpu_gemm_d2d(cublasHandle_t handle, float *C_d, const float *A_d,
                           const float *W_d, int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha, W_d, K,
                A_d, K,
                &beta, C_d, N);
}

/* ========================================================================
 * Init / Free
 * ======================================================================== */

qwen_gpu_dec_ctx_t *qwen_gpu_dec_init(qwen_gpu_ctx_t *gpu,
                                       const qwen_config_t *cfg) {
    if (!gpu || !cfg) return NULL;

    /* Env var to force cuBLAS-only mode (for A/B comparison) */
    {
        char *env_val = NULL;
        size_t env_len = 0;
        _dupenv_s(&env_val, &env_len, "QWEN_NO_CUDA_KERNELS");
        int skip = (env_val != NULL);
        free(env_val);
        if (skip) {
            if (qwen_verbose >= 1)
                fprintf(stderr, "GPU dec: disabled by QWEN_NO_CUDA_KERNELS env var\n");
            return NULL;
        }
    }

    qwen_gpu_dec_ctx_t *dctx = (qwen_gpu_dec_ctx_t *)calloc(1, sizeof(*dctx));
    if (!dctx) return NULL;

    /* Load CUBIN */
    if (load_cubin_module(dctx) != 0) {
        free(dctx);
        return NULL;
    }

    int hidden = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int inter = cfg->dec_intermediate;
    int layers = cfg->dec_layers;

    dctx->n_layers = layers;

    /* Allocate activation buffers */
    size_t vram = 0;
#define ALLOC_D(ptr, count) do {                                          \
    size_t bytes = (size_t)(count) * sizeof(float);                        \
    cudaError_t e = cudaMalloc((void **)&(ptr), bytes);                    \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "GPU dec: alloc failed for " #ptr ": %s\n",       \
                cudaGetErrorString(e));                                     \
        goto fail;                                                         \
    }                                                                      \
    vram += bytes;                                                         \
} while (0)

    ALLOC_D(dctx->d_x, hidden);
    ALLOC_D(dctx->d_x_norm, hidden);
    ALLOC_D(dctx->d_q, q_dim);
    ALLOC_D(dctx->d_k, kv_dim);
    ALLOC_D(dctx->d_v, kv_dim);
    ALLOC_D(dctx->d_attn_out, q_dim);
    ALLOC_D(dctx->d_proj_out, hidden);
    ALLOC_D(dctx->d_gate_buf, 2 * inter);
    ALLOC_D(dctx->d_ffn_out, hidden);
    ALLOC_D(dctx->d_logits, cfg->vocab_size);
    dctx->vram_activations = vram;

    /* Device scalars for results */
    cudaMalloc((void **)&dctx->d_argmax_result, sizeof(int));
    cudaMalloc((void **)&dctx->d_probe_result, sizeof(int));

    /* Pinned host memory for async download */
    cudaMallocHost((void **)&dctx->h_argmax_result, sizeof(int));
    cudaMallocHost((void **)&dctx->h_probe_result, sizeof(int));

    /* Allocate norm weight pointer arrays */
    dctx->d_input_norm = (float **)calloc(layers, sizeof(float *));
    dctx->d_post_attn_norm = (float **)calloc(layers, sizeof(float *));
    dctx->d_q_norm = (float **)calloc(layers, sizeof(float *));
    dctx->d_k_norm = (float **)calloc(layers, sizeof(float *));

    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU dec: activation buffers %.1f KB, logits %.1f MB\n",
                dctx->vram_activations / 1024.0,
                (size_t)cfg->vocab_size * sizeof(float) / (1024.0 * 1024.0));

    /* Initial KV cache: 1024 positions */
    if (qwen_gpu_kv_cache_grow(dctx, cfg, 1024) != 0)
        goto fail;

    return dctx;

fail:
    qwen_gpu_dec_free(dctx);
    return NULL;

#undef ALLOC_D
}

void qwen_gpu_dec_free(qwen_gpu_dec_ctx_t *dctx) {
    if (!dctx) return;

    /* Free activation buffers */
    if (dctx->d_x) cudaFree(dctx->d_x);
    if (dctx->d_x_norm) cudaFree(dctx->d_x_norm);
    if (dctx->d_q) cudaFree(dctx->d_q);
    if (dctx->d_k) cudaFree(dctx->d_k);
    if (dctx->d_v) cudaFree(dctx->d_v);
    if (dctx->d_attn_out) cudaFree(dctx->d_attn_out);
    if (dctx->d_proj_out) cudaFree(dctx->d_proj_out);
    if (dctx->d_gate_buf) cudaFree(dctx->d_gate_buf);
    if (dctx->d_ffn_out) cudaFree(dctx->d_ffn_out);
    if (dctx->d_logits) cudaFree(dctx->d_logits);

    /* Free device scalars */
    if (dctx->d_argmax_result) cudaFree(dctx->d_argmax_result);
    if (dctx->d_probe_result) cudaFree(dctx->d_probe_result);

    /* Free pinned host memory */
    if (dctx->h_argmax_result) cudaFreeHost(dctx->h_argmax_result);
    if (dctx->h_probe_result) cudaFreeHost(dctx->h_probe_result);

    /* Free KV cache */
    if (dctx->d_kv_k) cudaFree(dctx->d_kv_k);
    if (dctx->d_kv_v) cudaFree(dctx->d_kv_v);

    /* Free RoPE cache */
    if (dctx->d_rope_cos) cudaFree(dctx->d_rope_cos);
    if (dctx->d_rope_sin) cudaFree(dctx->d_rope_sin);

    /* Free per-layer norm weights */
    for (int i = 0; i < dctx->n_layers; i++) {
        if (dctx->d_input_norm && dctx->d_input_norm[i]) cudaFree(dctx->d_input_norm[i]);
        if (dctx->d_post_attn_norm && dctx->d_post_attn_norm[i]) cudaFree(dctx->d_post_attn_norm[i]);
        if (dctx->d_q_norm && dctx->d_q_norm[i]) cudaFree(dctx->d_q_norm[i]);
        if (dctx->d_k_norm && dctx->d_k_norm[i]) cudaFree(dctx->d_k_norm[i]);
    }
    free(dctx->d_input_norm);
    free(dctx->d_post_attn_norm);
    free(dctx->d_q_norm);
    free(dctx->d_k_norm);
    if (dctx->d_final_norm) cudaFree(dctx->d_final_norm);

    /* Unload CUBIN */
    if (dctx->module) cuModuleUnload(dctx->module);

    free(dctx);
}

/* ========================================================================
 * KV Cache Management (device)
 * ======================================================================== */

int qwen_gpu_kv_cache_grow(qwen_gpu_dec_ctx_t *dctx,
                            const qwen_config_t *cfg, int max_seq) {
    if (!dctx || !cfg) return -1;
    if (max_seq <= dctx->kv_cache_max) return 0;

    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int layers = cfg->dec_layers;

    /* Total size for all layers contiguous: [layers, max_seq, kv_dim] */
    size_t total_floats = (size_t)layers * max_seq * kv_dim;
    size_t total_bytes = total_floats * sizeof(float);

    float *new_k = NULL, *new_v = NULL;
    cudaError_t e = cudaMalloc((void **)&new_k, total_bytes);
    if (e != cudaSuccess) return -1;
    e = cudaMalloc((void **)&new_v, total_bytes);
    if (e != cudaSuccess) { cudaFree(new_k); return -1; }

    /* Zero-init for safety */
    cudaMemset(new_k, 0, total_bytes);
    cudaMemset(new_v, 0, total_bytes);

    /* Copy old data if exists (per-layer, since stride changed) */
    if (dctx->d_kv_k && dctx->kv_cache_max > 0) {
        size_t old_stride = (size_t)dctx->kv_cache_max * kv_dim;
        size_t new_stride = (size_t)max_seq * kv_dim;
        size_t copy_bytes = old_stride * sizeof(float);
        if (copy_bytes > new_stride * sizeof(float))
            copy_bytes = new_stride * sizeof(float);

        for (int l = 0; l < layers; l++) {
            cudaMemcpy(new_k + l * new_stride,
                       dctx->d_kv_k + l * old_stride,
                       copy_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_v + l * new_stride,
                       dctx->d_kv_v + l * old_stride,
                       copy_bytes, cudaMemcpyDeviceToDevice);
        }
    }

    if (dctx->d_kv_k) cudaFree(dctx->d_kv_k);
    if (dctx->d_kv_v) cudaFree(dctx->d_kv_v);
    dctx->d_kv_k = new_k;
    dctx->d_kv_v = new_v;
    dctx->vram_kv_cache = 2 * total_bytes;
    dctx->kv_cache_max = max_seq;

    if (qwen_verbose >= 2)
        fprintf(stderr, "GPU dec: KV cache grown to %d positions (%.1f MB)\n",
                max_seq, 2 * total_bytes / (1024.0 * 1024.0));
    return 0;
}

void qwen_gpu_kv_cache_reset(qwen_gpu_dec_ctx_t *dctx) {
    if (dctx) dctx->kv_synced_len = 0;
}

/* Upload CPU KV cache to GPU for positions [synced_len, cpu_len).
 * Called before the first GPU decode after a CPU prefill. */
int qwen_gpu_kv_cache_sync(qwen_gpu_dec_ctx_t *dctx,
                              const qwen_config_t *cfg,
                              const float *cpu_kv_k, const float *cpu_kv_v,
                              int cpu_max, int cpu_len) {
    if (!dctx || cpu_len <= dctx->kv_synced_len) return 0;

    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int layers = cfg->dec_layers;

    /* Ensure GPU cache can hold cpu_len positions */
    if (cpu_len > dctx->kv_cache_max) {
        int new_max = dctx->kv_cache_max;
        while (new_max < cpu_len) new_max *= 2;
        if (qwen_gpu_kv_cache_grow(dctx, cfg, new_max) != 0) return -1;
    }

    /* Upload per-layer: copy positions [synced_len, cpu_len) */
    int start = dctx->kv_synced_len;
    int count = cpu_len - start;
    size_t copy_bytes = (size_t)count * kv_dim * sizeof(float);
    size_t cpu_stride = (size_t)cpu_max * kv_dim;
    size_t gpu_stride = (size_t)dctx->kv_cache_max * kv_dim;

    for (int l = 0; l < layers; l++) {
        const float *src_k = cpu_kv_k + l * cpu_stride + (size_t)start * kv_dim;
        const float *src_v = cpu_kv_v + l * cpu_stride + (size_t)start * kv_dim;
        float *dst_k = dctx->d_kv_k + l * gpu_stride + (size_t)start * kv_dim;
        float *dst_v = dctx->d_kv_v + l * gpu_stride + (size_t)start * kv_dim;

        cudaMemcpy(dst_k, src_k, copy_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dst_v, src_v, copy_bytes, cudaMemcpyHostToDevice);
    }

    dctx->kv_synced_len = cpu_len;

    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU dec: synced KV cache positions [%d, %d) to device (%.1f MB)\n",
                start, cpu_len,
                2.0 * count * kv_dim * sizeof(float) * layers / (1024.0 * 1024.0));
    return 0;
}

/* ========================================================================
 * RoPE Upload
 * ======================================================================== */

int qwen_gpu_upload_rope(qwen_gpu_dec_ctx_t *dctx,
                          const float *cos_table, const float *sin_table,
                          int n_pos, int head_dim) {
    if (!dctx) return -1;
    if (n_pos <= dctx->rope_cache_cap) return 0;

    size_t bytes = (size_t)n_pos * head_dim * sizeof(float);

    /* Free old */
    if (dctx->d_rope_cos) cudaFree(dctx->d_rope_cos);
    if (dctx->d_rope_sin) cudaFree(dctx->d_rope_sin);

    cudaError_t e;
    e = cudaMalloc((void **)&dctx->d_rope_cos, bytes);
    if (e != cudaSuccess) return -1;
    e = cudaMalloc((void **)&dctx->d_rope_sin, bytes);
    if (e != cudaSuccess) { cudaFree(dctx->d_rope_cos); dctx->d_rope_cos = NULL; return -1; }

    cudaMemcpy(dctx->d_rope_cos, cos_table, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dctx->d_rope_sin, sin_table, bytes, cudaMemcpyHostToDevice);
    dctx->rope_cache_cap = n_pos;
    dctx->vram_rope = 2 * bytes;

    if (qwen_verbose >= 2)
        fprintf(stderr, "GPU dec: RoPE cache uploaded (%d positions, %.1f KB)\n",
                n_pos, 2 * bytes / 1024.0);
    return 0;
}

/* ========================================================================
 * Norm Weight Upload
 * ======================================================================== */

int qwen_gpu_upload_norm_weights(qwen_gpu_dec_ctx_t *dctx,
                                  const qwen_decoder_t *dec,
                                  const qwen_config_t *cfg) {
    if (!dctx || !dec || !cfg) return -1;

    int hidden = cfg->dec_hidden;
    int head_dim = cfg->dec_head_dim;
    int layers = cfg->dec_layers;
    size_t vram = 0;

    for (int i = 0; i < layers; i++) {
        const qwen_dec_layer_t *l = &dec->layers[i];
        size_t h_bytes = hidden * sizeof(float);
        size_t hd_bytes = head_dim * sizeof(float);

        cudaMalloc((void **)&dctx->d_input_norm[i], h_bytes);
        cudaMemcpy(dctx->d_input_norm[i], l->input_norm, h_bytes, cudaMemcpyHostToDevice);
        vram += h_bytes;

        cudaMalloc((void **)&dctx->d_post_attn_norm[i], h_bytes);
        cudaMemcpy(dctx->d_post_attn_norm[i], l->post_attn_norm, h_bytes, cudaMemcpyHostToDevice);
        vram += h_bytes;

        cudaMalloc((void **)&dctx->d_q_norm[i], hd_bytes);
        cudaMemcpy(dctx->d_q_norm[i], l->q_norm_weight, hd_bytes, cudaMemcpyHostToDevice);
        vram += hd_bytes;

        cudaMalloc((void **)&dctx->d_k_norm[i], hd_bytes);
        cudaMemcpy(dctx->d_k_norm[i], l->k_norm_weight, hd_bytes, cudaMemcpyHostToDevice);
        vram += hd_bytes;
    }

    /* Final norm */
    size_t h_bytes = hidden * sizeof(float);
    cudaMalloc((void **)&dctx->d_final_norm, h_bytes);
    cudaMemcpy(dctx->d_final_norm, dec->norm, h_bytes, cudaMemcpyHostToDevice);
    vram += h_bytes;

    dctx->vram_norms = vram;
    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU dec: norm weights uploaded (%.1f KB)\n", vram / 1024.0);
    return 0;
}

/* ========================================================================
 * Full GPU Decoder Forward (Single Token)
 * ======================================================================== */

int qwen_gpu_decoder_forward(qwen_gpu_dec_ctx_t *dctx,
                              qwen_gpu_ctx_t *gpu,
                              const qwen_decoder_t *dec,
                              const qwen_config_t *cfg,
                              const float *input_embed,
                              int pos,
                              int probe_layer,
                              int enc_kv_start,
                              int enc_kv_count,
                              int *out_peak_enc_pos) {
    if (!dctx || !gpu || !dec || !cfg || !input_embed)
        return QWEN_TOKEN_IM_END;

    cublasHandle_t cblas = (cublasHandle_t)qwen_gpu_get_cublas_handle(gpu);
    int hidden = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float eps = cfg->dec_rms_norm_eps;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Ensure KV cache is large enough */
    if (pos >= dctx->kv_cache_max) {
        int new_max = dctx->kv_cache_max;
        while (new_max <= pos) new_max *= 2;
        if (qwen_gpu_kv_cache_grow(dctx, cfg, new_max) != 0)
            return QWEN_TOKEN_IM_END;
    }

    /* Upload input embedding to device (single ~8KB transfer) */
    cudaMemcpy(dctx->d_x, input_embed, hidden * sizeof(float), cudaMemcpyHostToDevice);

    /* RoPE cos/sin for this position */
    float *d_rope_cos_pos = dctx->d_rope_cos + (size_t)pos * head_dim;
    float *d_rope_sin_pos = dctx->d_rope_sin + (size_t)pos * head_dim;

    /* Resolve probe layer */
    if (probe_layer < 0)
        probe_layer = (cfg->dec_layers * 4 + 3) / 7;
    if (probe_layer >= cfg->dec_layers)
        probe_layer = cfg->dec_layers - 1;

    /* Per-layer device KV cache accessor */
    size_t kv_layer_stride = (size_t)dctx->kv_cache_max * kv_dim;

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        const qwen_dec_layer_t *l = &dec->layers[layer];

        /* --- Input RMSNorm --- */
        {
            int rows = 1;
            void *args[] = {
                &dctx->d_x_norm, &dctx->d_x,
                &dctx->d_input_norm[layer],
                &rows, &hidden, &eps
            };
            launch_kernel(dctx->kernels[KF_RMS_NORM],
                          1, 1, 1, 256, 1, 1, 0, args);
        }

        /* --- QKV projections (device-to-device GEMMs) --- */
        {
            int wq_h = qwen_gpu_find_weight(gpu, l->wq_weight_bf16);
            int wk_h = qwen_gpu_find_weight(gpu, l->wk_weight_bf16);
            int wv_h = qwen_gpu_find_weight(gpu, l->wv_weight_bf16);
            float *d_wq = qwen_gpu_get_weight_ptr(gpu, wq_h);
            float *d_wk = qwen_gpu_get_weight_ptr(gpu, wk_h);
            float *d_wv = qwen_gpu_get_weight_ptr(gpu, wv_h);

            /* Q[1, q_dim] = x_norm[1, hidden] @ Wq[q_dim, hidden]^T */
            gpu_gemm_d2d(cblas, dctx->d_q, dctx->d_x_norm, d_wq, 1, hidden, q_dim);
            gpu_gemm_d2d(cblas, dctx->d_k, dctx->d_x_norm, d_wk, 1, hidden, kv_dim);
            gpu_gemm_d2d(cblas, dctx->d_v, dctx->d_x_norm, d_wv, 1, hidden, kv_dim);
        }

        /* --- Per-head Q/K RMSNorm --- */
        {
            void *args_q[] = {
                &dctx->d_q, &dctx->d_q_norm[layer],
                &n_heads, &head_dim, &eps
            };
            launch_kernel(dctx->kernels[KF_RMS_NORM_PER_HEAD],
                          (unsigned)n_heads, 1, 1, 128, 1, 1, 0, args_q);

            void *args_k[] = {
                &dctx->d_k, &dctx->d_k_norm[layer],
                &n_kv_heads, &head_dim, &eps
            };
            launch_kernel(dctx->kernels[KF_RMS_NORM_PER_HEAD],
                          (unsigned)n_kv_heads, 1, 1, 128, 1, 1, 0, args_k);
        }

        /* --- Apply NeoX RoPE --- */
        {
            int half_dim = head_dim / 2;
            void *args_q[] = {
                &dctx->d_q, &d_rope_cos_pos, &d_rope_sin_pos,
                &n_heads, &head_dim
            };
            launch_kernel(dctx->kernels[KF_APPLY_ROPE_NEOX],
                          (unsigned)n_heads, 1, 1, (unsigned)half_dim, 1, 1, 0, args_q);

            void *args_k[] = {
                &dctx->d_k, &d_rope_cos_pos, &d_rope_sin_pos,
                &n_kv_heads, &head_dim
            };
            launch_kernel(dctx->kernels[KF_APPLY_ROPE_NEOX],
                          (unsigned)n_kv_heads, 1, 1, (unsigned)half_dim, 1, 1, 0, args_k);
        }

        /* --- Append K,V to device KV cache --- */
        {
            float *d_kv_k_layer = dctx->d_kv_k + (size_t)layer * kv_layer_stride;
            float *d_kv_v_layer = dctx->d_kv_v + (size_t)layer * kv_layer_stride;
            unsigned int grid = ((unsigned)kv_dim + 255) / 256;
            void *args[] = {
                &d_kv_k_layer, &d_kv_v_layer,
                &dctx->d_k, &dctx->d_v,
                &pos, &kv_dim
            };
            launch_kernel(dctx->kernels[KF_KV_APPEND],
                          grid, 1, 1, 256, 1, 1, 0, args);
        }

        /* --- GQA Attention --- */
        {
            int total_seq = pos + 1;
            float *d_kv_k_layer = dctx->d_kv_k + (size_t)layer * kv_layer_stride;
            float *d_kv_v_layer = dctx->d_kv_v + (size_t)layer * kv_layer_stride;
            void *args[] = {
                &dctx->d_attn_out, &dctx->d_q,
                &d_kv_k_layer, &d_kv_v_layer,
                &total_seq, &n_kv_heads, &head_dim, &scale
            };
            launch_kernel(dctx->kernels[KF_ATTN_GQA2],
                          (unsigned)n_heads, 1, 1, 32, 1, 1, 0, args);
        }

        /* --- Attention probe (timestamp alignment) --- */
        if (layer == probe_layer && enc_kv_count > 0 && enc_kv_start >= 0) {
            int enc_end = enc_kv_start + enc_kv_count;
            int total_seq = pos + 1;
            if (enc_end > total_seq) enc_end = total_seq;

            float *d_kv_k_layer = dctx->d_kv_k + (size_t)layer * kv_layer_stride;
            void *args[] = {
                &dctx->d_probe_result, &dctx->d_q,
                &d_kv_k_layer,
                &enc_kv_start, &enc_end,
                &n_heads, &n_kv_heads, &head_dim, &scale
            };
            launch_kernel(dctx->kernels[KF_ATTN_PROBE],
                          1, 1, 1, 256, 1, 1, 0, args);
        }

        /* --- Output projection + residual --- */
        {
            int wo_h = qwen_gpu_find_weight(gpu, l->wo_weight_bf16);
            float *d_wo = qwen_gpu_get_weight_ptr(gpu, wo_h);
            gpu_gemm_d2d(cblas, dctx->d_proj_out, dctx->d_attn_out, d_wo, 1, q_dim, hidden);

            unsigned int grid = ((unsigned)hidden + 255) / 256;
            void *args[] = { &dctx->d_x, &dctx->d_proj_out, &hidden };
            launch_kernel(dctx->kernels[KF_ADD_INPLACE],
                          grid, 1, 1, 256, 1, 1, 0, args);
        }

        /* --- Post-attention RMSNorm --- */
        {
            int rows = 1;
            void *args[] = {
                &dctx->d_x_norm, &dctx->d_x,
                &dctx->d_post_attn_norm[layer],
                &rows, &hidden, &eps
            };
            launch_kernel(dctx->kernels[KF_RMS_NORM],
                          1, 1, 1, 256, 1, 1, 0, args);
        }

        /* --- SwiGLU MLP --- */
        {
            /* Fused gate+up GEMM: gate_buf[1, 2*inter] = x_norm[1, hidden] @ W[2*inter, hidden]^T */
            int wgu_h = qwen_gpu_find_weight(gpu, l->gate_up_fused_bf16);
            float *d_wgu = qwen_gpu_get_weight_ptr(gpu, wgu_h);
            gpu_gemm_d2d(cblas, dctx->d_gate_buf, dctx->d_x_norm, d_wgu, 1, hidden, 2 * intermediate);

            /* SwiGLU: ffn_out[inter] = SiLU(gate_buf[even]) * gate_buf[odd] */
            unsigned int grid = ((unsigned)intermediate + 255) / 256;
            void *swiglu_args[] = {
                &dctx->d_ffn_out, &dctx->d_gate_buf, &intermediate
            };
            launch_kernel(dctx->kernels[KF_SWIGLU_INTERLEAVED],
                          grid, 1, 1, 256, 1, 1, 0, swiglu_args);

            /* Down projection: proj_out[1, hidden] = ffn_out[1, inter] @ Wdown[hidden, inter]^T */
            int wd_h = qwen_gpu_find_weight(gpu, l->down_weight_bf16);
            float *d_wd = qwen_gpu_get_weight_ptr(gpu, wd_h);
            gpu_gemm_d2d(cblas, dctx->d_proj_out, dctx->d_ffn_out, d_wd, 1, intermediate, hidden);

            /* Residual add */
            void *add_args[] = { &dctx->d_x, &dctx->d_proj_out, &hidden };
            launch_kernel(dctx->kernels[KF_ADD_INPLACE],
                          ((unsigned)hidden + 255) / 256, 1, 1, 256, 1, 1, 0, add_args);
        }
    }

    /* --- Final RMSNorm (in-place on d_x) --- */
    {
        /* Use d_x_norm as temp output, then copy back.
         * Actually, since we use d_x as both input and will need d_x for
         * the lm_head matvec, we normalize into d_x_norm. */
        int rows = 1;
        void *args[] = {
            &dctx->d_x_norm, &dctx->d_x,
            &dctx->d_final_norm,
            &rows, &hidden, &eps
        };
        launch_kernel(dctx->kernels[KF_RMS_NORM],
                      1, 1, 1, 256, 1, 1, 0, args);
    }

    /* --- LM head (tied embeddings): logits = W_embed @ x_norm --- */
    {
        int we_h = qwen_gpu_find_weight(gpu, dec->tok_embeddings_bf16);
        float *d_we = qwen_gpu_get_weight_ptr(gpu, we_h);
        /* logits[1, vocab] = x_norm[1, hidden] @ W_embed[vocab, hidden]^T */
        gpu_gemm_d2d(cblas, dctx->d_logits, dctx->d_x_norm, d_we, 1, hidden, cfg->vocab_size);
    }

    /* --- Argmax on device --- */
    {
        int n = cfg->vocab_size;
        void *args[] = { &dctx->d_argmax_result, &dctx->d_logits, &n };
        launch_kernel(dctx->kernels[KF_ARGMAX],
                      1, 1, 1, 256, 1, 1, 0, args);
    }

    /* --- Download results (4 + 4 bytes) --- */
    cudaMemcpy(dctx->h_argmax_result, dctx->d_argmax_result,
               sizeof(int), cudaMemcpyDeviceToHost);

    if (out_peak_enc_pos && enc_kv_count > 0) {
        cudaMemcpy(dctx->h_probe_result, dctx->d_probe_result,
                   sizeof(int), cudaMemcpyDeviceToHost);
        *out_peak_enc_pos = *dctx->h_probe_result;
    }

    /* Mark this position as synced (GPU wrote it via KV_APPEND kernel) */
    dctx->kv_synced_len = pos + 1;

    return *dctx->h_argmax_result;
}

#endif /* USE_CUDA_KERNELS */
