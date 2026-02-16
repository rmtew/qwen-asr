/*
 * qwen_asr_gpu.h - GPU acceleration for Qwen3-ASR
 *
 * Two tiers of GPU support:
 *   USE_CUBLAS        - cuBLAS GEMM offload (activations round-trip CPU<->GPU)
 *   USE_CUDA_KERNELS  - Full GPU decoder (custom CUDA kernels + cuBLAS GEMMs,
 *                       activations stay on device, only embed in + token out)
 *
 * USE_CUDA_KERNELS implies USE_CUBLAS (weights already uploaded as f32).
 */

#ifndef QWEN_ASR_GPU_H
#define QWEN_ASR_GPU_H

#include <stdint.h>

#ifdef USE_CUBLAS

typedef struct qwen_gpu_ctx qwen_gpu_ctx_t;

/* Initialize cuBLAS context. Returns NULL if CUDA unavailable or init fails. */
qwen_gpu_ctx_t *qwen_gpu_init(void);

/* Free GPU context and all device memory. */
void qwen_gpu_free(qwen_gpu_ctx_t *gpu);

/* Upload f32 weight matrix to GPU.
 * host_ptr is used as the lookup key for qwen_gpu_find_weight().
 * Returns 0 on success, -1 on error. */
int qwen_gpu_upload_weight_f32(qwen_gpu_ctx_t *gpu, const float *host_ptr,
                                int rows, int cols);

/* Upload bf16 weight matrix to GPU (converted to f32 on upload).
 * host_ptr (bf16) is used as the lookup key.
 * Returns 0 on success, -1 on error. */
int qwen_gpu_upload_weight_bf16(qwen_gpu_ctx_t *gpu, const uint16_t *host_ptr,
                                 int rows, int cols);

/* Find GPU weight handle by CPU pointer. Returns handle >= 0 or -1. */
int qwen_gpu_find_weight(qwen_gpu_ctx_t *gpu, const void *host_ptr);

/* Get device pointer for a weight by handle. Returns NULL if invalid. */
float *qwen_gpu_get_weight_ptr(qwen_gpu_ctx_t *gpu, int handle);

/* GPU GEMM: C[M,N] = A[M,K] @ W[N,K]^T  (row-major, matches qwen_linear)
 * A is uploaded per-call from CPU. W is already on GPU (by handle).
 * C is downloaded to CPU after computation. */
void qwen_gpu_gemm(qwen_gpu_ctx_t *gpu, float *C_host,
                   const float *A_host, int weight_handle,
                   int M, int K, int N);

/* GPU Conv2D GEMM: out[c_out, sp] = W[c_out, ps] @ cols[ps, sp]
 * W is looked up on GPU by host pointer. cols is uploaded per-call. */
void qwen_gpu_conv2d_gemm(qwen_gpu_ctx_t *gpu, float *out_host,
                            const float *cols_host, const float *weight_host_key,
                            int c_out, int patch_size, int spatial_out);

/* GPU matvec + argmax: computes W[out_dim,in_dim] @ x[in_dim] and returns
 * the index of the maximum element. Used for decoder lm_head. */
int qwen_gpu_argmax_matvec(qwen_gpu_ctx_t *gpu,
                            const float *x_host, int weight_handle,
                            int in_dim, int out_dim);

/* Set FP16 upload mode. When enabled, qwen_gpu_upload_weight_bf16() converts
 * BF16->FP16 (2 bytes/element) instead of BF16->F32 (4 bytes/element).
 * GEMM automatically dispatches to cublasGemmEx for FP16 weights. */
void qwen_gpu_set_fp16_mode(qwen_gpu_ctx_t *gpu, int enable);

/* Query whether a weight (by handle) is stored as FP16. Returns 1 if FP16, 0 if F32. */
int qwen_gpu_weight_is_fp16(qwen_gpu_ctx_t *gpu, int handle);

/* Print GPU memory usage stats to stderr. */
void qwen_gpu_print_stats(qwen_gpu_ctx_t *gpu);

/* Get the cuBLAS handle (for use by GPU decoder context). */
void *qwen_gpu_get_cublas_handle(qwen_gpu_ctx_t *gpu);

#endif /* USE_CUBLAS */

/* ========================================================================
 * Full GPU Decoder (USE_CUDA_KERNELS)
 *
 * Keeps all activations on GPU for the entire decoder forward pass.
 * Custom CUDA kernels handle non-GEMM ops; cuBLAS handles GEMMs.
 * Only transfers: input embedding H2D (8KB), argmax result D2H (4 bytes),
 * probe result D2H (4 bytes).
 * ======================================================================== */

#ifdef USE_CUDA_KERNELS

#include "qwen_asr.h"

/* Forward declarations -- must match qwen_asr.h typedefs.
 * These are structs underlying the typedefs in qwen_asr.h. Callers that
 * include qwen_asr.h before this header get the full definitions. */
typedef struct qwen_gpu_dec_ctx qwen_gpu_dec_ctx_t;
typedef struct qwen_gpu_enc_ctx qwen_gpu_enc_ctx_t;

/* Create GPU decoder context. Loads CUBIN, allocates device buffers.
 * Returns NULL if CUBIN loading fails (arch mismatch, etc). */
qwen_gpu_dec_ctx_t *qwen_gpu_dec_init(qwen_gpu_ctx_t *gpu,
                                       const qwen_config_t *cfg);

/* Free GPU decoder context and all device buffers. */
void qwen_gpu_dec_free(qwen_gpu_dec_ctx_t *dctx);

/* Full GPU decoder forward pass (single token).
 * input_embed: host pointer to [hidden] f32 embedding
 * pos: KV cache position for this token
 * probe_layer: which layer to run attention probe (-1 for auto)
 * enc_kv_start/enc_kv_count: encoder token range in KV cache
 * out_peak_enc_pos: receives probe result (KV position of peak attention)
 * Returns: greedy argmax token ID */
int qwen_gpu_decoder_forward(qwen_gpu_dec_ctx_t *dctx,
                              qwen_gpu_ctx_t *gpu,
                              const qwen_decoder_t *dec,
                              const qwen_config_t *cfg,
                              const float *input_embed,
                              int pos,
                              int probe_layer,
                              int enc_kv_start,
                              int enc_kv_count,
                              int *out_peak_enc_pos);

/* Grow device KV cache to hold at least max_seq positions. */
int qwen_gpu_kv_cache_grow(qwen_gpu_dec_ctx_t *dctx,
                            const qwen_config_t *cfg,
                            int max_seq);

/* Reset KV cache length (for new sequence). */
void qwen_gpu_kv_cache_reset(qwen_gpu_dec_ctx_t *dctx);

/* Upload RoPE cos/sin tables to device.
 * cos/sin: [n_pos, head_dim] */
int qwen_gpu_upload_rope(qwen_gpu_dec_ctx_t *dctx,
                          const float *cos_table, const float *sin_table,
                          int n_pos, int head_dim);

/* Upload per-layer norm weights to device. */
int qwen_gpu_upload_norm_weights(qwen_gpu_dec_ctx_t *dctx,
                                  const qwen_decoder_t *dec,
                                  const qwen_config_t *cfg);

/* Sync CPU KV cache to GPU. Uploads positions [already_synced, cpu_len).
 * Must be called before the first GPU decode after a CPU prefill. */
int qwen_gpu_kv_cache_sync(qwen_gpu_dec_ctx_t *dctx,
                            const qwen_config_t *cfg,
                            const float *cpu_kv_k, const float *cpu_kv_v,
                            int cpu_max, int cpu_len);

/* ========================================================================
 * Full GPU Encoder (USE_CUDA_KERNELS)
 *
 * Keeps activations on GPU through the transformer layers.
 * Custom CUDA kernels: LayerNorm, GELU, bias_add, add_inplace.
 * cuBLAS handles all GEMM operations (device-to-device).
 * Attention computed on CPU (QKV downloaded, result uploaded).
 * Conv2D stem stays on CPU (small data, runs once per chunk).
 * ======================================================================== */

/* Create GPU encoder context. Borrows kernel handles from the decoder's
 * CUBIN module, uploads per-layer norm/bias weights.
 * Returns NULL on failure. */
qwen_gpu_enc_ctx_t *qwen_gpu_enc_init(qwen_gpu_ctx_t *gpu,
                                        qwen_gpu_dec_ctx_t *dctx,
                                        const qwen_encoder_t *enc,
                                        const qwen_config_t *cfg);

/* Free GPU encoder context and all device buffers. */
void qwen_gpu_enc_free(qwen_gpu_enc_ctx_t *ectx);

/* GPU encoder transformer forward pass.
 * x_host: [seq, d_model] input (post conv2d stem + position embeddings).
 * enc_output_host: [seq, output_dim] output (caller allocates).
 * window_starts[n_windows+1]: attention window boundaries.
 * Returns 0 on success. */
int qwen_gpu_encoder_forward(qwen_gpu_enc_ctx_t *ectx,
                               qwen_gpu_ctx_t *gpu,
                               const qwen_encoder_t *enc,
                               const qwen_config_t *cfg,
                               float *x_host,
                               float *enc_output_host,
                               int seq,
                               const int *window_starts,
                               int n_windows);

#endif /* USE_CUDA_KERNELS */

#endif /* QWEN_ASR_GPU_H */
