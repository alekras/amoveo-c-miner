/*
 * sha256_gpu.h
 *
 *  Created on: Aug 2, 2018
 *      Author: alexei
 */

#ifndef SRC_C_SHA256_GPU_H_
#define SRC_C_SHA256_GPU_H_

typedef struct {
  BYTE flag;
  WORD blockIdx;
  WORD blockIdy;
  WORD threadIdx;
  WORD threadIdy;
} GPU_thread_info;

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, GPU_thread_info * info_debug);
__global__ void kernel_sha256_val(BYTE *data, WORD len, BYTE *hash, WORD cycle);
__device__ WORD hash2int(BYTE h[32]);
__device__ WORD hash2int_w(WORD h[8]);
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void d_sha256_init(SHA256_CTX *ctx);
__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[]);

#endif /* SRC_C_SHA256_GPU_H_ */
