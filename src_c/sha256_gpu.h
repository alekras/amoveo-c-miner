/*
 * sha256_gpu.h
 *
 *  Created on: Aug 2, 2018
 *      Author: alexei
 */

#ifndef SRC_C_SHA256_GPU_H_
#define SRC_C_SHA256_GPU_H_

typedef struct {
  long int flag;
  WORD blockIdx;
  WORD blockIdy;
  WORD threadIdx;
  WORD threadIdy;
} GPU_thread_info;

typedef struct {
  WORD data[16];
//  WORD datalen;
//  unsigned long long bitlen;
  WORD state[8];
} AMO_SHA256_CTX;

__global__ void kernel_test(volatile bool *stop, volatile long int *cycles, GPU_thread_info * info_debug);
__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, long int * cycles_total);
__global__ void kernel_sha256_val(BYTE *data, WORD len, BYTE *hash, WORD cycle);
__device__ WORD hash2int(BYTE h[32]);
__device__ WORD hash2int_w(WORD h[8]);
__device__ bool sha256_transform(AMO_SHA256_CTX *ctx);
__device__ void sha256_msg_scheduler(AMO_SHA256_CTX *ctx, WORD *m);
__device__ bool sha256_msg_compression(AMO_SHA256_CTX *ctx, WORD *m);
__device__ void sha256_init(AMO_SHA256_CTX *ctx);
//__device__ void sha256_update(AMO_SHA256_CTX *ctx, const BYTE data[], size_t len);
//__device__ void sha256_final(AMO_SHA256_CTX *ctx, BYTE hash[]);
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void d_sha256_init(SHA256_CTX *ctx);
__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[]);

#endif /* SRC_C_SHA256_GPU_H_ */
