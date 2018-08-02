#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
  #include "sha256.h"
  extern FILE *fdebug;
}
typedef struct {
  BYTE flag;
  WORD blockIdx;
  WORD blockIdy;
  WORD threadIdx;
  WORD threadIdy;
} GPU_thread_info;
/*
 MultiProcessor Count = 9
 Maximum number of threads per block = 1024
 maxThreadsDim[3] contains the maximum size of each dimension of a block = [1024,1024,64]
 maxGridSize[3] contains the maximum size of each dimension of a grid = [2147483647,65535,65535]
 canMapHostMemory is 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not = 1
 asyncEngineCount is 1 when the device can concurrently copy memory between host and device while executing a kernel.
   It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.
   It is 0 if neither of these is supported = 2
 unifiedAddressing is 1 if the device shares a unified address space with the host and 0 otherwise = 1
 maxThreadsPerMultiProcessor is the number of maximum resident threads per multiprocessor = 2048

 */
__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, GPU_thread_info * info_debug);
__global__ void kernel_sha256_val(BYTE *data, WORD len, BYTE *hash, WORD cycle, volatile bool *stop);
__device__ WORD hash2int(BYTE h[32]);
__device__ WORD hash2int_w(WORD h[8]);
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void d_sha256_init(SHA256_CTX *ctx);
__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[]);

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(fdebug,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
    fflush(fdebug);
    if (abort) exit(code);
  }
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__, true); }

static GPU_thread_info *h_info_debug;
static GPU_thread_info *d_info_debug;
static struct timeval t_start, t_end;
static int GDIM, BDIM;

//Allocate space on Global Memory
static BYTE *h_data, *d_data;
static BYTE *h_hash, *d_hash;
static BYTE *h_nonce, *d_nonce;
static bool *h_success, *d_success;
static bool *h_stop, *d_stop;
static long int *h_cycles, *d_cycles;

extern "C" bool amoveo_update_gpu(BYTE *nonce, BYTE *data) {
  fprintf(stderr,"GPU: >>> Amoveo update *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);

  if (*h_stop) {
  } else {
    *h_stop = true;
    cudaDeviceSynchronize();
  }
  fprintf(stderr,"GPU: --- Amoveo update *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);

  bool success_ret = *h_success;
  if (success_ret) {
    fprintf(stderr,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);
    fprintf(fdebug,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);
    memcpy(nonce, h_nonce, 23 * sizeof(BYTE));
    memcpy(data, h_data, 32 * sizeof(BYTE));
    *h_success = false;
  }

  fprintf(stderr,"GPU: <<< Amoveo update *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);
  return success_ret;
}

extern "C" bool amoveo_stop_gpu(BYTE *nonce, BYTE *data) {
  fprintf(stderr,"GPU: >>> Amoveo stop *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);
//  *h_stop = true;

//  cudaDeviceSynchronize();

  gettimeofday(&t_end, NULL);
  double numHashes = ((double)GDIM)*((double)GDIM)*((double)BDIM)*((double)(*h_cycles));
  double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);
//  double total_elapsed = ((double)(t_end-t_start))/CLOCKS_PER_SEC;

  fprintf(fdebug,"Cycles = %d  Hash rate = %0.2f MH/s  took time = %0.1f secs\r\n", *h_cycles, numHashes/(1000000.0*total_elapsed), total_elapsed);
  fprintf(fdebug,"Nonce   : ");
  for(int i = 0; i < 23; i++)
      fprintf(fdebug,"%02X.",h_nonce[i]);
  fprintf(fdebug,"\n");
  fprintf(fdebug,"Data    : ");
  for(int i = 0; i < 32; i++)
      fprintf(fdebug,"%02X.",h_data[i]);
  fprintf(fdebug,"\n\n");
//  for (int i = 0; i < 2048; i++) {
//    fprintf(fdebug, "[fl:%d, blk.idx:[%d, %d], thr.idx:[%d, %d]]\n", h_info_debug[i].flag, h_info_debug[i].blockIdx, h_info_debug[i].blockIdy, h_info_debug[i].threadIdx, h_info_debug[i].threadIdy);
//  }
  fflush(fdebug);

//  if (*h_success) {
//    fprintf(stderr,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);
//    fprintf(fdebug,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);
//    memcpy(nonce, h_nonce, 23 * sizeof(BYTE));
//    memcpy(data, h_data, 32 * sizeof(BYTE));
//  }

  fprintf(stderr,"GPU: <<< Amoveo stop h_stop=%d success=%d\r\n", *h_stop, *h_success);
//  return *h_success;
  return false;
}

extern "C" void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle) {
  BYTE *h_data;
  bool *h_stop, *d_stop;
  cudaDeviceProp prop;
  CUDA_SAFE_CALL( cudaGetDeviceProperties (&prop, 0) );
  fprintf(stderr," MultiProcessor Count = %d\n", prop.multiProcessorCount);
  fprintf(stderr," Maximum number of threads per block = %d\n", prop.maxThreadsPerBlock);
  fprintf(stderr," maxThreadsDim[3] contains the maximum size of each dimension of a block = [%d,%d,%d]\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  fprintf(stderr," maxGridSize[3] contains the maximum size of each dimension of a grid = [%d,%d,%d]\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  fprintf(stderr," canMapHostMemory is 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not = %d\n", prop.canMapHostMemory);

  fprintf(stderr," asyncEngineCount is 1 when the device can concurrently copy memory between host and device while executing a kernel.\n");
  fprintf(stderr,"   It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.\n");
  fprintf(stderr,"   It is 0 if neither of these is supported = %d\n", prop.asyncEngineCount);
  fprintf(stderr," unifiedAddressing is 1 if the device shares a unified address space with the host and 0 otherwise = %d\n", prop.unifiedAddressing);
  fprintf(stderr," maxThreadsPerMultiProcessor is the number of maximum resident threads per multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);
  fflush(stderr);

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_stop, sizeof(bool), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_stop, h_stop, 0) );

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_data, len * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_data, h_data, 0) );
//Copy data to device
  memcpy(h_data, data, len * sizeof(BYTE));

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_hash, 32 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_hash, h_hash, 0) );

  *h_stop = true;
  kernel_sha256_val<<<1, 1>>>(d_data, len, d_hash, cycle, d_stop);
  getchar();
  *h_stop = false;

  cudaDeviceSynchronize();

//Copy result back to host
  memcpy(hash, h_hash, 32 * sizeof(BYTE));

//Free memory on device
  CUDA_SAFE_CALL(cudaFreeHost(h_data));
  CUDA_SAFE_CALL(cudaFreeHost(d_hash));
}

extern "C" void amoveo_gpu_alloc_mem() {
  CUDA_SAFE_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_data, 32 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_data, h_data, 0) );

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_nonce, 23 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_nonce, h_nonce, 0) );

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_success, sizeof(bool), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_success, h_success, 0) );

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_stop, sizeof(bool), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_stop, h_stop, 0) );

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_cycles, sizeof(long int), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_cycles, h_cycles, 0) );

//  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 2048 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 4 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_info_debug, h_info_debug, 0) );
  *h_stop = true;
  *h_success = false;
}

extern "C" void amoveo_gpu_free_mem() {
//Free memory on device
  CUDA_SAFE_CALL( cudaFreeHost(d_data) );
  CUDA_SAFE_CALL( cudaFreeHost(d_nonce) );
  CUDA_SAFE_CALL( cudaFreeHost(h_success) );
  CUDA_SAFE_CALL( cudaFreeHost(h_stop) );
  CUDA_SAFE_CALL( cudaFreeHost(h_cycles) );
  CUDA_SAFE_CALL( cudaFreeHost(h_info_debug) );
}

extern "C" void amoveo_mine_gpu(BYTE nonce[23],
                                unsigned int difficulty,
                                BYTE data[32],
                                unsigned int gdim,
                                unsigned int bdim,
                                WORD device_id) {
  GDIM = gdim;
  BDIM = bdim;
//Initialize Cuda Grid variables
  dim3 DimGrid(gdim, gdim);
  dim3 DimBlock(bdim,1);
//  fprintf(stderr,"GPU: >>> Amoveo mine gpu(diff = %d)\r\n", difficulty);

//Copy data to device
  memcpy(h_data, data, 32 * sizeof(BYTE));
  memcpy(h_nonce, nonce, 23 * sizeof(BYTE));
//  fprintf(fdebug,"{Nonce} : ");
//  for(int i = 0; i < 23; i++)
//      fprintf(fdebug,"%02X.",nonce[i]);
//  fprintf(fdebug,"\n");
//  fprintf(fdebug,"{Data}  : ");
//  for(int i = 0; i < 32; i++)
//      fprintf(fdebug,"%02X.",data[i]);
//  fprintf(fdebug,"\n");
//  fflush(fdebug);
//  for (int i = 0; i < 2048; i++) {
//    h_info_debug[i].flag = 0;
//  }

  *h_success = false;
  *h_stop = false;

  gettimeofday(&t_start, NULL);
  kernel_sha256<<<DimGrid, DimBlock>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, device_id, d_info_debug);
//  fprintf(stderr,"GPU: <<< Amoveo mine gpu\r\n");
}

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, GPU_thread_info * info_debug) {
  int i, j, k, work;
  long int r;
  WORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  WORD jdx = blockIdx.y * blockDim.y + threadIdx.y;
  WORD sizeRow = gridDim.x * blockDim.x;
  WORD sizeCol = gridDim.y * blockDim.y;

  BYTE text[55];//32+23
  SHA256_CTX ctx;
  BYTE hash[32];
  for (i = 0; i < 32; i++) {
    text[i] = data[i];
  }

  k = 0;
  for (j = 0; i < 36; i++, k++, j += 8) { // k = 0
    text[i] = nonce[k] ^ ((BYTE)(idx >> j));
  }
  for (j = 0; i < 40; i++, k++, j += 8) { // k = 4
    text[i] = nonce[k] ^ ((BYTE)(jdx >> j));
  }
  text[i++] = nonce[k++] ^ ((BYTE)device_id);

  for (;i < 55; i++, k++) { // i = 41, k = 9
    text[i] = nonce[k];
  }

  r = 0;
  while (!(*stop)) {
    if(*success) {
      break;
    }
    for (i = 41; i < 55; i++) {
      text[i] += 1;
      if (text[i] != 0) {
        break;
      }
    }

    d_sha256_init(&ctx);
    d_sha256_update(&ctx,text,55);
    d_sha256_final(&ctx,hash);
    r++;

    work = hash2int_w(ctx.state);
    if( work > difficulty) {
      *success = true;
      for (i = 0; i < 23; i++) {
        nonce[i] = text[i + 32];
      }
      BYTE * ptr = (BYTE*)&ctx.state;
      for (i = 0; i < 4; ++i) {
        data[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        data[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        data[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        data[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        data[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        data[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        data[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        data[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
      }
//      for (i = 0; i < 32; i++) {
//        data[i] = hash[i];
//      }
      break;
    }
    if( work > (difficulty >> 1) ) {
      BYTE * ptr = (BYTE*)&ctx.state;
      for (i = 0; i < 4; ++i) {
        data[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        data[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        data[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        data[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        data[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        data[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        data[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        data[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
      }
//      for (i = 0; i < 32; i++) {
//        data[i] = hash[i];
//      }
    }
  }
//  int offset = jdx * sizeRow + idx;
//  if (offset < 2048) {
//    info_debug[offset].flag = 7;
//    info_debug[offset].blockIdx = blockIdx.x;
//    info_debug[offset].blockIdy = blockIdx.y;
//    info_debug[offset].threadIdx = threadIdx.x;
//    info_debug[offset].threadIdy = threadIdx.y;
//  }
  if ((idx == 0) && (jdx == 0)) {
    *cycles = r;
  }
}

__global__ void kernel_sha256_val(BYTE *data, WORD len, BYTE *hash, WORD cycle, volatile bool *stop) {
  while(*stop) {};

  SHA256_CTX ctx;
  d_sha256_init(&ctx);

  int idx;
  for (idx = 0; idx < cycle; ++idx)
    d_sha256_update(&ctx, data, len);

  d_sha256_final(&ctx,hash);
}

__device__ WORD hash2int(BYTE h[32]) {
  WORD our_diff = 0;

  for(int i = 0; i < 31; i++) {
    BYTE mask = 0x80;
    for(int j = 0; j < 8; j++) {
      if ( (h[i] & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
        our_diff += ((h[i] << j) + (h[i + 1] >> (8 - j)));
        return our_diff;
      }
    }
  }
  return our_diff;
}

__device__ WORD hash2int_w(WORD h[8]) {
  WORD our_diff = 0;

  for(int i = 0; i < 8; i++) {
    WORD mask = 0x80000000;
    for(int j = 0; j < 32; j++) {
      if ( (h[i] & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
        if ((24 - j) >= 0) {
          our_diff += (h[i] >> (24 -j));
        } else {
          our_diff += ((h[i] << (j - 24)) + (h[i + 1] >> (56 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

//Constants for SHA-256
__device__ static const WORD k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

//SHA-256 functions taken from Brad Conte's implementation
//https://github.com/B-Con/crypto-algorithms/blob/master/sha256.c
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[]) {
  WORD x, res0, res1;
  WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
//  BYTE * ptr = (BYTE*)&m,
//    * ptr1 = ptr + 1,
//    * ptr2 = ptr + 2,
//    * ptr3 = ptr + 3,
//    * data_ptr = (BYTE*)data;
//
//  for (i = 0, j = 0; i < 16; ++i, j += 4) {
//    *(ptr3 + j) = *(data_ptr++);
//    *(ptr2 + j) = *(data_ptr++);
//    *(ptr1 + j) = *(data_ptr++);
//    *(ptr + j) = *(data_ptr++);
//  }

  for (i = 0, j = 0; i < 16; ++i, j += 4)
    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  for ( ; i < 64; ++i) {
// sig0:
    x = m[i - 15];
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 7;\n\t"  // t1 = x >> 7
        " shf.r.clamp.b32    t2, %1, %1, 18;\n\t"  // t2 = x >> 18
        " xor.b32            t1, t1, t2;\n\t"
        " shr.u32            t2, %1, 3;\n\t" // t2 = x >> 3
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res0) : "r" (x));
//    res0 = (x >> 7) | (x << 25);
//    res0 ^= (x >>18) | (x << 14);
//    res0 ^= (x >> 3);
//sig1:
    x = m[i - 2];
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 17;\n\t"  // t1 = x >> 17
        " shf.r.clamp.b32    t2, %1, %1, 19;\n\t"  // t2 = x >> 19
        " xor.b32            t1, t1, t2;\n\t"
        " shr.u32            t2, %1, 10;\n\t" // t2 = x >> 10
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res1) : "r" (x));
//    res1 = (x >> 17) | (x << 15);
//    res1 ^= (x >> 19) | (x << 13);
//    res1 ^= (x >> 10);

    m[i] = res1 + m[i - 7] + res0 + m[i - 16];
  }

  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];

  for (i = 0; i < 64; ++i) {
// ep0:
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 2;\n\t"  // t1 = x >> 2
        " shf.r.clamp.b32    t2, %1, %1, 13;\n\t"  // t2 = x >> 13
        " xor.b32            t1, t1, t2;\n\t"
        " shf.r.clamp.b32    t2, %1, %1, 22;\n\t" // t2 = x >> 22
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res0) : "r" (a));
//    x = a;
//    res0 = (x >> 2) | (x << 30);
//    res0 ^= (x >> 13) | (x << 19);
//    res0 ^= (x >> 22) | (x << 10);
// ep1:
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 6;\n\t"  // t1 = x >> 6
        " shf.r.clamp.b32    t2, %1, %1, 11;\n\t"  // t2 = x >> 11
        " xor.b32            t1, t1, t2;\n\t"
        " shf.r.clamp.b32    t2, %1, %1, 25;\n\t" // t2 = x >> 25
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res1) : "r" (e));
//    x = e;
//    res1 = (x >> 6) | (x << 26);
//    res1 ^= (x >> 11) | (x << 21);
//    res1 ^= (x >> 25) | (x << 7);

    t1 = h + res1 + ((e & f) ^ (~e & g)) + k[i] + m[i];
    t2 = res0 + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

__device__ void d_sha256_init(SHA256_CTX *ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len) {
  WORD i;
  WORD dl = ctx->datalen;

  for (i = 0; i < len; ++i) {
    ctx->data[dl] = data[i];
    dl++;
    if (dl == 64) {
      d_sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      dl = 0;
    }
  }
  ctx->datalen = dl;
}

__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[]) {
  WORD i = ctx->datalen;

  // Pad whatever data is left in the buffer.
  ctx->data[i++] = 0x80;
  if (i < 56) {
    while (i < 56)
      ctx->data[i++] = 0x00;
  } else {
    while (i < 64)
      ctx->data[i++] = 0x00;
    d_sha256_transform(ctx, ctx->data);
    memset(ctx->data, 0, 56);
  }

  // Append to the padding the total message's length in bits and transform.
  ctx->bitlen += ctx->datalen * 8;
  unsigned long long int bl = ctx->bitlen;
  ctx->data[63] = bl;
  ctx->data[62] = bl >> 8;
  ctx->data[61] = bl >> 16;
  ctx->data[60] = bl >> 24;
  ctx->data[59] = bl >> 32;
  ctx->data[58] = bl >> 40;
  ctx->data[57] = bl >> 48;
  ctx->data[56] = bl >> 56;
  d_sha256_transform(ctx, ctx->data);

  // Since this implementation uses little endian byte ordering and SHA uses big endian,
  // reverse all the bytes when copying the final state to the output hash.
//  BYTE * ptr = (BYTE*)&ctx->state;
//  for (i = 0; i < 4; ++i) {
//    hash[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
//  }
}
