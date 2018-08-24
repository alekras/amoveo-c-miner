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
#include "sha256_gpu.h"

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
WORD hash_2_int(BYTE h[32]);
extern "C" void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle);

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(fdebug,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
    fflush(fdebug);
    if (abort) exit(code);
  }
}

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__, true); }
#define CUDA_UNSAFE_CALL(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__, false); }

static GPU_thread_info *h_info_debug, *d_info_debug;
static struct timeval t_start, t_end;
static long int *h_cycles_total, *d_cycles_total;

static BYTE text[55];//32+23
static unsigned int GDIM, BDIM, saved_difficulty;

//Allocate space on Global Memory
static BYTE *h_data, *d_data;
//static BYTE *h_hash, *d_hash;
static BYTE *h_nonce, *d_nonce;
static bool *h_success, *d_success;
static bool *h_stop, *d_stop;
static long int *h_cycles, *d_cycles;

extern "C" bool amoveo_update_gpu(BYTE *nonce, BYTE *data) {
//  fprintf(stderr,"GPU: >>> Amoveo update *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);
  bool success_ret = *h_success;
  if (success_ret) {
    fprintf(stderr,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);
    fprintf(fdebug,"GPU: !!!!!!!!!! Amoveo update h_success=%d\r\n", *h_success);

    *h_stop = true;

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    BYTE result[32];
    for (int i = 0; i < 23; i++) {
      text[i + 32] = h_nonce[i];
    }

    amoveo_hash_gpu(text, 55, result, 1);

    fprintf(stderr," Input     : ");
    for(int i = 0; i < 32; i++)
        fprintf(stderr,"%d,",text[i]);
    fprintf(stderr,"\r\n");
    fprintf(stderr," GPU Data  : ");
    for(int i = 0; i < 32; i++)
        fprintf(stderr,"%d,",h_data[i]);
    fprintf(stderr,"\r\n");
    fprintf(stderr," Check     : ");
    for(int i = 0; i < 32; i++)
        fprintf(stderr,"%d,",result[i]);
    fprintf(stderr,"\r\n");
    fprintf(stderr," Nonce     : ");
    for(int i = 0; i < 23; i++)
        fprintf(stderr,"%d,",h_nonce[i]);
    fprintf(stderr,"\r\n");
    int d = hash_2_int(result);
    fprintf(stderr,"Difficulty: %d.", hash_2_int(result));
    fprintf(stderr,"\r\n");
    fflush(stderr);

    memcpy(nonce, h_nonce, 23 * sizeof(BYTE));
    memcpy(data, h_data, 32 * sizeof(BYTE));
    if (d < saved_difficulty) {
      success_ret = false;
    }
    *h_success = false;
  }

//  fprintf(stderr,"GPU: <<< Amoveo update *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);
  return success_ret;
}

extern "C" void amoveo_stop_gpu() {
//  fprintf(stderr,"GPU: >>> Amoveo stop *h_stop=%d *h_success=%d\r\n", *h_stop, *h_success);
  *h_stop = true;

  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  gettimeofday(&t_end, NULL);
  double numHashes = ((double)GDIM)*((double)BDIM)*((double)(*h_cycles));
  double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);

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

//  fprintf(stderr,"GPU: <<< Amoveo stop h_stop=%d success=%d\r\n", *h_stop, *h_success);
}

extern "C" void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle) {
  BYTE *h_data, *d_data;
  BYTE *h_hash, *d_hash;
//  cudaDeviceProp prop;
//  CUDA_SAFE_CALL( cudaGetDeviceProperties (&prop, 0) );
//  fprintf(stderr," MultiProcessor Count = %d\n", prop.multiProcessorCount);
//  fprintf(stderr," Maximum number of threads per block = %d\n", prop.maxThreadsPerBlock);
//  fprintf(stderr," maxThreadsDim[3] contains the maximum size of each dimension of a block = [%d,%d,%d]\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
//  fprintf(stderr," maxGridSize[3] contains the maximum size of each dimension of a grid = [%d,%d,%d]\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
//  fprintf(stderr," canMapHostMemory is 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not = %d\n", prop.canMapHostMemory);
//
//  fprintf(stderr," asyncEngineCount is 1 when the device can concurrently copy memory between host and device while executing a kernel.\n");
//  fprintf(stderr,"   It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.\n");
//  fprintf(stderr,"   It is 0 if neither of these is supported = %d\n", prop.asyncEngineCount);
//  fprintf(stderr," unifiedAddressing is 1 if the device shares a unified address space with the host and 0 otherwise = %d\n", prop.unifiedAddressing);
//  fprintf(stderr," maxThreadsPerMultiProcessor is the number of maximum resident threads per multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);
//  fflush(stderr);

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_data, len * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_data, h_data, 0) );
//Copy data to device
  memcpy(h_data, data, len * sizeof(BYTE));

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_hash, 32 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_hash, h_hash, 0) );

  kernel_sha256_val<<<1, 1>>>(d_data, len, d_hash, cycle);

  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

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

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_cycles_total, (18 * 1024) * sizeof(long int), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_cycles_total, h_cycles_total, 0) );
//  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 2048 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 4 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_info_debug, h_info_debug, 0) );
  *h_stop = true;
  *h_success = false;
}

extern "C" void amoveo_gpu_free_mem() {
//Free memory on device
  CUDA_SAFE_CALL( cudaFreeHost(h_data) );
  CUDA_SAFE_CALL( cudaFreeHost(h_nonce) );
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
  saved_difficulty = difficulty;
//Initialize Cuda Grid variables
//  dim3 DimGrid(gdim, gdim);
//  dim3 DimBlock(bdim,1);
//  fprintf(stderr,"GPU: >>> Amoveo mine gpu(diff = %d)\r\n", difficulty);

//Copy data to device
  memcpy(h_data, data, 32 * sizeof(BYTE));
  memcpy(h_nonce, nonce, 23 * sizeof(BYTE));
  memcpy(text, data, 32 * sizeof(BYTE));
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
  kernel_sha256<<<gdim, bdim>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, device_id, d_cycles_total);
//  fprintf(stderr,"GPU: <<< Amoveo mine gpu\r\n");
}

WORD hash_2_int(BYTE h[32]) {
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

extern "C" void test1(int difficulty, int gdim, int bdim, BYTE data[32]) {
  int n, m;
  amoveo_gpu_alloc_mem();

  m = 0;
  while (m < 3) {
    *h_success = false;
    *h_stop = false;

    memcpy(h_data, data, 32 * sizeof(BYTE));
    memset(h_nonce, 0, 23);

//  int gdim = 9,
//      bdim = 1024;
    gettimeofday(&t_start, NULL);
    h_nonce[0] = (BYTE)t_start.tv_usec;
    h_nonce[22] = (BYTE)(t_start.tv_usec >> 8);
    kernel_sha256<<<gdim, bdim>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, 0, d_cycles_total);

    m++;
    n = 0;
    while(n < 12) {
      sleep(5);
      fprintf(stderr,"  m=%d:n=%d, success=%d, stop=%d, cycles=%d.\r\n", m, n, *h_success, *h_stop, *h_cycles);
      if (*h_success) {
        break;
      }
      n++;
    }
    *h_stop = true;
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    fprintf(stderr,"* m=%d:n=%d, success=%d, stop=%d, cycles=%d.\r\n", m, n, *h_success, *h_stop, *h_cycles);
    gettimeofday(&t_end, NULL);
    double numHashes = ((double)gdim)*((double)bdim)*((double)(*h_cycles));
    double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);

    fprintf(stderr,"Cycles = %d  Hash rate = %0.2f MH/s  took time = %0.1f secs\r\n", *h_cycles, numHashes/(1000000.0*total_elapsed), total_elapsed);
    fprintf(stderr," Nonce   : ");
    for(int i = 0; i < 23; i++)
      fprintf(stderr,"%02X.",h_nonce[i]);
    fprintf(stderr,"\n");
    fprintf(stderr," Data    : ");
    for(int i = 0; i < 32; i++)
      fprintf(stderr,"%02X.",h_data[i]);
    fprintf(stderr,"\n");
    //  for (int i = 0; i < 2048; i++) {
    //    fprintf(stderr, "[fl:%d, blk.idx:[%d, %d], thr.idx:[%d, %d]]\n", h_info_debug[i].flag, h_info_debug[i].blockIdx, h_info_debug[i].blockIdy, h_info_debug[i].threadIdx, h_info_debug[i].threadIdy);
    //  }

    BYTE text[55];//32+23
    BYTE result[32];
    for (int i = 0; i < 32; i++) {
      text[i] = data[i];
    }
    for (int i = 0; i < 23; i++) {
      text[i + 32] = h_nonce[i];
    }

//    amoveo_gpu_free_mem();
//    CUDA_SAFE_CALL( cudaDeviceReset() );
//    CUDA_SAFE_CALL( cudaSetDevice(0) );
//    amoveo_gpu_alloc_mem();
    amoveo_hash_gpu(text, 55, result, 1);

    fprintf(stderr," Check   : ");
    for(int i = 0; i < 32; i++)
      fprintf(stderr,"%02X.",result[i]);
    fprintf(stderr,"\n");
    fprintf(stderr," Difficulty: %d.", hash_2_int(result));
    fprintf(stderr,"\n");
    fflush(stderr);
  }
  amoveo_gpu_free_mem();
}

extern "C" void test2(int gdim, int bdim) {
  amoveo_gpu_alloc_mem();

  *h_stop = false;

  kernel_test<<<gdim, bdim>>>(d_stop, d_cycles, d_info_debug);

  int n = 0;
  while(n < 10) {
    sleep(1);
    fprintf(stderr, "  n=%d, cycles=%d.\r\n", n, *h_cycles);
    n++;
  }
  *h_stop = true;
  cudaDeviceSynchronize();

  fprintf(stderr, "  n=%d, cycles=%d.\r\n", n, *h_cycles);
  n = 0;
  for (int i = 0; i < 100; i++) {
	for (int j = 0; j < 10; j++) {
	    fprintf(stderr, "[%d,%d]:%d ", h_info_debug[n].blockIdx, h_info_debug[n].threadIdx, h_info_debug[n].flag);
	    n++;
	}
    fprintf(stderr, "\r\n");
  }

  amoveo_gpu_free_mem();
}
