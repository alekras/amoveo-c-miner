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

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_data, len * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_data, h_data, 0) );
//Copy data to device
  memcpy(h_data, data, len * sizeof(BYTE));

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_hash, 32 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_hash, h_hash, 0) );

  kernel_sha256_val<<<1, 1>>>(d_data, len, d_hash, cycle, d_stop);

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
