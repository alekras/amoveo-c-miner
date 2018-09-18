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
static bool /*stop,*/ *h_stop, *d_stop;
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

  long long int total = 0;
  long int min_cycles = 10000000, max_cycles = 0, zero_cycles = 0;
  for(int i = 0; i < (GDIM * BDIM); i++) {
    long int temp = h_cycles_total[i];
    if (temp == 0) {
      zero_cycles++;
//        fprintf(stderr,"temp = 0 i = %d\n", i);
    }
    total = total + temp;
    if (temp < min_cycles) {
      min_cycles = temp;
    }
    if (temp > max_cycles) {
      max_cycles = temp;
    }
  }
  double numHashes = (double)(total);
//  double numHashes = ((double)GDIM)*((double)BDIM)*((double)(*h_cycles));
  double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);

  fprintf(fdebug,"Cycles = %ld, (max=%ld, min=%ld, zero=%ld) Hash rate = %0.2f MH/s  took time = %0.1f secs\r\n",
      total, max_cycles, min_cycles, zero_cycles, numHashes/(1000000.0 * total_elapsed), total_elapsed);
//  fprintf(fdebug,"Cycles = %d  Hash rate = %0.2f MH/s  took time = %0.1f secs\r\n", *h_cycles, numHashes/(1000000.0*total_elapsed), total_elapsed);
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

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_data, len * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_data, h_data, 0) );
//Copy data to device
  memcpy(h_data, data, len * sizeof(BYTE));

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_hash, 32 * sizeof(BYTE), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_hash, h_hash, 0) );

  kernel_sha256_val<<<1,1>>>(d_data, len, d_hash, cycle);

  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

//Copy result back to host
  memcpy(hash, h_hash, 32 * sizeof(BYTE));

//Free memory on device
  CUDA_SAFE_CALL(cudaFreeHost(h_data));
  CUDA_SAFE_CALL(cudaFreeHost(d_hash));
}

extern "C" void gpu_info(int device) {
  cudaDeviceProp prop;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop, device) );
  fprintf(fdebug," -- name = %s\n", prop.name);
  fprintf(fdebug," -- totalGlobalMem = %d is the total amount of global memory available on the device in bytes.\n", prop.totalGlobalMem);
  fprintf(fdebug," -- totalConstMem = %d is the total amount of constant memory available on the device in bytes.\n", prop.totalConstMem);

  fprintf(fdebug," -- multiProcessor Count = %d\n", prop.multiProcessorCount);
  fprintf(fdebug," -- maxThreadsPerMultiProcessor is the number of maximum resident threads per multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);
  fprintf(fdebug," -- warpSize = %d is the warp size in threads.\n", prop.warpSize);
  fprintf(fdebug," -- sharedMemPerMultiprocessor = %d is the maximum amount of shared memory available to a multiprocessor in bytes;\n", prop.sharedMemPerMultiprocessor);
  fprintf(fdebug,"       this amount is shared by all thread blocks simultaneously resident on a multiprocessor.\n");
  fprintf(fdebug," -- regsPerMultiprocessor = %d is the maximum number of 32-bit registers available to a multiprocessor;\n", prop.regsPerMultiprocessor);
  fprintf(fdebug,"    this number is shared by all thread blocks simultaneously resident on a multiprocessor\n");

  fprintf(fdebug," -- clockRate = %d is the clock frequency in kilohertz.\n", prop.clockRate);
  fprintf(fdebug," -- memoryClockRate = %d is the peak memory clock frequency in kilohertz.\n", prop.memoryClockRate);
  fprintf(fdebug," -- major = %d, minor = %d are the major and minor revision numbers defining the device's compute capability.\n", prop.major, prop.minor);
  fprintf(fdebug," -- computeMode = %d is the compute mode that the device is currently in.\n", prop.computeMode);

  fprintf(fdebug," -- maxThreadsPerBlock = %d is maximum number of threads per block.\n", prop.maxThreadsPerBlock);
  fprintf(fdebug," -- sharedMemPerBlock = %d is the maximum amount of shared memory available to a thread block in bytes.\n", prop.sharedMemPerBlock);
  fprintf(fdebug," -- regsPerBlock = %d is the maximum number of 32-bit registers available to a thread block.\n", prop.regsPerBlock);
  fprintf(fdebug," -- maxThreadsDim[3] = [%d,%d,%d] contains the maximum size of each dimension of a block.\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  fprintf(fdebug," -- maxGridSize[3] = [%d,%d,%d] contains the maximum size of each dimension of a grid.\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  fprintf(fdebug," -- canMapHostMemory = %d is 1 if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not.\n", prop.canMapHostMemory);
  fprintf(fdebug," -- asyncEngineCount = %d is 1 when the device can concurrently copy memory between host and device while executing a kernel.\n", prop.asyncEngineCount);
  fprintf(fdebug,"       It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.\n");
  fprintf(fdebug,"       It is 0 if neither of these is supported.\n");
  fprintf(fdebug," -- unifiedAddressing = %d is 1 if the device shares a unified address space with the host and 0 otherwise.\n", prop.unifiedAddressing);
  fflush(fdebug);
}

extern "C" void amoveo_gpu_alloc_mem(int device, int gdim, int bdim) {
  CUDA_SAFE_CALL( cudaSetDevice(device) );
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

  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_cycles_total, (gdim * bdim) * sizeof(long int), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostGetDevicePointer(&d_cycles_total, h_cycles_total, 0) );
//  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 2048 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
  CUDA_SAFE_CALL( cudaHostAlloc((void **)&h_info_debug, 1 * sizeof(GPU_thread_info), cudaHostAllocMapped) );
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
  CUDA_SAFE_CALL( cudaFreeHost(h_cycles_total) );
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

//Copy data to device
  memcpy(h_data, data, 32 * sizeof(BYTE));
  memcpy(h_nonce, nonce, 23 * sizeof(BYTE));
  memcpy(text, data, 32 * sizeof(BYTE));

  *h_success = false;
  *h_stop = false;

  gettimeofday(&t_start, NULL);
//  kernel_sha256<<<gdim, bdim, (bdim * 64 * sizeof(WORD))>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, device_id, d_cycles_total);
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
        if (j == 7) {
          our_diff += h[i + 1];
        } else {
          j++;
//          our_diff += (((h[i] << j)  & 0xFF) + (h[i + 1] >> (8 - j)));
          our_diff += (((h[i] << j)  & 0xFF) + (((h[i + 1] >> 1) & 0x7F) >> (7 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

extern "C" void test1(int device, int difficulty, int gdim, int bdim, BYTE data[32]) {
  int n, m;
  amoveo_gpu_alloc_mem(device, gdim, bdim);

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
    kernel_sha256<<<gdim, bdim, (bdim * 64 * sizeof(WORD))>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, 0, d_cycles_total);
//    kernel_sha256<<<gdim, bdim>>>(d_data, difficulty, d_nonce, d_success, d_stop, d_cycles, 0, d_cycles_total);

    m++;
    n = 0;
    while(n < 6) {
      sleep(5);
      fprintf(stderr,"  m=%d:n=%d, success=%d.\r\n", m, n, *h_success);
      if (*h_success) {
        break;
      }
      n++;
    }
    *h_stop = true;
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    fprintf(stderr,"* m=%d:n=%d, success=%d, stop=%d, cycles=%d.\r\n", m, n-1, *h_success, *h_stop, *h_cycles);
    gettimeofday(&t_end, NULL);

    long long int total = 0;
    long int min_cycles = 10000000, max_cycles = 0, zero_cycles = 0;
    for(int i = 0; i < (gdim * bdim); i++) {
      long int temp = h_cycles_total[i];
      if (temp == 0) {
        zero_cycles++;
//        fprintf(stderr,"temp = 0 i = %d\n", i);
      }
      total = total + temp;
      if (temp < min_cycles) {
        min_cycles = temp;
      }
      if (temp > max_cycles) {
        max_cycles = temp;
    	}
    }
    double numHashes = (double)(total);
    double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);

    fprintf(stderr,"Cycles = %ld, (max=%ld, min=%ld, zero=%ld) Hash rate = %0.2f MH/s  took time = %0.1f secs\r\n", total, max_cycles, min_cycles, zero_cycles, numHashes/(1000000.0*total_elapsed), total_elapsed);
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

extern "C" void test2(int device, int gdim, int bdim) {
  amoveo_gpu_alloc_mem(device, gdim, bdim);

  *h_stop = false;

//  kernel_test<<<gdim, bdim>>>(d_stop, d_cycles, d_info_debug);

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
