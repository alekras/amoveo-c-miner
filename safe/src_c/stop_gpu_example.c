#include <stdio.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__
void mykernel(volatile int *data) {
  while (*data) {};
  printf("finished\n");
}

int main() {

  int *d_data, *h_data;
  cudaSetDeviceFlags(cudaDeviceMapHost);
  cudaCheckErrors("cudaSetDeviceFlags error");
  cudaHostAlloc((void **)&h_data, sizeof(int), cudaHostAllocMapped);
  cudaCheckErrors("cudaHostAlloc error");
  cudaHostGetDevicePointer(&d_data, h_data, 0);
  cudaCheckErrors("cudaHostGetDevicePointer error");
  *h_data = 1;
  printf("kernel starting\n");
  mykernel<<<1,1>>>(d_data);
  cudaCheckErrors("kernel fail");
  getchar();
  *h_data = 0;
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel fail 2");
  return 0;
}
