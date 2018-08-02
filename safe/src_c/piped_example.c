//$ cat t832.cu
#include <stdio.h>
#include <stdbool.h>

#define IMGSZ 8000000
// for this example, NUM_FRAMES must be less than 255
#define NUM_FRAMES 128
#define nTPB 256
#define nBLK 64


unsigned char cur_frame = 0;
unsigned char validated_frame = 0;


bool validate_image(unsigned char *img) {
  validated_frame++;
  for (int i = 0; i < IMGSZ; i++) if (img[i] != validated_frame) {printf("image validation failed at %d, was: %d, should be: %d\n",i, img[i], validated_frame); return false;}
  return true;
}

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data) {
    validate_image((unsigned char *)data);
}


bool capture_image(unsigned char *img){

  for (int i = 0; i < IMGSZ; i++) img[i] = cur_frame;
  if (++cur_frame == NUM_FRAMES) {cur_frame--; return true;}
  return false;
}

__global__ void img_proc_kernel(unsigned char *img){

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  while(idx < IMGSZ){
    img[idx]++;
    idx += gridDim.x*blockDim.x;}
}

int main(){

  // setup

  bool done = false;
  unsigned char *h_imgA, *h_imgB, *d_imgA, *d_imgB;
  size_t dsize = IMGSZ*sizeof(unsigned char);
  cudaHostAlloc(&h_imgA, dsize, cudaHostAllocDefault);
  cudaHostAlloc(&h_imgB, dsize, cudaHostAllocDefault);
  cudaMalloc(&d_imgA, dsize);
  cudaMalloc(&d_imgB, dsize);

  cudaStream_t st1, st2;
  cudaStreamCreate(&st1);
  cudaStreamCreate(&st2);
  unsigned char *cur = h_imgA;
  unsigned char *d_cur = d_imgA;
  unsigned char *nxt = h_imgB;
  unsigned char *d_nxt = d_imgB;
  cudaStream_t *curst = &st1;
  cudaStream_t *nxtst = &st2;


  done = capture_image(cur); // grabs a frame and puts it in cur
  // enter main loop
  while (!done){
    cudaMemcpyAsync(d_cur, cur, dsize, cudaMemcpyHostToDevice, *curst); // send frame to device
    img_proc_kernel<<<nBLK, nTPB, 0, *curst>>>(d_cur); // process frame
    cudaMemcpyAsync(cur, d_cur, dsize, cudaMemcpyDeviceToHost, *curst);
  // insert a cuda stream callback here to copy the cur frame to output
    cudaStreamAddCallback(*curst, &my_callback, (void *)cur, 0);
    cudaStreamSynchronize(*nxtst); // prevent overrun
    done = capture_image(nxt); // capture nxt image while GPU is processing cur
    unsigned char *tmp = cur;
    cur = nxt;
    nxt = tmp;   // ping - pong
    tmp = d_cur;
    d_cur = d_nxt;
    d_nxt = tmp;
    cudaStream_t *st_tmp = curst;
    curst = nxtst;
    nxtst = st_tmp;
    }
}
//$ nvcc -o t832 t832.cu
//$ cuda-memcheck ./t832
//========= CUDA-MEMCHECK
//========= ERROR SUMMARY: 0 errors
//$
