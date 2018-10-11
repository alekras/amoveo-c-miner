export CUDA_VISIBLE_DEVICES=0,1

nvcc -Xptxas -O3 -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu_host.o src_c/sha256_gpu_host.cu
nvcc -Xptxas -O3 -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu_device.o src_c/sha256_gpu_device.cu
nvcc -Xptxas -O3 -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o amo_sha256_gpu_device.o src_c/amo_sha256_gpu_device.cu
gcc -O3 -std=c99 -c -o gpu_mine_test.o test_c/gpu_mine_test.c

nvcc -O3 -arch=compute_61 -code=sm_61,compute_61 -o mine_test gpu_mine_test.o sha256_gpu_host.o sha256_gpu_device.o amo_sha256_gpu_device.o
# ./mine_test