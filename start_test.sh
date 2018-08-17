export CUDA_VISIBLE_DEVICES=0

nvcc -Xptxas -O3 -v -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu_host.o src_c/sha256_gpu_host.cu
nvcc -Xptxas -O3 -v -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu_device.o src_c/sha256_gpu_device.cu
gcc -O3 -std=c99 -v -c -o sha256_gpu_test.o test_c/sha256_gpu_test.c

nvcc -O3 -v -arch=compute_61 -code=sm_61,compute_61 -o sha256_test sha256_gpu_test.o sha256_gpu_host.o sha256_gpu_device.o

# ./sha256_test