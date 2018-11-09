
set -x

export CUDA_VISIBLE_DEVICES=0,1
COMPILE_FLAGS=" -Xptxas -O3 -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c "
#nvcc -Xptxas -O3 -v -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu_host.o src_c/sha256_gpu_host.cu
nvcc ${COMPILE_FLAGS} -o sha256_gpu_host.o src_c/sha256_gpu_host.cu
nvcc ${COMPILE_FLAGS} -o sha256_gpu_device.o src_c/sha256_gpu_device.cu
nvcc ${COMPILE_FLAGS} -o amo_sha256_gpu_device.o src_c/amo_sha256_gpu_device_j.cu
# gcc -O3 -std=c99 -c src_c/amoveo_pow.c
gcc -O3 -c src_c/port.cpp
nvcc -O3 -arch=compute_61 -code=sm_61,compute_61 -o amoveo_c_miner sha256_gpu_host.o sha256_gpu_device.o amo_sha256_gpu_device.o port.o
# gcc sha256.o amoveo_pow.o port.o -o amoveo_c_miner -lstdc++
rm *.o

# next recompile the erlang.
/opt/local/bin/rebar3 do version,compile
# erlc -o _build/default/lib/amoveo_miner/ebin src/miner.erl 
# finally start an erlang interpreter so you can call the program.
erl -pa _build/default/lib/*/ebin \
 -config miner \
 -eval "miner:start()" \
 -detached

# run like this `miner:start()`


