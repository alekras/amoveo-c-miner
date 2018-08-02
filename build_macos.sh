
set -x

#export CUDA_VISIBLE_DEVICES=0

#nvcc -Xptxas -O3 -v -lrt -lm -arch=compute_61 -code=sm_61,compute_61 -D_FORCE_INLINES -c -o sha256_gpu.o src_c/sha256_gpu.cu
gcc -O3 -std=c99 -c src_c/amoveo_pow.c
gcc -O3 -c src_c/port.cpp
#nvcc -O3 -v -arch=compute_61 -code=sm_61,compute_61 -o amoveo_c_miner sha256_gpu.o amoveo_pow.o port.o
gcc amoveo_pow.o port.o -o amoveo_c_miner -lstdc++
rm *.o

# next recompile the erlang.
erlc -DMACOS -o _build/default/lib/amoveo_miner/ebin src/miner.erl 
erlc -o _build/default/lib/amoveo_miner/ebin src/httpc_mock.erl 
# finally start an erlang interpreter so you can call the program.
erl -pa _build/default/lib/*/ebin \
 -eval "miner:start()"
# run like this `miner:start()`


