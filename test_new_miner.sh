
set -x

#export CUDA_VISIBLE_DEVICES=0,1
gcc -O3 -std=c99 -c test_c/kernel_simulator.c
gcc -O3 -c src_c/port.cpp
gcc kernel_simulator.o port.o -o amoveo_c_miner -lstdc++
rm *.o

# next recompile the erlang.
/opt/local/bin/rebar3 do version,compile
# erlc -o _build/default/lib/amoveo_miner/ebin src/miner.erl 
# finally start an erlang interpreter so you can call the program.
erl -pa _build/default/lib/*/ebin \
 -config miner \
 -eval "miner_serv:start()"

# run like this `miner:start()`


