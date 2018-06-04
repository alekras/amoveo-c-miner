#For development purposes, it is convenient to recompile your code every time you run it. This way if anything changed, the changes will be included.
# Since the things being compiled are so small, they can be compiled instantly, and there is no cost to recompiling every time we run the software.


gcc -O3 -std=c99 -c src_c/sha256.c src_c/amoveo_pow.c
gcc -O3 -c src_c/port.cpp
gcc sha256.o amoveo_pow.o port.o -o amoveo_c_miner -lstdc++
rm *.o

# next recompile the erlang.
erlc -o _build/default/lib/amoveo_miner/ebin src/miner.erl 
# finally start an erlang interpreter so you can call the program.
erl -pa _build/default/lib/*/ebin \
 -eval "miner:start()"
# run like this `miner:start()`


