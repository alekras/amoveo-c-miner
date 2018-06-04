# gcc -o extprg complex.c erl_comm.c port.c
gcc -o extprg port.cpp -lstdc++
cp extprg ebin/

erl \
 -pa ebin \
 -s port_test start
 