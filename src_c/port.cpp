#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
//#include <netinet/in.h>
#include <sys/time.h>
#include <unistd.h>

#include "sha256.h"

extern "C" {
  char amoveo_update_gpu(BYTE *nonce, BYTE *data);
  void amoveo_stop_gpu();
  void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle);
  void amoveo_gpu_alloc_mem(int, int, int);
  void amoveo_gpu_free_mem();
  void amoveo_mine_gpu(BYTE nonce[23],
                       WORD difficulty,
                       BYTE data[32],
                       WORD GDIM,
                       WORD BDIM,
                       WORD id);
  void gpu_info(int device);
  FILE *fdebug;
}

int is_big_endian(void) {
  union {
    uint32_t i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 1;
}

uint32_t swapByteOrder(uint32_t ui) {
  ui = (ui >> 24) |
       ((ui<<8) & 0x00FF0000) |
       ((ui>>8) & 0x0000FF00) |
       (ui << 24);
  return ui;
}

uint32_t read_32b_integer(std::istream& s) {
  uint32_t value;
  s.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!is_big_endian()) {
    value = swapByteOrder(value);
  }
//    std::cerr << 'v' << value << "\n\r";
//    std::cerr.flush();
  return value;
}

std::ostream&  write_32b_integer(std::ostream& s, uint32_t len) {
//    std::cerr << 'w' << len << "\n\r";
//    std::cerr.flush();

  if (!is_big_endian()) {
    len = swapByteOrder(len);
  }
  s.write(reinterpret_cast<char*>(&len), sizeof(len));
  return s;
}

int main(int argc, char **argv) {
  char debugfilename[16];
  WORD GDIM = 36, BDIM = 96;
//  struct timeval t_start, t_end;
  sprintf(debugfilename,"debug%s.txt",argv[1]); //id);
  fdebug = fopen(debugfilename,"w");

  BYTE success, command;
  WORD len, block_diff, share_diff,
       id = (WORD) strtol(argv[1], (char **)0, 10);
  BYTE* bhash = new BYTE[32];
  BYTE* nonce = new BYTE[23];
// Helps to detect std::cin closing
  std::cin.exceptions ( std::istream::failbit | std::istream::badbit );

  fprintf(fdebug,"--Start Erlang Port-- ID = %d\n", id);
  fflush(fdebug);

  try {
    amoveo_gpu_alloc_mem(id, GDIM, BDIM);
    gpu_info(id);
// Read messages from Erlang port
    while(true) {
// read packet length, 4 bytes
      len = read_32b_integer(std::cin);
      std::cin.read(reinterpret_cast<char*>(&command), 1);
//      std::cerr << "PORT: command= " << command << ".\n\r";
//      std::cerr.flush();
//      fprintf(fdebug,"PORT: fprintf() %d\n", command);
//      fflush(fdebug);


      if (command == 'I') {
// read data, 32 bytes
        std::cin.read((char*)bhash, 32);
// read data, 23 bytes
        std::cin.read((char*)nonce, 23);
// read difficulty, 4 bytes
        block_diff = read_32b_integer(std::cin);
// read difficulty, 4 bytes
        share_diff = read_32b_integer(std::cin);
// read core_id, 4 bytes
        id = read_32b_integer(std::cin);
//        gettimeofday(&t_start, NULL);
//        block_diff = 8000;
        amoveo_mine_gpu(nonce, block_diff, bhash, GDIM, BDIM, id);
//        std::cerr << "PORT: after command= " << command << ".\n\r";
//        std::cerr.flush();
      } else if (command == 'U') {
        success = (char) amoveo_update_gpu(nonce, bhash);
//        std::cerr << "PORT: after command= " << command << "; success= " << (int)success << ".\n\r";
//        std::cerr.flush();
        if (success) {
          write_32b_integer(std::cout, 24 + 32);
          std::cout.write(reinterpret_cast<char*>(&success), 1);
          std::cout.write((char*)nonce, 23);
          std::cout.write((char*)bhash, 32);
          std::cout.flush();
        } else {
          write_32b_integer(std::cout, 1);
          std::cout.write(reinterpret_cast<char*>(&command), 1);
          std::cout.flush();
        }
      } else if (command == 'S') {
        amoveo_stop_gpu();
//        std::cerr << "PORT: after command= " << command << ".\n\r";
//        std::cerr.flush();
//        gettimeofday(&t_end, NULL);
//       double numHashes = ((double)GDIM)*((double)GDIM)*((double)BDIM)*((double)(*h_cycles));
//        double total_elapsed = (double)(t_end.tv_usec - t_start.tv_usec) / 1000000 + (double)(t_end.tv_sec - t_start.tv_sec);

//        fprintf(fdebug,"PORT:: took time = %0.1f secs\r\n", total_elapsed);

        write_32b_integer(std::cout, 1);
        std::cout.write(reinterpret_cast<char*>(&command), 1);
        std::cout.flush();
      }
    }
    amoveo_gpu_free_mem();
  } catch ( std::istream::failure e1 ) {
    std::cerr << "PORT: std::cin closed. Big endian= " << is_big_endian() << ".\n\r";
    std::cerr.flush();
    return 0;
  } catch ( std::exception e2 ) {
    return 0;
  }
  delete[] bhash;
  delete[] nonce;
  return 0;
}
