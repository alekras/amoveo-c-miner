#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src_c/sha256.h"

FILE *fdebug;

char amoveo_update_gpu(BYTE *nonce, BYTE *data);
void amoveo_stop_gpu();
void amoveo_gpu_alloc_mem(int, int, int);
void amoveo_gpu_free_mem();
void amoveo_mine_gpu(BYTE nonce[23],
                     WORD difficulty,
                     BYTE data[32],
                     WORD GDIM,
                     WORD BDIM,
                     WORD id);
void gpu_info(int device);

int count = 0;

char amoveo_update_gpu(BYTE *nonce, BYTE *data) {
  fprintf(fdebug,"---- Update: data and nonce \n");
  fflush(fdebug);
  count++;
  if ((count % 5) == 0) {
    for (int i = 0; i < 23; i++) {
      nonce[i] = i;
    }
    for (int i = 0; i < 32; i++) {
      data[i] = 31 - i;
    }
    return (char)1;
  } else {
    return (char)0;
  }
}

void amoveo_mine_gpu(BYTE nonce[23],
                     WORD difficulty,
                     BYTE data[32],
                     WORD GDIM,
                     WORD BDIM,
                     WORD id) {
  fprintf(fdebug,"---- amoveo_mine_gpu/6.\n");
  fflush(fdebug);
}

void amoveo_stop_gpu() {
  fprintf(fdebug,"---- amoveo_stop_gpu/0.\n");
  fflush(fdebug);

}

void amoveo_gpu_alloc_mem(int device, int gdim, int bdim) {
  fprintf(fdebug,"---- amoveo_gpu_alloc_mem/3.\n");
  fflush(fdebug);
}

void amoveo_gpu_free_mem() {
  fprintf(fdebug,"---- amoveo_gpu_free_mem/0.\n");
  fflush(fdebug);
}

void gpu_info(int device) {
  fprintf(fdebug,"---- gpu_info/1.\n");
  fflush(fdebug);
}

