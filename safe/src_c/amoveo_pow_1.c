#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sha256.h"

FILE *fdebug;

void amoveo_update_gpu(BYTE *nonce, BYTE *data);
void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle);
bool amoveo_mine_gpu(BYTE nonce[23],
                                unsigned int difficulty,
                                BYTE data[32],
                                unsigned int GDIM,
                                unsigned int BDIM);

void amoveo_update_gpu(BYTE *nonce, BYTE *data) /* WORD block_diff */ {
  fprintf(fdebug,"----Update: data and nonce \n");
  fflush(fdebug);
}

bool amoveo_mine_gpu(BYTE nonce[23],
                     unsigned int difficulty,
                     BYTE data[32],
                     unsigned int GDIM,
                     unsigned int BDIM) {
  bool success;
  int stepNumbers = 8;
  double numHashes, round_elapsed, total_elapsed;
  fprintf(fdebug,"Debug info: gdim= %d bdim= %d steps= %d\n",
      GDIM, BDIM, stepNumbers);
  fflush(fdebug);

  for (int step = 0; step < stepNumbers; step++) {
    sleep(2);
    success = (step > 6)?true:false; //amoveo_mine_gpu(nonce, block_diff, bhash, GDIM, BDIM, step, gpuRounds);
    fprintf(fdebug,"Round %d/%d. Work Difficulty : %d.\n",
      step,stepNumbers,difficulty);
    fflush(fdebug);

    if(success) {
      break;
    }
  }
  return success;
}

void amoveo_hash_gpu(BYTE *data, WORD len, BYTE *hash, WORD cycle) {
  fprintf(fdebug,"----Calculate Hash:\n");
  fflush(fdebug);
}

bool start_mine(BYTE bhash[32], BYTE nonce[23], WORD block_diff, WORD share_diff, WORD id) {
  clock_t t_start, t_round, t_end;
  bool success;
  int stepNumbers = 5;
  int GDIM = 512;
  int BDIM = 128;
  int gpuRounds = 40;
  double numHashes, round_elapsed, total_elapsed;
//  char debugfilename[16];

//  sprintf(debugfilename,"debug%d.txt",id);
//  fdebug = fopen(debugfilename,"w");

  fprintf(fdebug,"Debug info: gdim= %d bdim= %d gpuRounds= %d steps= %d\n",
      GDIM, BDIM, gpuRounds, stepNumbers);
  fflush(fdebug);

//  BYTE tstHash[32];
//  amoveo_hash_gpu(bhash, 32, tstHash, 1);
//  fprintf(fdebug,"Test hash : ");
//  for(int i = 0; i < 32; i++)
//      fprintf(fdebug,"%02X",tstHash[i]);
//  fprintf(fdebug,"\n");
//  fflush(fdebug);

  t_start = clock();
  for (int step = 0; step < stepNumbers; step++) {
    t_round = clock();
    sleep(2);
    success = (step > 3)?true:false; //amoveo_mine_gpu(nonce, block_diff, bhash, GDIM, BDIM, step, gpuRounds);
    numHashes = ((double)GDIM)*((double)GDIM)*((double)BDIM)*((double)gpuRounds);
    t_end = clock();
    round_elapsed = ((double)(t_end-t_round))/CLOCKS_PER_SEC;
    total_elapsed = ((double)(t_end-t_start))/CLOCKS_PER_SEC;
    fprintf(fdebug,"Round %d/%d Hash Rate : %0.2f MH/s took %0.1f s, %0.1f s total. Work Difficulty : %d. Share Difficulty : %d\n",
      step,stepNumbers,numHashes/(1000000.0*round_elapsed),round_elapsed,total_elapsed,block_diff,share_diff);
    fflush(fdebug);

    if(success) {
//      check difficulty for sure
      fprintf(fdebug,"Nonce found after %f seconds - Difficulty %d\n",total_elapsed,0);
//      fprintf(fdebug,"Block : ");
//      for(int i = 0; i < 32; i++)
//          fprintf(fdebug,"%02X",bdata[i]);
//      fprintf(fdebug,"\n");
      fprintf(fdebug,"Nonce : ");
      for(int i = 0; i < 23; i++)
          fprintf(fdebug,"%02X",nonce[i]);
      fprintf(fdebug,"\n");
      fflush(fdebug);
      break;
    }
  }
  return success;
}

