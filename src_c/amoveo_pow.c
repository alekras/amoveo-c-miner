#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"

//maybe use <stdint.h>

WORD hash2integer(BYTE h[32]);
//static WORD pair2sci(WORD l[2]);
static BYTE* next_nonce(BYTE nonce[32]);
int check_pow(BYTE nonce[32], int, int, BYTE data[32]);
//BYTE* mine(BYTE nonce[32], int, int, BYTE data[32]);
//void test_hash();
//void test_hash2integer();
//void test_check_pow();
//void mine_test(BYTE nonce[32], int difficulty, BYTE data[32]);

WORD hash2integer(BYTE h[32]) {
  WORD x = 0;
  WORD y;
  BYTE hi;
  for (int i = 0; i < 31; i++) {
    hi = h[i];
    if (hi == 0) {
      x += 8;
      y = h[i+1];
      continue;
    } else if (hi < 2) {
      x += 7;
      y = (hi * 128) + (h[i+1] / 2);
    } else if (hi < 4) {
      x += 6;
      y = (hi * 64) + (h[i+1] / 4);
    } else if (hi < 8) {
      x += 5;
      y = (hi * 32) + (h[i+1] / 8);
    } else if (hi < 16) {
      x += 4;
      y = (hi * 16) + (h[i+1] / 16);
    } else if (hi < 32) {
      x += 3;
      y = (hi * 8) + (h[i+1] / 32);
    } else if (hi < 64) {
      x += 2;
      y = (hi * 4) + (h[i+1] / 64);
    } else if (hi < 128) {
      x += 1;
      y = (hi * 2) + (h[i+1] / 128);
    } else {
      y = hi;
    }
    break;
  }
  return ((256 * x) + y);
}

int check_pow(BYTE nonce[32], int difficulty, int share_difficulty, BYTE data[32]) {
  BYTE text[66];//32+2+32
  for (int i = 0; i < 32; i++) {
    text[i] = data[i];
  }
  text[32] = difficulty / 256;
  text[33] = difficulty % 256;
  for (int i = 0; i < 32; i++) {
    text[i+34] = nonce[i];
  }
  SHA256_CTX ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, text, 66);
  BYTE buf[32];
  sha256_final(&ctx, buf);
  int i = hash2integer(buf);
  //printf("pow did this much work %d \n", i);
  return(i > share_difficulty);
}

static BYTE* next_nonce(BYTE nonce[32]){
  //we should use 32 bit or 64 bit integer
  for (int i = 0; i < 32; i++) {
    if (nonce[i] == 255) {
      nonce[i] = 0;
    } else {
      nonce[i] += 1;
      return nonce;
    }
  }
  return(0);
}

BYTE * start_mine(BYTE bhash[32], BYTE nonce[32], WORD block_diff, WORD share_diff, WORD id)
{
//  mine(nonce, block_diff, share_diff, bhash); //nonce, difficulty, data
  while (1) {
    if (check_pow(nonce, block_difficulty, share_difficulty, data))
      return nonce;
    nonce = next_nonce(nonce);
  }
  return(nonce);
}

