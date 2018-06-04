#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"

//maybe use <stdint.h>

WORD hash2integer(BYTE h[32]);
static WORD pair2sci(WORD l[2]);
static BYTE* next_nonce(BYTE nonce[32]);
int check_pow(BYTE nonce[32], int, int, BYTE data[32]);
BYTE* mine(BYTE nonce[32], int, int, BYTE data[32]);
void test_hash();
void test_hash2integer();
void test_check_pow();
void mine_test(BYTE nonce[32], int difficulty, BYTE data[32]);

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

BYTE* mine(BYTE nonce[32], int difficulty, int share_difficulty, BYTE data[32]) {
  while (1) {
    if (check_pow(nonce, difficulty, share_difficulty, data))
      return nonce;
    nonce = next_nonce(nonce);
  }
}

void write_nonce(BYTE x[32]) {
  FILE *f = fopen("nonce.txt", "w");
  if (f == NULL) {
      printf("Error opening file!\n");
      //exit(1);
    }
  rewind(f);//unnecessary line?
  fwrite(x, 1, 32, f);
  fclose(f);
  return;
}

int read_input(BYTE B[32], BYTE N[32], WORD id) {
  FILE *fileptr;
  fileptr = fopen("mining_input", "rb");
  fseek(fileptr, 0, SEEK_END);  // Jump to the end of the file
  int filelen = ftell(fileptr); // Get the current byte offset in the file
  //ftell returns a long, maybe we shouldn't truncate it.
  rewind(fileptr);
  fread(B, 32, 1, fileptr);
  fread(N, 32, 1, fileptr);
  N[28] = id % 256;
  N[29] = (id / 256) % 256;
  N[30] = ((id / 256) / 256) % 256;
  N[31] = (((id / 256) / 256) / 256) % 256;
  BYTE buffer[10] = { 0 };
  fread(buffer, filelen-64, 1, fileptr);
  int diff = 0;
  BYTE c = 1;
  for (int i = 0; i < 10; i++) {
    c = buffer[i];
    if (c == 0) {
      break;
    }
    diff *= 10;
    diff += (c - '0');
  }
  fclose(fileptr); // Close the file
  return diff;
}

BYTE * start_mine(BYTE bhash[32], BYTE nonce[32], WORD block_diff, WORD share_diff, WORD id)
{
  if (block_diff < 10) {
    fprintf(stderr, "speed test starting\n");
    clock_t begin = clock();
    mine_test(nonce, block_diff, bhash);/* here, do your time-consuming job */
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    double speed = 1 / time_spent;
    fprintf(stderr, "speed result: %f megahashes per second per CPU\n", speed);
    return(0);
  }
//  fprintf(stderr, "difficulty is %i\n\r", block_diff);
  mine(nonce, block_diff, share_diff, bhash); //nonce, difficulty, data
//  write_nonce(nonce);
  //test_check_pow();
  //test_hash();
  //test_hash2integer();
  return(nonce);
}

void mine_test(BYTE nonce[32], int difficulty, BYTE data[32]) {
  for (int i = 0; i < 1000000; i++) {
    check_pow(nonce, 10000, 10000, data);
    nonce = next_nonce(nonce);
  }
}

void test_check_pow() {
  //["pow","2w1EW/I07ZnVg8hK4TzPEUA2XXyh2MpdyLgntn/42dI=",6452,995842502870377667814772]
  static BYTE data[32] = {
    70,131,192,39,4,93,79,150,232,178,119,91,87,183,101,245,
    48,56,180,196,47,44,184,68,130,255,91,39,44,98,51,216
  };
  static BYTE nonce[32] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,210,224,186,
    241,224,230,139,188,229,116
  };
  int out = check_pow(nonce, 12000, 6452, data);
  printf("check pow gave %i\n", out);
  //should be a 1.
  return;
}

void test_hash() {
  BYTE bhash[66] = {0};
  SHA256_CTX ctx;
  for (int i = 0; i < 66; i++) {
    bhash[i] = 0;
  }
  sha256_init(&ctx);
  sha256_update(&ctx, bhash, 66);
  BYTE buf[32];
  sha256_final(&ctx, buf);
  printf("%i, %i, %i, %i, %i, %i, %i, %i, \n",
         buf[0], buf[1], buf[2], buf[3],
         buf[4], buf[5], buf[6], buf[7]);
  // should be 239, 187, 3, 183, 167, 246, 253, 60,
  return;
}

void test_hash2integer() {
  BYTE buf[32] = {
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    200,200,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
  };
  printf("%i\n", hash2integer(buf));
  // should be 32968
  return;
}
