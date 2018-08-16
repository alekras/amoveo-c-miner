/*********************************************************************
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include "sha256.h"

FILE *fdebug;

/*********************** FUNCTION DEFINITIONS ***********************/
void test1(int difficulty, int gdim, int bdim, BYTE data[32]);
void test2(int gdim, int bdim);

void reverse_order(BYTE * h, int len) {
  BYTE a, b;
  for (int i = 0; i < len; i += 4) {
    a = h[i];
    b = h[i + 1];
    h[i] = h[i + 3];
    h[i + 1] = h[i + 2];
    h[i + 2] = b;
    h[i + 3] = a;
  }
}

void bytes_2_words(WORD * w, BYTE * h, int len) {
  for (int i = 0, j = 0; i < len; i++, j +=4) {
    w[i] = (h[j] << 24) | (h[j + 1] << 16) | (h[j + 2] << 8) | (h[j + 3]);
  }
}

BYTE hash1[SHA256_BLOCK_SIZE] = {0xc1,0x3e,0x06,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
                                   0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
BYTE hash2[SHA256_BLOCK_SIZE] = {0x01,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,
                                   0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1};
BYTE hash3[SHA256_BLOCK_SIZE] = {0x00,0x00,0x00,0x00,0x00,0x1f,0xf0,0x92,0x81,0xa1,0xc7,0xe2,0x84,0xd7,0x3e,0x67,
                                   0xf1,0x80,0x9a,0x48,0xa4,0x97,0x20,0x0e,0x04,0x6d,0x39,0xcc,0xc7,0x11,0x2c,0xd0};
WORD words[8];
WORD dfc;

int main(int argc, char **argv) {
  char debugfilename[16];
  sprintf(debugfilename,"debug_test.txt");
  fdebug = fopen(debugfilename,"w");

  int diff, gdim, bdim;
  if (argc == 1) {
	printf("Usage: %s <difficulty> <gdim> <bdim>\n", argv[0]);
	return 0;
  }
  if (argc > 1) {
	diff = (int) strtol(argv[1], (char **)0, 10);
  }
  if (argc > 2) {
	gdim = (int) strtol(argv[2], (char **)0, 10);
  }
  if (argc > 3) {
	bdim = (int) strtol(argv[3], (char **)0, 10);
  }

//  test1(diff, gdim, bdim, hash1);
  test2(gdim, bdim);

  return 0;
}
