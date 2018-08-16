/*********************************************************************
 * Filename:   sha256.c
 * Author:     Brad Conte (brad AT bradconte.com)
 * Copyright:
 * Disclaimer: This code is presented "as is" without any guarantees.
 * Details:    Performs known-answer tests on the corresponding SHA1
          implementation. These tests do not encompass the full
          range of available test vectors, however, if the tests
          pass it is very, very likely that the code is correct
          and was compiled properly. This code also serves as
          example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "sha256.h"

void amoveo_hash_gpu(BYTE *, WORD, BYTE *, WORD);
FILE *fdebug;

/*********************** FUNCTION DEFINITIONS ***********************/
int sha256_test()
{
  BYTE text1[] = {"abc"};
  BYTE text2[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
  BYTE text3[] = {"aaaaaaaaaa"};
  BYTE text4[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnop"};
  BYTE hash1[SHA256_BLOCK_SIZE] = {0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
                                   0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
  BYTE hash2[SHA256_BLOCK_SIZE] = {0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,
                                   0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1};
  BYTE hash3[SHA256_BLOCK_SIZE] = {0xcd,0xc7,0x6e,0x5c,0x99,0x14,0xfb,0x92,0x81,0xa1,0xc7,0xe2,0x84,0xd7,0x3e,0x67,
                                   0xf1,0x80,0x9a,0x48,0xa4,0x97,0x20,0x0e,0x04,0x6d,0x39,0xcc,0xc7,0x11,0x2c,0xd0};
  BYTE hash4[SHA256_BLOCK_SIZE] = {0xaa,0x35,0x3e,0x00,0x9e,0xdb,0xae,0xbf,0xc6,0xe4,0x94,0xc8,0xd8,0x47,0x69,0x68,
                                   0x96,0xcb,0x8b,0x39,0x8e,0x01,0x73,0xa4,0xb5,0xc1,0xb6,0x36,0x29,0x2d,0x87,0xc7};
  BYTE buf[SHA256_BLOCK_SIZE];
  SHA256_CTX ctx;
  int idx;
  int pass = 1;

//  amoveo_hash_gpu(text1, strlen((const char *)text1), buf, 1);
//  pass = pass && !memcmp(hash1, buf, SHA256_BLOCK_SIZE);

//  amoveo_hash_gpu(text2, strlen((const char *)text2), buf, 1);
//  pass = pass && !memcmp(hash2, buf, SHA256_BLOCK_SIZE);

//  amoveo_hash_gpu(text3, strlen((const char *)text3), buf, 100000);
//  pass = pass && !memcmp(hash3, buf, SHA256_BLOCK_SIZE);

  amoveo_hash_gpu(text4, strlen((const char *)text4), buf, 1);
  pass = pass && !memcmp(hash4, buf, SHA256_BLOCK_SIZE);

  return(pass);
}

int main() {
  char debugfilename[16];
  sprintf(debugfilename,"debug%s.txt","2"); //id);
  fdebug = fopen(debugfilename,"w");
  printf("SHA-256 tests: %s\n", sha256_test() ? "SUCCEEDED" : "FAILED");

  return(0);
}
