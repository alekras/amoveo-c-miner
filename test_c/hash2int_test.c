/*********************************************************************
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "../src_c/sha256.h"

/*********************** FUNCTION DEFINITIONS ***********************/
//(256*number of leading 0 bits) + byte starting with 1.
WORD h2i(BYTE h[32]) {
  WORD our_diff = 0;
  BYTE hi, mask = 0x80;

  for(int i = 0; i < 31; i++) {
    mask = 0x80;
    hi = h[i];
    for(int j = 0; j < 8; j++) {
      if ( (hi & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
        if (j == 7) {
          our_diff += h[i + 1];
        } else {
          j++;
          our_diff += (((hi << j) & 0xFF) + (h[i + 1] >> (8 - j)));
//          printf(" %08X  %08X\n", h[i], (h[i] << j));
//          printf(" %08X  %08X\n", h[i+1], (((h[i + 1] >> 1) & 0x7F) >> (7 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

WORD h2i_rvs(BYTE h[32]) {
  WORD our_diff = 0;

  for (int w = 0; w < 32; w += 4) {
    for (int i = 3; i >= 0; i--) {
      BYTE mask = 0x80;
      for (int j = 0; j < 8; j++) {
        if ( (h[w + i] & mask) == 0 ) {
          our_diff++;
          mask = mask >> 1;
        } else {
          our_diff *= 256;
          our_diff += ((h[w + i] << j) + (h[w + i - 1] >> (8 - j)));
          return our_diff;
        }
      }
    }
  }
  return our_diff;
}

WORD h2i_w_old(WORD h[8]) {
  WORD our_diff = 0;

  for(int i = 0; i < 8; i++) {
    WORD mask = 0x80000000;
    for(int j = 0; j < 32; j++) {
      if ( (h[i] & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
//        printf(" 0) %d  ", our_diff );
        if (j == 31) {
          j = 0;
          i++;
        } else {
          j++;
        }
        if ((24 - j) >= 0) {
          our_diff += (h[i] >> (24 -j)) & 0xff;
//          printf(" 1) %08X  %08X\n",h[i], (h[i] >> (24 -j)) & 0xff );
        } else {
//          WORD t = ((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j);
          our_diff += ((h[i] << (j - 24)) + (((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j)));
//          printf(" 2) %08X  %08X  %08X\n",h[i], (h[i] << (j - 24)), (((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

WORD h2i_w(WORD h[8]) {
  WORD our_diff = 0;

  for(int i = 0; i < 8; i++) {
    WORD mask = 0xffff0000;
    int j = 16; // 2^4
    our_diff+= 16;
//    for(int k = 0; k < 5; k++) {
      if ( (h[i] & mask) == 0 ) {  // k = 0
        j+= 8;
        mask = mask >> 8;
      } else {
        j-=8;
        mask = mask << 8;
      }
      if ( (h[i] & mask) == 0 ) {  // k = 1
        j+= 4;
        mask = mask >> 4;
      } else {
        j-=4;
        mask = mask << 4;
      }
      if ( (h[i] & mask) == 0 ) {  // k = 2
        j+= 2;
        mask = mask >> 2;
      } else {
        j-=2;
        mask = mask << 2;
      }
      if ( (h[i] & mask) == 0 ) {  // k = 3
        j+= 1;
        mask = mask >> 1;
      } else {
        j-=1;
        mask = mask << 1;
      }
      if ( (h[i] & mask) == 0 ) {  // k = 4
//        our_diff+= 1;
//        mask = mask >> 1;
      } else {
//        our_diff-=1;
//        mask = mask << 1;
      }
//    }



      main()
      {
          int h = 0x70000000;
          int mask = 0xffffffff;
          int j = 32; // 2^4
          printf(" %08X  %08X %d\n", h, mask, j);
          if ( (h & mask) == 0 ) {  // k = 0
          printf("Stop: %08X  %08X %d\n", h, mask, j);
             return;
          } else {
             j-=16;
             mask = mask << 16;
          }
          printf(" %08X  %08X %d\n", h, mask, j);
          if ( (h & mask) == 0 ) {  // k = 1
             j+= 8;
             mask = mask >> 8;
          } else {
             j-=8;
             mask = mask << 8;
          }
          printf(" %08X  %08X %d\n", h, mask, j);
          if ( (h & mask) == 0 ) {  // k = 2
             j+= 4;
             mask = mask >> 4;
          } else {
             j-=4;
             mask = mask << 4;
          }
          printf(" %08X  %08X %d\n", h, mask, j);
          if ( (h & mask) == 0 ) {  // k = 3
             j+= 2;
             mask = mask >> 2;
          } else {
             j-=2;
             mask = mask << 2;
          }
          printf(" %08X  %08X %d\n", h, mask, j);
          if ( (h & mask) == 0 ) {  // k = 4
            j+= 1;
            mask = mask >> 1;
          } else {
            j-=1;
            mask = mask << 1;
            if ( (h & mask) == 0 ) {
              printf("Stop: %08X  %08X %d\n", h, mask, j);
              return;
            } else {
              j = 0;
            }
          }
          printf("Final: %08X  %08X %d\n", h, mask, j);

      }





      {
        our_diff *= 256;
//        printf(" 0) %d  ", our_diff );
        if (j == 31) {
          j = 0;
          i++;
        } else {
          j++;
        }
        if ((24 - j) >= 0) {
          our_diff += (h[i] >> (24 -j)) & 0xff;
//          printf(" 1) %08X  %08X\n",h[i], (h[i] >> (24 -j)) & 0xff );
        } else {
//          WORD t = ((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j);
          our_diff += ((h[i] << (j - 24)) + (((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j)));
//          printf(" 2) %08X  %08X  %08X\n",h[i], (h[i] << (j - 24)), (((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

#define step0(hi,hii) \
  for(int j = 31; j >= 0; j--) { \
    if ( (hi >> j) == 0 ) { \
      our_diff++; \
    } else { \
      our_diff *= 256; \
      if (j == 0) { \
        our_diff += hii & 0xff; \
      } else { \
        if (j >= 8) {our_diff += (hi >> (j-8)) & 0xff;} \
        else { our_diff += hi - (1 << j) + (hii >> (24 + j)) & 0xff; } \
      } \
      goto end; \
    }}

#define step(hi,hii) \
  for (int j = 0; j < 32; j++) { \
    if ( (hi & 0x80000000) == 0 ) { \
      our_diff++; \
      hi <<= 1; \
    } else { \
      our_diff *= 256; \
      if (j == 31) { \
        our_diff += hii >> 24; \
      } else { \
        our_diff += (hi >> 23) & 0xff; \
        if (j > 23) { our_diff += (hii >> (56 - j)); } \
      } \
      goto end; \
    }}

WORD h2i_w_new(WORD h[8]) {
  WORD our_diff = 0;
  step(h[0], h[1])
  step(h[1], h[2])
  step(h[2], h[3])
  step(h[3], h[4])
  step(h[4], h[5])
  step(h[5], h[6])
  step(h[6], h[7])
  end: return our_diff;
}

WORD hash2int(BYTE h[32]) {
  WORD our_diff = 0;
  BYTE diff_flag = 1;
  BYTE hash_int;
  BYTE hash_bit;
  BYTE hash_bit2;
  for(unsigned int i = 0; i < 256; i++) {
    if (diff_flag == 1) {
      hash_int = i / 8;
      hash_bit = (7 - (i % 8));
      hash_bit2 = ((h[hash_int]) >> hash_bit) & 1;
      if (hash_bit2 == 1) {
        diff_flag = 0;
        our_diff *= 256;
        our_diff += ((h[hash_int] << (i % 8)) + (h[hash_int+1] >> (hash_bit + 1)));
      } else {
        our_diff++;
      }
    }
  }
  return our_diff;
}

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

//Amoveo's hash2int function to calculate difficulty
WORD hash2intg(BYTE h[32]) {
  WORD x = 0;
  WORD z = 0;
  BYTE b;
  for (int i = 0; i < 31; i++) {
    b = h[i];
    if (b == 0) {
      x += 8;
      continue;
    } else if (b < 2) {
      x += 7;
      z = h[i+1];
    } else if (b < 4) {
      x += 6;
      z = (h[i+1] / 2) + ((h[i] % 2) * 128);
    } else if (b < 8) {
      x += 5;
      z = (h[i+1] / 4) + ((b % 4) * 64);
    } else if (b < 16) {
      x += 4;
      z = (h[i+1] / 8) + ((b % 8) * 32);
    } else if (b < 32) {
      x += 3;
      z = (h[i+1] / 16) + ((b % 16) * 16);
    } else if (b < 64) {
      x += 2;
      z = (h[i+1] / 32) + ((b % 32) * 8);
    } else if (b < 128) {
      x += 1;
      z = (h[i+1] / 64) + ((b % 64) * 4);
    } else {
      z = (h[i+1] / 128) + ((b % 128) * 2);
    }
    break;
  }
  return 256*x+z;
}

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

int diffic_test()
{
  BYTE hash1[SHA256_BLOCK_SIZE] = {0x00,0x00,0x06,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
                                   0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
  BYTE hash2[SHA256_BLOCK_SIZE] = {0x01,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,
                                   0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1};
  BYTE hash3[SHA256_BLOCK_SIZE] = {0x00,0x00,0x00,0x00,0x00,0x1f,0xf0,0x92,0x81,0xa1,0xc7,0xe2,0x84,0xd7,0x3e,0x67,
                                   0xf1,0x80,0x9a,0x48,0xa4,0x97,0x20,0x0e,0x04,0x6d,0x39,0xcc,0xc7,0x11,0x2c,0xd0};
  WORD words[8];
  WORD dfc;

  dfc = h2i(hash1);
  printf("** h2i Difficulty: %d\n", dfc);
  dfc = hash2int(hash1);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2integer(hash1);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2intg(hash1);
  printf("Difficulty: %d\n", dfc);
//  for(int i = 0; i < sizeof(hash1); i++)  printf("%02X.",hash1[i]);
//  printf("\n");
  bytes_2_words(words, hash1, 8);
  reverse_order(hash1, sizeof(hash1));
//  for(int i = 0; i < sizeof(hash1); i++)  printf("%02X.",hash1[i]);
//  printf("\n");
  dfc = h2i_rvs(hash1);
  printf("Difficulty: %d\n", dfc);
//  for(int i = 0; i < sizeof(words); i++)  printf("%08X.",words[i]);
//  printf("\n");
  dfc = h2i_w(words);
  printf("** h2i_w     Difficulty: %d\n", dfc);
  dfc = h2i_w_new(words);
  printf("** h2i_w_new Difficulty: %d\n", dfc);
  printf("\n");

  dfc = h2i(hash2);
  printf("** h2i Difficulty: %d\n", dfc);
  dfc = hash2int(hash2);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2integer(hash2);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2intg(hash2);
  printf("Difficulty: %d\n", dfc);
  bytes_2_words(words, hash2, 8);
  reverse_order(hash2, sizeof(hash2));
  dfc = h2i_rvs(hash2);
  printf("Difficulty: %d\n", dfc);
  dfc = h2i_w(words);
  printf("** h2i_w     Difficulty: %d\n", dfc);
  dfc = h2i_w_new(words);
  printf("** h2i_w_new Difficulty: %d\n", dfc);
  printf("\n");

  dfc = h2i(hash3);
  printf("** h2i Difficulty: %d\n", dfc);
  dfc = hash2int(hash3);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2integer(hash3);
  printf("Difficulty: %d\n", dfc);
  dfc = hash2intg(hash3);
  printf("Difficulty: %d\n", dfc);
  bytes_2_words(words, hash3, 8);
  reverse_order(hash3, sizeof(hash3));
  dfc = h2i_rvs(hash3);
  printf("Difficulty: %d\n", dfc);
  dfc = h2i_w(words);
  printf("** h2i_w     Difficulty: %d\n", dfc);
  dfc = h2i_w_new(words);
  printf("** h2i_w_new Difficulty: %d\n", dfc);
  printf("\n");
  return dfc;
}

int main() {
  printf("Hash difficulty test: %d\n", diffic_test());

  BYTE text[4] = {0xfe,0xfe,0xfe,0xfe};

//  while (1) {
//  for (int i = 0; i < 4; i++) {
//    text[i] += 1;
//    if (text[i] != 0) {
//      break;
//    }
//  }
//  for(int i = 0; i < 4; i++) printf("%02X.",text[i]);
//  printf("\n");
//  getchar();
//  }

  return(0);
}
