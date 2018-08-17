/*********************************************************************
 * Filename:   sha256.c
 * Author:     Brad Conte (brad AT bradconte.com)
 * Copyright:
 * Disclaimer: This code is presented "as is" without any guarantees.
 * Details:    Implementation of the SHA-256 hashing algorithm.
              SHA-256 is one of the three algorithms in the SHA2
              specification. The others, SHA-384 and SHA-512, are not
              offered in this implementation.
              Algorithm specification can be found here:
              * http://csrc.nist.gov/publications/fips/fips180-2/fips180-2withchangenotice.pdf
              This implementation uses little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include "sha256.h"

/**************************** VARIABLES *****************************/
static const WORD k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

unsigned int rotr(unsigned int x, int s) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;

  *ms = x;
  val = val >> s;

  return *ms | *ls;
}

unsigned int ep0(unsigned int x) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val; //least significant
  unsigned int *ms = ls + 1;  // least significant
  unsigned int res;

  *ms = x;
  val = val >> 2;
  res = *ms | *ls;

  val = val >> 11;
  res = res ^ (*ms | *ls);

  val = val >> 9;

  return res ^ (*ms | *ls);
}

unsigned int ep1(unsigned int x) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;
  unsigned int res;

  *ms = x;
  val = val >> 6;
  res = *ms | *ls;

  val = val >> 5;
  res = res ^ (*ms | *ls);

  val = val >> 14;

  return res ^ (*ms | *ls);
}

unsigned int sig0(unsigned int x) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;
  unsigned int res = 0;

  *ms = x;
  val = val >> 7;
  res = *ms | *ls;

  val = val >> 11;
  res = res ^ (*ms | *ls);

  return res ^ (x >> 3);
}

unsigned int sig1(unsigned int x) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;
  unsigned int res = 0;

  *ms = x;
  val = val >> 17;
  res = *ms | *ls;

  val = val >> 2;
  res = res ^ (*ms | *ls);

  return res ^ (x >> 10);
}

WORD hash2integer(BYTE h[32]) {
  WORD x = 0;
  WORD y;
  BYTE hi;
  for (int i = 0; i < 31; i++) {
    hi = h[i];

    if (hi < 16) {
      if (hi < 4) {
        if (hi < 2) {
          if (hi == 0) {
            // == 0
            x += 8;
            y = h[i+1];
            continue;
          } else {
            // == 1
            x += 7;
            y = (hi * 128) + (h[i+1] / 2);
          }
        } else {
          // 2..3
          x += 6;
          y = (hi * 64) + (h[i+1] / 4);
        }
      } else {
        if (hi < 8) {
          // == 7..4
          x += 5;
          y = (hi * 32) + (h[i+1] / 8);
        } else {
          // 15..8
          x += 4;
          y = (hi * 16) + (h[i+1] / 16);
        }
      }
    } else { // >= 16
      if (hi < 64) {
        if (hi < 32) {
          // 31..16
          x += 3;
          y = (hi * 8) + (h[i+1] / 32);
        } else {
          // 63..32
          x += 2;
          y = (hi * 4) + (h[i+1] / 64);
        }
      } else {
        if (hi < 128) {
          // 127..64
          x += 1;
          y = (hi * 2) + (h[i+1] / 128);
        } else {
          // 255..128
          y = hi;
        }
      }
    }

//    if (hi == 0) {
//      x += 8;
//      y = h[i+1];
//      continue;
//    } else if (hi < 2) { // hi == 1
//      x += 7;
//      y = (hi * 128) + (h[i+1] / 2);
//    } else if (hi < 4) { // hi == 2,3
//      x += 6;
//      y = (hi * 64) + (h[i+1] / 4);
//    } else if (hi < 8) { // hi == 7,6,5,4
//      x += 5;
//      y = (hi * 32) + (h[i+1] / 8);
//    } else if (hi < 16) { // hi 15..8
//      x += 4;
//      y = (hi * 16) + (h[i+1] / 16);
//    } else if (hi < 32) { // hi 31..16
//      x += 3;
//      y = (hi * 8) + (h[i+1] / 32);
//    } else if (hi < 64) { // hi 63..32
//      x += 2;
//      y = (hi * 4) + (h[i+1] / 64);
//    } else if (hi < 128) { // hi 127..64
//      x += 1;
//      y = (hi * 2) + (h[i+1] / 128);
//    } else { // hi 255..128
//      y = hi;
//    }
    break;
  }
  return ((256 * x) + y);
}

/*********************** FUNCTION DEFINITIONS ***********************/
/* data[] converts to ctx->state[]                                  */

void sha256_init(SHA256_CTX *ctx)
{
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

void sha256_transform(SHA256_CTX *ctx, const BYTE data[]) {
  WORD x, res0, res1;
  WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
  BYTE * ptr = (BYTE*)&m,
      * ptr1 = ptr + 1,
      * ptr2 = ptr + 2,
      * ptr3 = ptr + 3,
      * data_ptr = (BYTE*)data;

  for (i = 0, j = 0; i < 16; ++i, j += 4) {
    *(ptr3 + j) = *(data_ptr++);
    *(ptr2 + j) = *(data_ptr++);
    *(ptr1 + j) = *(data_ptr++);
    *(ptr + j) = *(data_ptr++);
  }
//    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  for ( ; i < 64; ++i) {
// sig0:
    x = m[i - 15];
    res0 = (x >> 7) | (x << 25);
    res0 ^= (x >>18) | (x << 14);
    res0 ^= (x >> 3);
//sig1:
    x = m[i - 2];
    res1 = (x >> 17) | (x << 15);
    res1 ^= (x >> 19) | (x << 13);
    res1 ^= (x >> 10);

    m[i] = res1 + m[i - 7] + res0 + m[i - 16];
  }
  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];

  for (i = 0; i < 64; ++i) {
// ep0:
    x = a;
    res0 = (x >> 2) | (x << 30);
    res0 ^= (x >> 13) | (x << 19);
    res0 ^= (x >> 22) | (x << 10);
// ep1:
    x = e;
    res1 = (x >> 6) | (x << 26);
    res1 ^= (x >> 11) | (x << 21);
    res1 ^= (x >> 25) | (x << 7);

    t1 = h + res1 + ((e & f) ^ (~e & g)) + k[i] + m[i];
    t2 = res0 + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
  WORD i;
  WORD dl = ctx->datalen;
  for (i = 0; i < len; ++i) {
    ctx->data[dl] = data[i];
    dl++;
    if (dl == 64) {
      sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      dl = 0;
    }
  }
  ctx->datalen = dl;
}

void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
  WORD i = ctx->datalen;

  // Pad whatever data is left in the buffer.
  ctx->data[i++] = 0x80;
  if (i < 56) {
    while (i < 56)
      ctx->data[i++] = 0x00;
  } else {
    while (i < 64)
      ctx->data[i++] = 0x00;
    sha256_transform(ctx, ctx->data);
    memset(ctx->data, 0, 56);
  }

  // Append to the padding the total message's length in bits and transform.
  ctx->bitlen += ctx->datalen * 8;
  unsigned long long int bl = ctx->bitlen;
  ctx->data[63] = bl;
  ctx->data[62] = bl >> 8;
  ctx->data[61] = bl >> 16;
  ctx->data[60] = bl >> 24;
  ctx->data[59] = bl >> 32;
  ctx->data[58] = bl >> 40;
  ctx->data[57] = bl >> 48;
  ctx->data[56] = bl >> 56;
  sha256_transform(ctx, ctx->data);

  // Since this implementation uses little endian byte ordering and SHA uses big endian,
  // reverse all the bytes when copying the final state to the output hash.
  BYTE * ptr = (BYTE*)&ctx->state;
  for (i = 0; i < 4; ++i) {
    hash[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
  }
}
