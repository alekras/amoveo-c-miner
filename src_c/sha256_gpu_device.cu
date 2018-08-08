#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
  #include "sha256.h"
  extern FILE *fdebug;
}
#include "sha256_gpu.h"

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, GPU_thread_info * info_debug) {
  int i, j, k, work;
  long int r;
  WORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  WORD jdx = blockIdx.y * blockDim.y + threadIdx.y;
  WORD sizeRow = gridDim.x * blockDim.x;
  WORD sizeCol = gridDim.y * blockDim.y;

  BYTE text[55];//32+23
  SHA256_CTX ctx;
  BYTE hash[32];
  for (i = 0; i < 32; i++) {
    text[i] = data[i];
  }

  k = 0;
  for (j = 0; i < 36; i++, k++, j += 8) { // k = 0
    text[i] = nonce[k] ^ ((BYTE)(idx >> j));
  }
  for (j = 0; i < 40; i++, k++, j += 8) { // k = 4
    text[i] = nonce[k] ^ ((BYTE)(jdx >> j));
  }
  text[i++] = nonce[k++] ^ ((BYTE)device_id);

  for (;i < 55; i++, k++) { // i = 41, k = 9
    text[i] = nonce[k];
  }

  r = 0;
  while (!(*stop)) {
    if(*success) {
      break;
    }
    for (i = 41; i < 55; i++) {
      text[i] += 1;
      if (text[i] != 0) {
        break;
      }
    }

    d_sha256_init(&ctx);
    d_sha256_update(&ctx,text,55);
    d_sha256_final(&ctx,hash);
    r++;
    if ((idx == 0) && (jdx == 0)) {
      *cycles = r;
    }

    work = hash2int_w(ctx.state);
    if( work > difficulty) {
      *success = true;
      for (i = 0; i < 23; i++) {
        nonce[i] = text[i + 32];
      }
      BYTE * ptr = (BYTE*)ctx.state;
      for (i = 0; i < 4; ++i) {
        data[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        data[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        data[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        data[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        data[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        data[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        data[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        data[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
      }
//      for (i = 0; i < 32; i++) {
//        data[i] = hash[i];
//      }
//      break;
    }
//    if( work > (difficulty >> 1) ) {
//      BYTE * ptr = (BYTE*)&ctx.state;
//      for (i = 0; i < 4; ++i) {
//        data[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
//        data[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
//      }
//      for (i = 0; i < 32; i++) {
//        data[i] = hash[i];
//      }
//    }
  }
//  int offset = jdx * sizeRow + idx;
//  if (offset < 2048) {
//    info_debug[offset].flag = 7;
//    info_debug[offset].blockIdx = blockIdx.x;
//    info_debug[offset].blockIdy = blockIdx.y;
//    info_debug[offset].threadIdx = threadIdx.x;
//    info_debug[offset].threadIdy = threadIdx.y;
//  }
  if ((idx == 0) && (jdx == 0)) {
    *cycles = r;
  }
}

__global__ void kernel_sha256_val(BYTE *data, WORD len, BYTE *hash, WORD cycle) {

  SHA256_CTX ctx;
  d_sha256_init(&ctx);

  int idx;
  for (idx = 0; idx < cycle; ++idx)
    d_sha256_update(&ctx, data, len);

  d_sha256_final(&ctx,hash);

  BYTE * ptr = (BYTE*)ctx.state;
  for (int i = 0; i < 4; ++i) {
    hash[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
  }
  for (int i = 0; i < 32; i += 4) {
	hash[i]     = *(ptr + 3 + i);  
	hash[i + 1] = *(ptr + 2 + i);  
	hash[i + 2] = *(ptr + 1 + i);  
	hash[i + 3] = *(ptr + i);  
  }
}

__device__ WORD hash2int(BYTE h[32]) {
  WORD our_diff = 0;

  for(int i = 0; i < 31; i++) {
    BYTE mask = 0x80;
    for(int j = 0; j < 8; j++) {
      if ( (h[i] & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
        our_diff += ((h[i] << j) + (h[i + 1] >> (8 - j)));
        return our_diff;
      }
    }
  }
  return our_diff;
}

__device__ WORD hash2int_w(WORD h[8]) {
  WORD our_diff = 0;

  for(int i = 0; i < 8; i++) {
    WORD mask = 0x80000000;
    for(int j = 0; j < 32; j++) {
      if ( (h[i] & mask) == 0 ) {
        our_diff++;
        mask = mask >> 1;
      } else {
        our_diff *= 256;
        if ((24 - j) >= 0) {
          our_diff += (h[i] >> (24 -j));
        } else {
          our_diff += ((h[i] << (j - 24)) + (h[i + 1] >> (56 - j)));
        }
        return our_diff;
      }
    }
  }
  return our_diff;
}

//Constants for SHA-256
__device__ static const WORD k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

//SHA-256 functions taken from Brad Conte's implementation
//https://github.com/B-Con/crypto-algorithms/blob/master/sha256.c
__device__ void d_sha256_transform(SHA256_CTX *ctx, const BYTE data[]) {
  WORD x, res0, res1;
  WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
//  BYTE * ptr = (BYTE*)&m,
//    * ptr1 = ptr + 1,
//    * ptr2 = ptr + 2,
//    * ptr3 = ptr + 3,
//    * data_ptr = (BYTE*)data;
//
//  for (i = 0, j = 0; i < 16; ++i, j += 4) {
//    *(ptr3 + j) = *(data_ptr++);
//    *(ptr2 + j) = *(data_ptr++);
//    *(ptr1 + j) = *(data_ptr++);
//    *(ptr + j) = *(data_ptr++);
//  }

  for (i = 0, j = 0; i < 16; ++i, j += 4)
    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  for ( ; i < 64; ++i) {
// sig0:
    x = m[i - 15];
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 7;\n\t"  // t1 = x >> 7
        " shf.r.clamp.b32    t2, %1, %1, 18;\n\t"  // t2 = x >> 18
        " xor.b32            t1, t1, t2;\n\t"
        " shr.u32            t2, %1, 3;\n\t" // t2 = x >> 3
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res0) : "r" (x));
//    res0 = (x >> 7) | (x << 25);
//    res0 ^= (x >>18) | (x << 14);
//    res0 ^= (x >> 3);
//sig1:
    x = m[i - 2];
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 17;\n\t"  // t1 = x >> 17
        " shf.r.clamp.b32    t2, %1, %1, 19;\n\t"  // t2 = x >> 19
        " xor.b32            t1, t1, t2;\n\t"
        " shr.u32            t2, %1, 10;\n\t" // t2 = x >> 10
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res1) : "r" (x));
//    res1 = (x >> 17) | (x << 15);
//    res1 ^= (x >> 19) | (x << 13);
//    res1 ^= (x >> 10);

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
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 2;\n\t"  // t1 = x >> 2
        " shf.r.clamp.b32    t2, %1, %1, 13;\n\t"  // t2 = x >> 13
        " xor.b32            t1, t1, t2;\n\t"
        " shf.r.clamp.b32    t2, %1, %1, 22;\n\t" // t2 = x >> 22
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res0) : "r" (a));
//    x = a;
//    res0 = (x >> 2) | (x << 30);
//    res0 ^= (x >> 13) | (x << 19);
//    res0 ^= (x >> 22) | (x << 10);
// ep1:
    asm("{\n\t"
        " .reg .u32 t1;\n\t"            // temp reg t1,
        " .reg .u32 t2;\n\t"            // temp reg t2,
        " shf.r.clamp.b32    t1, %1, %1, 6;\n\t"  // t1 = x >> 6
        " shf.r.clamp.b32    t2, %1, %1, 11;\n\t"  // t2 = x >> 11
        " xor.b32            t1, t1, t2;\n\t"
        " shf.r.clamp.b32    t2, %1, %1, 25;\n\t" // t2 = x >> 25
        " xor.b32            %0, t1, t2;\n\t"
        "}"
        : "=r"(res1) : "r" (e));
//    x = e;
//    res1 = (x >> 6) | (x << 26);
//    res1 ^= (x >> 11) | (x << 21);
//    res1 ^= (x >> 25) | (x << 7);

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

__device__ void d_sha256_init(SHA256_CTX *ctx) {
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

__device__ void d_sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len) {
  WORD i;
  WORD dl = ctx->datalen;

  for (i = 0; i < len; ++i) {
    ctx->data[dl] = data[i];
    dl++;
    if (dl == 64) {
      d_sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      dl = 0;
    }
  }
  ctx->datalen = dl;
}

__device__ void d_sha256_final(SHA256_CTX *ctx, BYTE hash[]) {
  WORD i = ctx->datalen;

  // Pad whatever data is left in the buffer.
  ctx->data[i++] = 0x80;
  if (i < 56) {
    while (i < 56)
      ctx->data[i++] = 0x00;
  } else {
    while (i < 64)
      ctx->data[i++] = 0x00;
    d_sha256_transform(ctx, ctx->data);
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
  d_sha256_transform(ctx, ctx->data);

  // Since this implementation uses little endian byte ordering and SHA uses big endian,
  // reverse all the bytes when copying the final state to the output hash.
//  BYTE * ptr = (BYTE*)&ctx->state;
//  for (i = 0; i < 4; ++i) {
//    hash[i]      = *(ptr + 3 - i); //(ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 4]  = *(ptr + 7 - i); //(ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 8]  = *(ptr + 11 - i); //(ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 12] = *(ptr + 15 - i); //(ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 16] = *(ptr + 19 - i); //(ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 20] = *(ptr + 23 - i); //(ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 24] = *(ptr + 27 - i); //(ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
//    hash[i + 28] = *(ptr + 31 - i); //(ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
//  }
}
