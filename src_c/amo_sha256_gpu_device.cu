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

__global__ void kernel_test(volatile bool *stop, volatile long int *cycles, GPU_thread_info * info_debug) {
  int i, k;
  long int r;
  WORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  WORD sizeRow = gridDim.x * blockDim.x;

  r = 0;
  while (!(*stop)) {
    for (i = 0; i < 1000000; i++) {
      k = k * 2;
    }
    r++;
    if (idx == (sizeRow -1)) {
      *cycles = r;
    }
  }
  if (idx >= (sizeRow - 1000)) {
    info_debug[idx - sizeRow + 1000].blockIdx = blockIdx.x;
    info_debug[idx - sizeRow + 1000].blockIdy = blockIdx.y;
    info_debug[idx - sizeRow + 1000].threadIdx = threadIdx.x;
    info_debug[idx - sizeRow + 1000].threadIdy = threadIdx.y;
    info_debug[idx - sizeRow + 1000].flag = r;
  }
}

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, long int * cycles_total) {
  int i, j, work, index;
  long int r;
  WORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  AMO_SHA256_CTX ctx;

  for (i = 0, j = 0; i < 8; ++i, j += 4)
    ctx.data[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);

  for (i = 8, j = 0; i < 13; ++i, j += 4)
    ctx.data[i] = (nonce[j] << 24) | (nonce[j + 1] << 16) | (nonce[j + 2] << 8) | (nonce[j + 3]);
//i = 8, j = 0
//i = 9, j = 4
//i =10, j = 8
//i =11, j =12
//i =12, j =16
//i =13, j =20

  ctx.data[13] = (nonce[20] << 24) | (nonce[21] << 16) | (nonce[22] << 8) | (0x80);

  unsigned long long int bl = 55 * 8;
  ctx.data[14] = (WORD)(bl >> 32);
  ctx.data[15] = (WORD)bl;

  ctx.data[8] = ctx.data[8] ^ idx;
  ctx.data[9] = ctx.data[9] ^ device_id;

  r = 0;
  index = idx % 100;
  while (true) {
    if ((r % 100) == index) {
      if (*stop) {
//        *cycles = r;
//        __threadfence();
//        asm("trap;");
        break;
      }
    }

    for (i = 10; i < 13; i++) {
      ctx.data[i] += 1;
      if (ctx.data[i] != 0) {
        break;
      }
    }

    sha256_init(&ctx);
    sha256_transform(&ctx);
    r++;

    work = hash2int_w(ctx.state);
    if( work > difficulty) {
      *success = true;
      *stop = true;
      BYTE * ptrn = (BYTE*)(&(ctx.data[8]));
      for (i = 0; i < 20; i += 4) {
        nonce[i]     = *(ptrn + 3 + i);
        nonce[i + 1] = *(ptrn + 2 + i);
        nonce[i + 2] = *(ptrn + 1 + i);
        nonce[i + 3] = *(ptrn + i);
      }
      nonce[i]     = *(ptrn + 3 + i);
      nonce[i + 1] = *(ptrn + 2 + i);
      nonce[i + 2] = *(ptrn + 1 + i);

      BYTE * ptr = (BYTE*)ctx.state;
      for (i = 0; i < 32; i += 4) {
        data[i]     = *(ptr + 3 + i);
        data[i + 1] = *(ptr + 2 + i);
        data[i + 2] = *(ptr + 1 + i);
        data[i + 3] = *(ptr + i);
      }
//      *cycles = r;

//      __threadfence();
//      asm("trap;");
//      break;
    }
//    if( work > 9000 ) {
//      BYTE * ptr = (BYTE*)&ctx.state;
//      for (int i = 0; i < 32; i += 4) {
//        data[i]     = *(ptr + 3 + i);
//        data[i + 1] = *(ptr + 2 + i);
//        data[i + 2] = *(ptr + 1 + i);
//        data[i + 3] = *(ptr + i);
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
  cycles_total[idx] = r;
  if (idx == 1) {
    *cycles = r;
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
          our_diff += ((h[i] << (j - 24)) + (((h[i + 1] >> 1) & 0x7FFFFFFF) >> (55 - j)));
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
__device__ void sha256_transform(AMO_SHA256_CTX *ctx) {
  WORD x, res0, res1;
  WORD a, b, c, d, e, f, g, h, i, t1, t2, m[64];

  for (i = 0; i < 16; ++i)
    m[i] = ctx->data[i];
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

__device__ void sha256_init(AMO_SHA256_CTX *ctx) {
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}
