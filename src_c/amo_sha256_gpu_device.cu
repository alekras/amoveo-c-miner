#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
  #include "sha256.h"
}
#include "sha256_gpu.h"

/** MACROSES **/

/*  Macros sig0 is doing:
    res = (x >> 7) | (x << 25);
    res ^= (x >>18) | (x << 14);
    res ^= (x >> 3);
 assembler commands:
1. temp reg t1,
2. temp reg t2,
3. t1 = x >> 7
4. t2 = x >> 18
5. t1 = t1 ^ t2
6. t2 = x >> 3
7. res = t1 ^ t2
 */
#define sig0(x,res) \
  asm("{\n\t" \
      " .reg .u32 t1;\n\t" \
      " .reg .u32 t2;\n\t" \
      " shf.r.clamp.b32    t1, %1, %1, 7;\n\t" \
      " shf.r.clamp.b32    t2, %1, %1, 18;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " shr.u32            t2, %1, 3;\n\t" \
      " xor.b32            %0, t1, t2;\n\t" \
      "}" \
    : "=r"(res) : "r" (x));

/*  Macros sig1 is doing:
    res = (x >> 17) | (x << 15);
    res ^= (x >> 19) | (x << 13);
    res ^= (x >> 10);
 assembler commands:
1. temp reg t1,
2. temp reg t2,
3. t1 = x >> 17
4. t2 = x >> 19
5. t1 = t1 ^ t2
6. t2 = x >> 10
7. res = t1 ^ t2
 */
#define sig1(x,res) \
  asm("{\n\t" \
      " .reg .u32 t1;\n\t" \
      " .reg .u32 t2;\n\t" \
      " shf.r.clamp.b32    t1, %1, %1, 17;\n\t" \
      " shf.r.clamp.b32    t2, %1, %1, 19;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " shr.u32            t2, %1, 10;\n\t" \
      " xor.b32            %0, t1, t2;\n\t" \
      "}" \
    : "=r"(res) : "r" (x));

/*  Macros ep0 is doing:
    res = (x >> 2) | (x << 30);
    res ^= (x >> 13) | (x << 19);
    res ^= (x >> 22) | (x << 10);
 assembler commands:
1. temp reg t1,
2. temp reg t2,
3. t1 = x >> 2
4. t2 = x >> 13
5. t1 = t1 ^ t2
6. t2 = x >> 22
7. res = t1 ^ t2
 */
#define ep0(x,res) \
    asm("{\n\t" \
        " .reg .u32 t1;\n\t" \
        " .reg .u32 t2;\n\t" \
        " shf.r.clamp.b32    t1, %1, %1, 2;\n\t" \
        " shf.r.clamp.b32    t2, %1, %1, 13;\n\t" \
        " xor.b32            t1, t1, t2;\n\t" \
        " shf.r.clamp.b32    t2, %1, %1, 22;\n\t" \
        " xor.b32            %0, t1, t2;\n\t" \
        "}" \
        : "=r"(res) : "r" (x));

/*  Macros ep1 is doing:
    res = (x >> 6) | (x << 26);
    res ^= (x >> 11) | (x << 21);
    res ^= (x >> 25) | (x << 7);
 assembler commands:
1. temp reg t1,
2. temp reg t2,
3. t1 = x >> 6
4. t2 = x >> 11
5. t1 = t1 ^ t2
6. t2 = x >> 25
7. res = t1 ^ t2
 */
#define ep1(x,res) \
    asm("{\n\t" \
        " .reg .u32 t1;\n\t" \
        " .reg .u32 t2;\n\t" \
        " shf.r.clamp.b32    t1, %1, %1, 6;\n\t" \
        " shf.r.clamp.b32    t2, %1, %1, 11;\n\t" \
        " xor.b32            t1, t1, t2;\n\t" \
        " shf.r.clamp.b32    t2, %1, %1, 25;\n\t" \
        " xor.b32            %0, t1, t2;\n\t" \
        "}" \
        : "=r"(res) : "r" (x));
/*
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
*/

//extern __shared__ WORD shared_memory[];

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, long int * cycles_total) {
  int i, j, work;
  long int r;
  long int idx = blockIdx.x * blockDim.x + threadIdx.x;
  AMO_SHA256_CTX ctx;

  for (i = 0, j = 0; i < 8; ++i, j += 4) {
    ctx.data[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  }

  for (i = 8, j = 0; i < 13; ++i, j += 4) {
    ctx.data[i] = (nonce[j] << 24) | (nonce[j + 1] << 16) | (nonce[j + 2] << 8) | (nonce[j + 3]);
  }
  ctx.data[13] = (nonce[20] << 24) | (nonce[21] << 16) | (nonce[22] << 8) | (0x80);

  unsigned long long int bl = 55 * 8;
  ctx.data[14] = (WORD)(bl >> 32);
  ctx.data[15] = (WORD)bl;

  ctx.data[9] = ctx.data[9] ^ idx;
  ctx.data[10] = ctx.data[10] ^ device_id;

  r = 0;
  int index = 99; //(int)(idx % 100);
  while (true) {
    if ((r % 100) == index) {
      if (*stop) {
//        *cycles = r;
//        __threadfence();
//        asm("trap;");
        break;
      }
    }

    if ((++ctx.data[11]) == 0) {
      ctx.data[12]++;
    }
//    for (i = 11; i < 13; i++) { // do not touch [13]
//      ctx.data[i]++;
//      if (ctx.data[i] != 0) {
//        break;
//      }
//    }

    r++;
    sha256_init(&ctx);
    sha256_transform(&ctx);
    work = hash2int_w(ctx.state);
    if( work > difficulty) {
      *success = true;
      *stop = true;
      BYTE * ptrn = (BYTE*)(&(ctx.data[8]));
      #pragma unroll 1
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
      #pragma unroll 1
      for (i = 0; i < 32; i += 4) {
        data[i]     = *(ptr + 3 + i);
        data[i + 1] = *(ptr + 2 + i);
        data[i + 2] = *(ptr + 1 + i);
        data[i + 3] = *(ptr + i);
      }
    }
  }
  cycles_total[idx] = r;
  if (idx == 1) {
    *cycles = r;
  }
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
        if (j == 31) {
          j = 0;
          i++;
        } else {
          j++;
        }
        if ((24 - j) >= 0) {
          our_diff += (h[i] >> (24 -j)) & 0xff;
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
__device__ __constant__ static const WORD k[64] = {
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
  WORD a, b, c, d, e, f, g, h, i, t1, m[64];

  for (i = 0; i < 16; ++i)
    m[i] = ctx->data[i];
  for ( ; i < 64; ++i) {
    x = m[i - 15];
    sig0(x, res0)
    x = m[i - 2];
    sig1(x,res1)

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
    ep0(a,res0)
    ep1(e,res1)

    t1 = h + res1 + ((e & f) ^ (~e & g)) + k[i] + m[i];
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    t1 += res0 + ((a & b) ^ (a & c) ^ (b & c));
    c = b;
    b = a;
    a = t1;
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
