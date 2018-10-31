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

/*  Macros sig0_1 is doing:
    res0 = (x >> 7) | (x << 25);
    res0 ^= (x >>18) | (x << 14);
    res0 ^= (x >> 3);
    res1 = (x >> 17) | (x << 15);
    res1 ^= (x >> 19) | (x << 13);
    res1 ^= (x >> 10);
    res = res0 + res1
 assembler commands:
1. temp reg t1,
2. temp reg t2,
3. t1 = x >> 7
4. t2 = x >> 18
5. t1 = t1 ^ t2
6. t2 = x >> 3
7. res = t1 ^ t2
8. t1 = x >> 17
9. t2 = x >> 19
10. t1 = t1 ^ t2
11. t2 = x >> 10
12. t1 = t1 ^ t2
13. res = res + t1
 */
#define sig0_1(x,y,res) \
  asm("{\n\t" \
      " .reg .u32 t1;\n\t" \
      " .reg .u32 t2;\n\t" \
      " shf.r.clamp.b32    t1, %1, %1, 7;\n\t" \
      " shf.r.clamp.b32    t2, %1, %1, 18;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " shr.u32            t2, %1, 3;\n\t" \
      " xor.b32            %0, t1, t2;\n\t" \
      " shf.r.clamp.b32    t1, %2, %2, 17;\n\t" \
      " shf.r.clamp.b32    t2, %2, %2, 19;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " shr.u32            t2, %2, 10;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " add.s32            %0, %0, t1;\n\t" \
      "}" \
    : "+r"(res) : "r"(x), "r"(y));

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
        : "=r"(res) : "r"(x));

#define epp(a,b,c,d,e,f,g,h,km) \
    asm("{\n\t" \
        " .reg .u32 t1;\n\t" \
        " .reg .u32 t2;\n\t" \
        " .reg .u32 t3;\n\t" \
        " .reg .u32 res0;\n\t" \
        " .reg .u32 res1;\n\t" \
        " shf.r.clamp.b32    t1, %0, %0, 2;\n\t" \
        " shf.r.clamp.b32    t2, %0, %0, 13;\n\t" \
        " xor.b32            t1, t1, t2;\n\t" \
        " shf.r.clamp.b32    t2, %0, %0, 22;\n\t" \
        " xor.b32            res0, t1, t2;\n\t" \
 \
        " shf.r.clamp.b32    t1, %4, %4, 6;\n\t" \
        " shf.r.clamp.b32    t2, %4, %4, 11;\n\t" \
        " xor.b32            t1, t1, t2;\n\t" \
        " shf.r.clamp.b32    t2, %4, %4, 25;\n\t" \
        " xor.b32            res1, t1, t2;\n\t" \
        " add.s32            t1, %7, res1;\n\t"   /* t1 = h + res1 */ \
        " add.s32            t1, t1, %8;\n\t"     /* t1 = h + res1 + km */ \
        " not.b32            t2, %4;\n\t"         /* t2 = ~e */ \
        " and.b32            t2, t2, %6;\n\t"     /* t2 = (~e & g) */ \
        " and.b32            t3, %4, %5;\n\t"     /* t3 = (e & f) */ \
        " xor.b32            t2, t2, t3;\n\t"     /* t2 = ((e & f) ^ (~e & g)) */ \
        " add.s32            t1, t1, t2;\n\t"     /* t1 = h + res1 + km + ((e & f) ^ (~e & g)) */ \
        " mov.u32            %7, %6;\n\t"         /* h = g */ \
        " mov.u32            %6, %5;\n\t"         /* g = f */ \
        " mov.u32            %5, %4;\n\t"         /* f = e */ \
        " add.s32            %4, %3, t1;\n\t"     /* e = d + t1 */ \
        " mov.u32            %3, %2;\n\t"         /* f = e */ \
        " xor.b32            t3, %1, %2;\n\t"     /* t3 = (b ^ c) */ \
        " and.b32            t3, %0, t3;\n\t"     /* t3 = a & (b ^ c) */ \
        " and.b32            t2, %1, %2;\n\t"     /* t2 = (b & c) */ \
        " xor.b32            t2, t2, t3;\n\t"     /* t2 = (a & (b ^ c)) ^ (b & c) */ \
        " add.s32            t1, t1, res0;\n\t"   /* t1 = t1 + res0 */ \
        " add.s32            t1, t1, t2;\n\t"     /* t1 = t1 + res0 + (a & (b ^ c)) ^ (b & c) */ \
        " mov.u32            %2, %1;\n\t"         /* c = b */ \
        " mov.u32            %1, %0;\n\t"         /* b = a */ \
        " mov.u32            %0, t1;\n\t"         /* a = t1 */ \
        "}" \
        : "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e), "+r"(f), "+r"(g), "+r"(h) : "r"(km));
/*             0        1        2        3        4        5        6        7         8  */

#define step(h0,h1) \
  hi = h0; hii = h1; \
  for (int j = 0; j < 32; j++) { \
    if ( (hi & mask) == 0 ) { \
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
__device__ __constant__ static const WORD a0 = 0x6a09e667;
__device__ __constant__ static const WORD b0 = 0xbb67ae85;
__device__ __constant__ static const WORD c0 = 0x3c6ef372;
__device__ __constant__ static const WORD d0 = 0xa54ff53a;
__device__ __constant__ static const WORD e0 = 0x510e527f;
__device__ __constant__ static const WORD f0 = 0x9b05688c;
__device__ __constant__ static const WORD g0 = 0x1f83d9ab;
__device__ __constant__ static const WORD h0 = 0x5be0cd19;

//extern __shared__ WORD shared_memory[];

__global__ void kernel_sha256(BYTE *data, WORD difficulty, BYTE *nonce, volatile bool *success, volatile bool *stop, volatile long int *cycles, WORD device_id, long int * cycles_total) {
  WORD i, j, our_diff;
  long int r;
  long int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  AMO_SHA256_CTX ctx;
  WORD ctx_data[27];
//  WORD ctx_state[8];
  WORD res0, res1;
  WORD m[64];
  WORD a, b, c, d, e, f, g, h, t1;
//  WORD *m = &shared_memory[64 * threadIdx.x];

  #pragma unroll 1
  for (i = 0, j = 0; i < 8; ++i, j += 4) {
    ctx_data[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  }

  #pragma unroll 1
  for (i = 8, j = 0; i < 13; ++i, j += 4) {
    ctx_data[i] = (nonce[j] << 24) | (nonce[j + 1] << 16) | (nonce[j + 2] << 8) | (nonce[j + 3]);
  }
  ctx_data[13] = (nonce[20] << 24) | (nonce[21] << 16) | (nonce[22] << 8) | (0x80);

  unsigned long long int bl = 55 * 8;
  ctx_data[14] = (WORD)(bl >> 32);
  ctx_data[15] = (WORD)bl;

  ctx_data[10] = ctx_data[10] ^ idx;
  ctx_data[11] = ctx_data[11] ^ device_id;
// extended data - constants:

#pragma unroll 1
  for (int i = 16 ; i < 21; ++i) {
//    sig0(ctx_data[i - 15], res0)
//    sig1(ctx_data[i - 2], res1)
    sig0_1(ctx_data[i - 15], ctx_data[i - 2], res0)
    ctx_data[i] = ctx_data[i - 7] + res0 + ctx_data[i - 16];
  }
#pragma unroll 1
  for (int i = 21 ; i < 27; ++i) {
    sig0(ctx_data[i - 15], res0)
//    sig1(ctx_data[i - 2], res1)
    ctx_data[i] = ctx_data[i - 7] + res0 + ctx_data[i - 16];
  }

  r = 0;
  int index = 9; //(int)(idx % 100);
  while (true) {
    if ((r % 100) == index) {
      if (*stop) {
//        *cycles = r;
//        __threadfence();
//        asm("trap;");
        break;
      }
    }

    if ((++ctx_data[12]) == 0) {
      ctx_data[13] += 0x0010;
      ctx_data[20] += 0x0010; // depends on [13]
    }
    ctx_data[19]++; // depends on [12]
    ctx_data[26]++; // depends on [19] -> [12]

    r++;
//    sha256_init(ctx_state);
//    sha256_transform(ctx_data, ctx_state);
//    __device__ void sha256_transform(WORD *ctx_data, WORD *ctx_state) {
    {
    //#pragma unroll 1
      for (i = 0; i < 27; ++i)
        m[i] = ctx_data[i];

      for (i = 21 ; i < 27; ++i) {
    //    sig0(ctx_data[i - 15], res0)
        sig1(m[i - 2], res1)
        m[i] += res1;
      }
    //#pragma unroll 1
      for (i = 27 ; i < 64; ++i) {
//        sig0(m[i - 15], res0)
//        sig1(m[i - 2], res1)
        sig0_1(m[i - 15], m[i - 2], res0)
        m[i] = m[i - 7] + res0 + m[i - 16];
      }

      a = a0;
      b = b0;
      c = c0;
      d = d0;
      e = e0;
      f = f0;
      g = g0;
      h = h0;

      for (i = 0; i < 64; i++) m[i] += k[i];
      i = 0;
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])

      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])

      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])

      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])
      epp(a,b,c,d,e,f,g,h,m[i++])

//#pragma unroll 1
//      for (i = 0; i < 64; i++) {
//        res0 = k[i] + m[i];
//        epp(a,b,c,d,e,f,g,h,res0)
//        ep0(a,res0)
//        ep1(e,res1)
//
//        t1 = h + res1 + ((e & f) ^ (~e & g)) + k[i] + m[i];
//        h = g;
//        g = f;
//        f = e;
//        e = d + t1;
//        d = c;
////        t1 += res0 + ((a & b) ^ (a & c) ^ (b & c));
//        t1 += res0 + ((a & (b ^ c)) ^ (b & c));
//        c = b;
//        b = a;
//        a = t1;
//      }

      a += a0;
      b += b0;
      c += c0;
      d += d0;
      e += e0;
      f += f0;
      g += g0;
      h += h0;

    }
//    if ((difficulty > 8192) && (ctx.state[0] != 0)) { //difficulty > 8192
    if (a != 0) { //difficulty > 8192
      continue;
    }
    {
      WORD hi, hii, mask = 0x80000000;
      our_diff = 0;
      step(a, b)
      step(b, c)
      step(c, d)
//      step(d, e)
//      step(e, f)
//      step(f, g)
//      step(g, h)
    }
    end:
//    work = hash2int_w();
    if( our_diff > difficulty) {
      *success = true;
      *stop = true;
      BYTE * ptrn = (BYTE*)(&(ctx_data[8]));
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

      WORD ctx_state[8];
      ctx_state[0] = a;
      ctx_state[1] = b;
      ctx_state[2] = c;
      ctx_state[3] = d;
      ctx_state[4] = e;
      ctx_state[5] = f;
      ctx_state[6] = g;
      ctx_state[7] = h;
      BYTE * ptr = (BYTE*)ctx_state;
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

