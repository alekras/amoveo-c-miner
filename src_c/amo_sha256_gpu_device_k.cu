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
#include "macros.h"

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
  WORD ctx_data[27];
  WORD res0;
  WORD m[64];
  WORD a, b, c, d, e, f, g, h;
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
    sig0_1(ctx_data[i - 15], ctx_data[i - 2], ctx_data[i - 7], ctx_data[i - 16], ctx_data[i])
  }
#pragma unroll 1
  for (int i = 21 ; i < 27; ++i) {
    sig0(ctx_data[i - 15], res0)
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
    {
      for (i = 0; i < 27; ++i)
        m[i] = ctx_data[i];

//      for (i = 21 ; i < 27; ++i) {
//        sig1(m[i - 2], m[i])
      sig1(m[19], m[21])
      sig1(m[20], m[22])
      sig1(m[21], m[23])
      sig1(m[22], m[24])
      sig1(m[23], m[25])
      sig1(m[24], m[26])
//      }
//      for (i = 27 ; i < 64; ++i) {
//        sig0_1(m[i - 15], m[i - 2], m[i - 7], m[i - 16], m[i])
      sig0_1(m[12], m[25], m[20], m[11], m[27])
      sig0_1(m[13], m[26], m[21], m[12], m[28])
      sig0_1(m[14], m[27], m[22], m[13], m[29])
      sig0_1(m[15], m[28], m[23], m[14], m[30])
      sig0_1(m[16], m[29], m[24], m[15], m[31])
      sig0_1(m[17], m[30], m[25], m[16], m[32])
      sig0_1(m[18], m[31], m[26], m[17], m[33])
      sig0_1(m[19], m[32], m[27], m[18], m[34])
      sig0_1(m[20], m[33], m[28], m[19], m[35])
      sig0_1(m[21], m[34], m[29], m[20], m[36])
      sig0_1(m[22], m[35], m[30], m[21], m[37])
      sig0_1(m[23], m[36], m[31], m[22], m[38])
      sig0_1(m[24], m[37], m[32], m[23], m[39])
      sig0_1(m[25], m[38], m[33], m[24], m[40])
      sig0_1(m[26], m[39], m[34], m[25], m[41])
      sig0_1(m[27], m[40], m[35], m[26], m[42])
      sig0_1(m[28], m[41], m[36], m[27], m[43])
      sig0_1(m[29], m[42], m[37], m[28], m[44])
      sig0_1(m[30], m[43], m[38], m[29], m[45])
      sig0_1(m[31], m[44], m[39], m[30], m[46])
      sig0_1(m[32], m[45], m[40], m[31], m[47])
      sig0_1(m[33], m[46], m[41], m[32], m[48])
      sig0_1(m[34], m[47], m[42], m[33], m[49])
      sig0_1(m[35], m[48], m[43], m[34], m[50])
      sig0_1(m[36], m[49], m[44], m[35], m[51])
      sig0_1(m[37], m[50], m[45], m[36], m[52])
      sig0_1(m[38], m[51], m[46], m[37], m[53])
      sig0_1(m[39], m[52], m[47], m[38], m[54])
      sig0_1(m[40], m[53], m[48], m[39], m[55])
      sig0_1(m[41], m[54], m[49], m[40], m[56])
      sig0_1(m[42], m[55], m[50], m[41], m[57])
      sig0_1(m[43], m[56], m[51], m[42], m[58])
      sig0_1(m[44], m[57], m[52], m[43], m[59])
      sig0_1(m[45], m[58], m[53], m[44], m[60])
      sig0_1(m[46], m[59], m[54], m[45], m[61])
      sig0_1(m[47], m[60], m[55], m[46], m[62])
      sig0_1(m[48], m[61], m[56], m[47], m[63])
//      }

      a = a0;
      b = b0;
      c = c0;
      d = d0;
      e = e0;
      f = f0;
      g = g0;
      h = h0;

      for (i = 0; i < 64; i++) m[i] += k[i];

      epp(a,b,c,d,e,f,g,h,m[ 0])
      epp(a,b,c,d,e,f,g,h,m[ 1])
      epp(a,b,c,d,e,f,g,h,m[ 2])
      epp(a,b,c,d,e,f,g,h,m[ 3])
      epp(a,b,c,d,e,f,g,h,m[ 4])
      epp(a,b,c,d,e,f,g,h,m[ 5])
      epp(a,b,c,d,e,f,g,h,m[ 6])
      epp(a,b,c,d,e,f,g,h,m[ 7])
      epp(a,b,c,d,e,f,g,h,m[ 8])
      epp(a,b,c,d,e,f,g,h,m[ 9])
      epp(a,b,c,d,e,f,g,h,m[10])
      epp(a,b,c,d,e,f,g,h,m[11])
      epp(a,b,c,d,e,f,g,h,m[12])
      epp(a,b,c,d,e,f,g,h,m[13])
      epp(a,b,c,d,e,f,g,h,m[14])
      epp(a,b,c,d,e,f,g,h,m[15])

      epp(a,b,c,d,e,f,g,h,m[16])
      epp(a,b,c,d,e,f,g,h,m[17])
      epp(a,b,c,d,e,f,g,h,m[18])
      epp(a,b,c,d,e,f,g,h,m[19])
      epp(a,b,c,d,e,f,g,h,m[20])
      epp(a,b,c,d,e,f,g,h,m[21])
      epp(a,b,c,d,e,f,g,h,m[22])
      epp(a,b,c,d,e,f,g,h,m[23])
      epp(a,b,c,d,e,f,g,h,m[24])
      epp(a,b,c,d,e,f,g,h,m[25])
      epp(a,b,c,d,e,f,g,h,m[26])
      epp(a,b,c,d,e,f,g,h,m[27])
      epp(a,b,c,d,e,f,g,h,m[28])
      epp(a,b,c,d,e,f,g,h,m[29])
      epp(a,b,c,d,e,f,g,h,m[30])
      epp(a,b,c,d,e,f,g,h,m[31])

      epp(a,b,c,d,e,f,g,h,m[32])
      epp(a,b,c,d,e,f,g,h,m[33])
      epp(a,b,c,d,e,f,g,h,m[34])
      epp(a,b,c,d,e,f,g,h,m[35])
      epp(a,b,c,d,e,f,g,h,m[36])
      epp(a,b,c,d,e,f,g,h,m[37])
      epp(a,b,c,d,e,f,g,h,m[38])
      epp(a,b,c,d,e,f,g,h,m[39])
      epp(a,b,c,d,e,f,g,h,m[40])
      epp(a,b,c,d,e,f,g,h,m[41])
      epp(a,b,c,d,e,f,g,h,m[42])
      epp(a,b,c,d,e,f,g,h,m[43])
      epp(a,b,c,d,e,f,g,h,m[44])
      epp(a,b,c,d,e,f,g,h,m[45])
      epp(a,b,c,d,e,f,g,h,m[46])
      epp(a,b,c,d,e,f,g,h,m[47])

      epp(a,b,c,d,e,f,g,h,m[48])
      epp(a,b,c,d,e,f,g,h,m[49])
      epp(a,b,c,d,e,f,g,h,m[50])
      epp(a,b,c,d,e,f,g,h,m[51])
      epp(a,b,c,d,e,f,g,h,m[52])
      epp(a,b,c,d,e,f,g,h,m[53])
      epp(a,b,c,d,e,f,g,h,m[54])
      epp(a,b,c,d,e,f,g,h,m[55])
      epp(a,b,c,d,e,f,g,h,m[56])
      epp(a,b,c,d,e,f,g,h,m[57])
      epp(a,b,c,d,e,f,g,h,m[58])
      epp(a,b,c,d,e,f,g,h,m[59])
      epp(a,b,c,d,e,f,g,h,m[60])
      epp(a,b,c,d,e,f,g,h,m[61])
      epp(a,b,c,d,e,f,g,h,m[62])
      epp(a,b,c,d,e,f,g,h,m[63])

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
      our_diff = 32;
//      step(a, b)
      step(b, c)
      step(c, d)
//      step(d, e)
//      step(e, f)
//      step(f, g)
//      step(g, h)
    }
    end:
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

