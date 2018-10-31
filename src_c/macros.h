/*
 * macros.h
 *
 *  Created on: Oct 31, 2018
 *      Author: alekras
 */

#ifndef SRC_C_MACROS_H_
#define SRC_C_MACROS_H_

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
7. t1 = t1 ^ t2
8. res = res + t1
 */
#define sig1(x,res) \
  asm("{\n\t" \
      " .reg .u32 t1;\n\t" \
      " .reg .u32 t2;\n\t" \
      " shf.r.clamp.b32    t1, %1, %1, 17;\n\t" \
      " shf.r.clamp.b32    t2, %1, %1, 19;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " shr.u32            t2, %1, 10;\n\t" \
      " xor.b32            t1, t1, t2;\n\t" \
      " add.s32            %0, %0, t1;\n\t" \
      "}" \
    : "+r"(res) : "r" (x));

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
#define sig0_1(x,y,a,b,res) \
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
      " add.s32            %0, %0, %3;\n\t" \
      " add.s32            %0, %0, %4;\n\t" \
      "}" \
    : "+r"(res) : "r"(x), "r"(y), "r"(a), "r"(b));

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

#endif /* SRC_C_MACROS_H_ */
