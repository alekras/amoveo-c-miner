#include <stdio.h>
/****************************** MACROS ******************************/
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

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
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;
  unsigned int res = 0;

  *ms = x;
  val = val >> 2;
  res = *ms | *ls;

  val = val >> 11;
  res = res ^ (*ms | *ls);

  val = val >> 9;

  return res ^ (*ms | *ls);;
}

unsigned int ep1(unsigned int x) {
  unsigned long int val = 0;
  unsigned int *ls = (unsigned int*)&val;
  unsigned int *ms = ls + 1;
  unsigned int res = 0;

  *ms = x;
  val = val >> 6;
  res = *ms | *ls;

  val = val >> 5;
  res = res ^ (*ms | *ls);

  val = val >> 14;

  return res ^ (*ms | *ls);;
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

  return res ^ (x >> 3);;
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

  return res ^ (x >> 10);;
}

int main() {
  unsigned int v = 0x23006307;

  printf("rotr: %08x ep0: %08x ep1: %08x sig0: %08x sig1: %08x\n",
    ROTRIGHT(v, 15), EP0(v), EP1(v), SIG0(v), SIG1(v)
  );

  printf("rotr: %08x ep0: %08x ep1: %08x sig0: %08x sig1: %08x\n",
    rotr(v, 15), ep0(v), ep1(v), sig0(v), sig1(v)
  );

  return(0);
}

int main_1() {
  unsigned int int_v;
  unsigned long int long_v = 0x0102030405060708;
  unsigned long long int longlong_v;
  unsigned short int short_v;

  printf("int size: %d\nlong size: %d\nlong long size: %d\nshort size: %d\n",
    (int)sizeof(int_v), (int)sizeof(long_v), (int)sizeof(longlong_v), (int)sizeof(short_v)
  );

  unsigned long int* ptr_long_v = &long_v;
  unsigned int* ptr_int_v = (unsigned int*)&long_v;
  printf("long value: %016lx\n", *ptr_long_v);
  printf("int value: %08x\n", *ptr_int_v);
  printf("int value: %08x\n", *(ptr_int_v + 1));

  long_v = 0x12345678;
  printf("before shift: %016lx\n", *ptr_long_v);
  long_v = long_v << 4;
  printf("after shift: %016lx\n", *ptr_long_v);
  long_v = *ptr_int_v | *(ptr_int_v + 1);
  printf("after | : %016lx\n", *ptr_long_v);
  return(0);
}
