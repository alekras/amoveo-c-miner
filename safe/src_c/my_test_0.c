#include <stdio.h>


int main()
{
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
