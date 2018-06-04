#include <stdio.h>
#include <stdlib.h>

typedef unsigned char byte;

int read_cmd(byte *);
int write_cmd(byte *, int);
int foo(int);
int bar(int);

int main(int argc, char *argv[]) {
  int fn, arg, res, len;
  byte buf[100];
  FILE *instream = fopen("/dev/stdin","r");

//  printf("start main loop\n");
  if(instream != NULL) {
    while ((len = read_cmd(buf)) > 0) {
  //    while (fread(buf, 1, 1, stdin) != 0) {
      fn = buf[0];
      arg = buf[1];
  //    printf("read command %i %i %i\n", fn, arg, len);

      if (fn == 1) {
        res = foo(arg);
      } else if (fn == 2) {
        res = bar(arg);
      }

      buf[0] = res;
      buf[1] = len;
      write_cmd(buf, 2);
    }
  } else {
    buf[0] = 62;
    buf[1] = 63;
    write_cmd(buf, 2);
  }
}

