#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef unsigned char byte;

int read_cmd(byte *);
int write_cmd(byte *, int);
int read_exact(byte *, int);
int write_exact(byte *, int);

int read_cmd(byte *buf) {
  int len;

  if (read_exact(buf, 2) != 2) {
    return(-1);
  }
  len = (buf[0] << 8) | buf[1];
//  printf("read_cmd len= %i\n", len);
  len = 2;
  return read_exact(buf, len);
}

int write_cmd(byte *buf, int len) {
  byte li;

  li = (len >> 8) & 0xff;
  write_exact(&li, 1);

  li = len & 0xff;
  write_exact(&li, 1);

  return write_exact(buf, len);
}

int read_exact(byte *buf, int len) {
  int i, got = 0;
//  i = fread(buf, 1, len, stdin);
//  printf("read_exact %i %i\n", len, i);
  do {
    if ((i = read(0, buf+got, len-got)) <= 0)
      return(i);
    got += i;
  } while (got<len);
//  do {
//    if ((i = fread(buf + got, 1, len - got, stdin)) <= 0) {
//      return(i);
//    }
//    got += i;
//  } while (got < len);

  return len;
}

int write_exact(byte *buf, int len) {
  int i, wrote = 0;

  do {
    if ((i = fwrite(buf + wrote, 1, len - wrote, stdout)) <= 0) {
      return (i);
    }
    wrote += i;
  } while (wrote < len);

  return len;
}
