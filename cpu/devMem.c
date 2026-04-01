#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "devMem.h"

void* mallocForDevice(void *x, int n, int size) {
  if (n <= 0) {
    return NULL;
  }
  size_t alloc_size = (size_t)n * (size_t)size;
  x = malloc(alloc_size);
  if (x == NULL) {
    fprintf(stderr, "ERROR: malloc failed for n=%d, size=%d (%.2f GB)\n", 
            n, size, (double)alloc_size / (1024.0*1024.0*1024.0));
    return NULL;
  }
  memset(x, 0, alloc_size);
  return x;
}

void memcpyDevice(void *dest, void *src, int n, int size, char *type) {
  size_t copy_size = (size_t)n * (size_t)size;
  memcpy(dest, src, copy_size);
}

void freeDevice(void *p) {
  free(p);
}
