#ifndef DEVMEM_H
#define DEVMEM_H

#ifdef __cplusplus
extern "C" {
#endif

void* mallocForDevice(void *x, int n, int size);
void memcpyDevice (void *dest, void *src, int n, int size, char *type);
void freeDevice(void *p);

#ifdef __cplusplus
}
#endif

#endif
