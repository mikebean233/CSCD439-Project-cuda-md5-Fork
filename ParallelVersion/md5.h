#ifndef PROJECT_MD5_H
#define PROJECT_MD5_H

__host__ __device__ void md5_vfy(unsigned char* data, uint length, uint *a1, uint *b1, uint *c1, uint *d1);
uint unhex(unsigned char x);
void md5_to_ints(unsigned char* md5, uint *r0, uint *r1, uint *r2, uint *r3);




#endif //PROJECT_MD5_H
