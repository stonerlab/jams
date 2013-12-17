#ifndef SYS_INTRINSICS_H
#define SYS_INTRINSICS_H

void* allocate_aligned64(size_t size ){
    char * x;
    posix_memalign( (void**)&x, 64, size );
    return x;
}
#endif
