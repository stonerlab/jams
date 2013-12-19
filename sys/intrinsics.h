#ifndef SYS_INTRINSICS_H
#define SYS_INTRINSICS_H

#include <new>
#include <cstdlib>
#include <stdexcept>

#include "defines.h"

namespace jblib {
  JB_INLINE void* allocate_aligned64(size_t size ){
    char * x;
    if( posix_memalign( (void**)&x, 64, size ) != 0 ) {
      throw std::bad_alloc();
    }
    return x;
  }
}
#endif
