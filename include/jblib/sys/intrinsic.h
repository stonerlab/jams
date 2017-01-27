#ifndef JBLIB_SYS_INTRINSIC_H
#define JBLIB_SYS_INTRINSIC_H

#include <new>
#include <cstdlib>
#include <stdexcept>

#include "jblib/sys/define.h"

namespace jblib {
  inline void* allocate_aligned64(size_t size ){
    char * x;
    if (posix_memalign(reinterpret_cast<void**>(&x), 64, size ) != 0) {
      throw std::bad_alloc();
    }
    return x;
  }
}
#endif  // JBLIB_SYS_INTRINSIC_H
