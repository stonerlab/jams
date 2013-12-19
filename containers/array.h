#ifndef JB_ARRAY_H
#define JB_ARRAY_H

#include "../sys/asserts.h"
#include "../sys/defines.h"
#include "../sys/types.h"
#include "../sys/intrinsics.h"

#include <cstring>
#include <utility>
#include <algorithm>

namespace jblib{
  template <typename Tp_, uint32 Dim_, typename Idx_=uint32>
    class Array 
    {
      public:
        Array(){}
        ~Array(){};
    };
}
#include "array1d.h"
#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "array5d.h"

#endif
