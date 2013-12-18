#ifndef JB_ARRAY_H
#define JB_ARRAY_H

#include "../sys/sys_assert.h"
#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"
#include "../sys/sys_intrinsics.h"

#include <cstring>
#include <utility>
#include <algorithm>

namespace jbLib{
  template <typename Tp_, uint32 Dim_, typename Idx_=uint32>
    class Array 
    {
      public:
        Array(){}
        ~Array(){};
    };
}
#include "Array1D.h"
#include "Array2D.h"
#include "Array3D.h"
#include "Array4D.h"
#include "Array5D.h"

#endif
