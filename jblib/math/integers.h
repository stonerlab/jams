#ifndef JB_MATH_INTEGERS_H
#define JB_MATH_INTEGERS_H

#include "../sys/defines.h"
#include "../sys/types.h"

namespace jblib {
  JB_INLINE int32 nint(const float64 x){
    return static_cast<int32>(x+0.5);
  }

  template <typename Type1, typename Type2>
    JB_INLINE Type1 sign(const Type1 x, const Type2 y){
      if( y >= 0.0 ){
        return std::abs(x);
      } else {
        return -std::abs(x);
      }
    }
}

#endif
