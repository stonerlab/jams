#ifndef JB_MATH_FUNCTIONS_H
#define JB_MATH_FUNCTIONS_H

#include "../sys/defines.h"
#include "../sys/types.h"

namespace jblib {

  template <typename Type>
    JB_INLINE Type square(const Type x){
      return (x*x);
    }


  JB_INLINE float64 coth( const float64 x ){
    return (1.0/tanh(x));
  }

  JB_INLINE float64 BrillouinFunc( const float64 x, const float64 S ){
    return (((2.0*S+1.0)/(2.0*S))*coth(((2.0*S+1.0)/(2.0*S))*x)
        - (1.0/(2.0*S))*coth((1.0/(2.0*S))*x));
  }

  JB_INLINE float64 BrillouinFuncDerivative( const float64 x, const float64 S ){
    return square((1.0/(2.0*S))/sinh((1.0/(2.0*S))*x)) -square(((2.0*S+1.0)/(2.0*S))/sinh(((2.0*S+1.0)/(2.0*S))*x));
  }

  JB_INLINE float64 LangevinFunc( const float64 x ){
    return (coth(x) - (1.0/x));
  }

}

#endif
