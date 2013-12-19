#ifndef JB_MATH_EQUALITIES_H
#define JB_MATH_EQUALITIES_H

#include <cmath>

#include "../sys/defines.h"
#include "../sys/types.h"

#include "constants.h"

namespace jblib {

  JB_INLINE bool floatEquality( const float64 x, const float64 y, const float64 epsilon=floatEpsilon){
    if( fabs( x - y ) < epsilon ){
      return true;
    }
    return false;
  }

  JB_INLINE bool floatLessThanOrEqual( const float64 x, const float64 y, const float64 epsilon=floatEpsilon){
    if( (x < y) || fabs( x - y ) < epsilon ){
      return true;
    }
    return false;
  }

  JB_INLINE bool floatGreaterThanOrEqual( const float64 x, const float64 y, const float64 epsilon=floatEpsilon){
    if( (x > y) || fabs( x - y ) < epsilon ){
      return true;
    }
    return false;
  }


  JB_INLINE bool threewayFloatEquality(const float64 x, const float64 y, const float64 z, const float64 epsilon=floatEpsilon){
    return ( floatEquality(x,y,epsilon) && floatEquality(y,z,epsilon) );
  }

  JB_INLINE bool threewayFloatNotEquality(const float64 x, const float64 y, const float64 z, const float64 epsilon=floatEpsilon){
    return ( (!floatEquality(x,y,epsilon) && !floatEquality(y,z,epsilon)) );
  }

}

#endif
