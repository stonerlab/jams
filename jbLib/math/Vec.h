#ifndef JB_VEC_H
#define JB_VEC_H

#include <cmath>

#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"

namespace jbVec{

    JB_INLINE double dot(const double a[3], const double b[3]){
        return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
    }

    JB_INLINE double mod(const double a[3]){
        return sqrt( dot(a,a) );
    }

    JB_INLINE double angle(const double a[3], const double b[3]){
        return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    }
}

#endif
