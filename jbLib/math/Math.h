#ifndef JB_MATH_H
#define JB_MATH_H

#include <cmath>

#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"

#define JB_RAND_MAX UINT32_MAX

namespace jbLib {
  namespace jbMath {

    const double twopi  = 6.2831853071795864769252867663;
    const double pi     = 3.1415926535897932384626433832795;
    const double halfpi = 1.5707963267948965579989817342721;

    const double floatEpsilon = 1e-8;
    const double maxDouble    = 1.7976931348623157e308;

    template <typename Type>
      JB_INLINE Type square(const Type x){
        return (x*x);
      }

    JB_INLINE bool floatEquality( const double x, const double y, const double epsilon=floatEpsilon){
      if( fabs( x - y ) < epsilon ){
        return true;
      }
      return false;
    }

    JB_INLINE bool floatLessThanOrEqual( const double x, const double y, const double epsilon=floatEpsilon){
      if( (x < y) || fabs( x - y ) < epsilon ){
        return true;
      }
      return false;
    }

    JB_INLINE bool floatGreaterThanOrEqual( const double x, const double y, const double epsilon=floatEpsilon){
      if( (x > y) || fabs( x - y ) < epsilon ){
        return true;
      }
      return false;
    }


    JB_INLINE bool threewayFloatEquality(const double x, const double y, const double z, const double epsilon=floatEpsilon){
      return ( jbMath::floatEquality(x,y,epsilon) && jbMath::floatEquality(y,z,epsilon) );
    }

    JB_INLINE bool threewayFloatNotEquality(const double x, const double y, const double z, const double epsilon=floatEpsilon){
      return ( (!jbMath::floatEquality(x,y,epsilon) && !jbMath::floatEquality(y,z,epsilon)) );
    }

    JB_INLINE double coth( const double x ){
      return (1.0/tanh(x));
    }

    JB_INLINE double BrillouinFunc( const double x, const double S ){
      return (((2.0*S+1.0)/(2.0*S))*coth(((2.0*S+1.0)/(2.0*S))*x)
          - (1.0/(2.0*S))*coth((1.0/(2.0*S))*x));
    }

    JB_INLINE double BrillouinFuncDerivative( const double x, const double S ){
      return square((1.0/(2.0*S))/sinh((1.0/(2.0*S))*x)) -square(((2.0*S+1.0)/(2.0*S))/sinh(((2.0*S+1.0)/(2.0*S))*x));
    }

    JB_INLINE double LangevinFunc( const double x ){
      return (coth(x) - (1.0/x));
    }


    double trivialSum(const double *data, const uint32 size);

    double KahanSum(const double *data, const uint32 size);

    JB_INLINE double Sum(const double *data, const uint32 size){
#ifdef JB_FASTSUM
      return trivialSum(data,size);
#else
      return KahanSum(data,size);
#endif
    }

    JB_INLINE int32 nint(const double x){
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

    JB_INLINE void cartesianToSpherical(const double x, const double y, 
        const double z, double& r, double& theta, double& phi){
      r       = sqrt(x*x+y*y+z*z);
      theta   = acos( z/r );
      phi     = atan2(y,x);
    }

    JB_INLINE void sphericalToCartesian(const double r, const double theta,
        const double phi, double& x, double& y, double& z){
      x = r*cos(theta)*cos(phi);
      y = r*cos(theta)*sin(phi);
      z = r*sin(theta);
    }

  }

}
#endif
