#ifndef JB_MAT_H
#define JB_MAT_H

#include <cmath>
#include <cassert>

#include "Vec.h"
#include "../sys/sys_defines.h"
#include "../sys/sys_types.h"

namespace jbLib {

  template < typename type, uint32 rows, uint32 cols >
    class Matrix {
      public:
        Matrix(){}
        ~Matrix(){};
    };

  template < typename type >
    class Matrix < type, 3, 3 >
    {
      public:
        Matrix<type,3,3>(){};
        Matrix<type,3,3>( const Vec<type> &a, const Vec<type> &b, const Vec<type> &c );
        Matrix<type,3,3>( const double xx, const double xy, const double xz, const double yx, const double yy, const double yz, const double zx, const double zy, const double zz );


        const Vec<type>& operator[](const int32 index) const;
        Vec<type>& operator[](const int32 index);

        Matrix<type,3,3> operator*( const type a ) const;
        Vec<type>		     operator*( const Vec<type> &vec ) const;

        template < typename ftype >
          friend Matrix<ftype,3,3>	operator*( const ftype a, const Matrix<ftype,3,3> &mat );
        template < typename ftype >
          friend Vec<ftype>        operator*( const Vec<ftype> &vec, const Matrix<ftype,3,3> &mat );

        double           determinant() const;
        Matrix<type,3,3> inverse() const;

      private:
        Vec<type> data_[3];

    };

  template < typename type >
    JB_INLINE Matrix<type,3,3>::Matrix( const Vec<type> &a, const Vec<type> &b, const Vec<type> &c ) {
      data_[0].x = a.x; data_[0].y = a.y; data_[0].z = a.z;
      data_[1].x = b.x; data_[1].y = b.y; data_[1].z = b.z;
      data_[2].x = c.x; data_[2].y = c.y; data_[2].z = c.z;
    }

  template < typename type >
    JB_INLINE Matrix<type,3,3>::Matrix( const double xx, const double xy, const double xz, const double yx, const double yy, const double yz, const double zx, const double zy, const double zz ){  	
      data_[0].x = xx; data_[0].y = xy; data_[0].z = xz;
      data_[1].x = yx; data_[1].y = yy; data_[1].z = yz;
      data_[2].x = zx; data_[2].y = zy; data_[2].z = zz;
    }

  template < typename type >
    JB_INLINE const Vec<type>& Matrix<type,3,3>::operator[](int32 index) const {
      return data_[index];
    }

  template < typename type >
    JB_INLINE Vec<type>& Matrix<type,3,3>::operator[](int32 index){
      return data_[index];
    }

  template < typename type >
    JB_INLINE Matrix<type,3,3> Matrix<type,3,3>::operator*( const type a ) const{
      return Matrix<type,3,3>(
          data_[0].x*a , data_[0].y*a , data_[0].z*a,
          data_[1].x*a , data_[1].y*a , data_[1].z*a,
          data_[2].x*a , data_[2].y*a , data_[2].z*a );
    }

  template < typename type >
    JB_INLINE Vec<type> Matrix<type,3,3>::operator*( const Vec<type> &vec ) const{
      return Vec<type>(
          data_[0].x*vec.x + data_[1].x*vec.y + data_[2].x*vec.z,
          data_[0].y*vec.x + data_[1].y*vec.y + data_[2].y*vec.z,
          data_[0].z*vec.x + data_[1].z*vec.y + data_[2].z*vec.z );
    }

  template < typename type >
    JB_INLINE double Matrix<type,3,3>::determinant() const {
      return ( data_[0].x*(data_[1].y*data_[2].z-data_[1].z*data_[2].y)
          +data_[0].y*(data_[1].z*data_[2].x-data_[1].x*data_[2].z)
          +data_[0].z*(data_[1].x*data_[2].y-data_[1].y*data_[2].x) ); 
    }

  template < typename type >
    JB_INLINE Matrix<type,3,3> Matrix<type,3,3>::inverse() const {

      // 00 01 02                00 10 20
      // 10 11 12 transpose - >  01 11 21
      // 20 21 22                02 12 22

      // xx xy xz
      // yx yy yz
      // zx zy zz

      Matrix<type,3,3> inverse;
      double det, invDet;

      inverse[0][0] = (data_[1][1]*data_[2][2]-data_[1][2]*data_[2][1]);
      inverse[1][0] = (data_[1][2]*data_[2][0]-data_[1][0]*data_[2][2]);
      inverse[2][0] = (data_[1][0]*data_[2][1]-data_[1][1]*data_[2][0]);

      det = data_[0][0]*inverse[0][0] + data_[0][1]*inverse[1][0] + data_[0][2]*inverse[2][0];

      assert( !jbMath::floatEquality(det,0.0) );

      invDet = 1.0/det;

      inverse[0][0] = invDet*(inverse[0][0]);
      inverse[0][1] = invDet*(data_[0][2] * data_[2][1] - data_[0][1] * data_[2][2]);
      inverse[0][2] = invDet*(data_[0][1] * data_[1][2] - data_[0][2] * data_[1][1]);

      inverse[1][0] = invDet*(inverse[1][0]);
      inverse[1][1] = invDet*(data_[0][0] * data_[2][2] - data_[0][2] * data_[2][0]);
      inverse[1][2] = invDet*(data_[0][2] * data_[1][0] - data_[0][0] * data_[1][2]);

      inverse[2][0] = invDet*(inverse[2][0]);
      inverse[2][1] = invDet*(data_[0][1] * data_[2][0] - data_[0][0] * data_[2][1]);
      inverse[2][2] = invDet*(data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0]); 

      return inverse;
    }


}
#endif
