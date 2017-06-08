// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_MATRIX_H
#define JBLIB_CONTAINERS_MATRIX_H

#include <cassert>
#include <cfloat>


#include "jblib/containers/vec.h"

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"
namespace jblib {

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
        Matrix<type,3,3>( const Vec3<type> &a, const Vec3<type> &b, const Vec3<type> &c );
        Matrix<type,3,3>( const type xx, const type xy, const type xz, const type yx, const type yy, const type yz, const type zx, const type zy, const type zz );


        const Vec3<type>& operator[](const int32 index) const;
        Vec3<type>& operator[](const int32 index);

        Matrix<type,3,3> operator*( const type a ) const;
        Matrix<type,3,3> operator/( const type a ) const;

        Vec3<type>		     operator*( const Vec3<type> &vec ) const;

        template < typename typeB >
        Vec3<typeB>         operator*( const Vec3<typeB> &vec ) const;

        Matrix<type,3,3> operator*( const Matrix<type,3,3> &rhs ) const;

        template < typename ftype >
          friend Matrix<ftype,3,3>	operator*( const ftype a, const Matrix<ftype,3,3> &mat );
        template < typename ftype >
          friend Vec3<ftype>        operator*( const Vec3<ftype> &vec, const Matrix<ftype,3,3> &mat );

        double           determinant() const;
        double           max_norm() const;
        Matrix<type,3,3> inverse() const;
        Matrix<type,3,3> transpose() const;


      private:
        Vec3<type> data_[3];

    };

  template < typename type >
    inline Matrix<type,3,3>::Matrix( const Vec3<type> &a, const Vec3<type> &b, const Vec3<type> &c ) {
      data_[0].x = a.x; data_[0].y = a.y; data_[0].z = a.z;
      data_[1].x = b.x; data_[1].y = b.y; data_[1].z = b.z;
      data_[2].x = c.x; data_[2].y = c.y; data_[2].z = c.z;
    }

  template < typename type >
    inline Matrix<type,3,3>::Matrix( const type xx, const type xy, const type xz, const type yx, const type yy, const type yz, const type zx, const type zy, const type zz ){
      data_[0].x = xx; data_[0].y = xy; data_[0].z = xz;
      data_[1].x = yx; data_[1].y = yy; data_[1].z = yz;
      data_[2].x = zx; data_[2].y = zy; data_[2].z = zz;
    }

  template < typename type >
    inline const Vec3<type>& Matrix<type,3,3>::operator[](int32 index) const {
      return data_[index];
    }

  template < typename type >
    inline Vec3<type>& Matrix<type,3,3>::operator[](int32 index){
      return data_[index];
    }

  template < typename type >
    inline Matrix<type,3,3> Matrix<type,3,3>::operator*( const type a ) const{
      return Matrix<type,3,3>(
          data_[0].x*a , data_[0].y*a , data_[0].z*a,
          data_[1].x*a , data_[1].y*a , data_[1].z*a,
          data_[2].x*a , data_[2].y*a , data_[2].z*a );
    }

    template < typename type >
      inline Matrix<type,3,3> Matrix<type,3,3>::operator/( const type a ) const{
        return Matrix<type,3,3>(
            data_[0].x/a , data_[0].y/a , data_[0].z/a,
            data_[1].x/a , data_[1].y/a , data_[1].z/a,
            data_[2].x/a , data_[2].y/a , data_[2].z/a );
      }

  template < typename type >
    inline Vec3<type> Matrix<type,3,3>::operator*( const Vec3<type> &vec ) const{
      return Vec3<type>(
          data_[0].x*vec.x + data_[0].y*vec.y + data_[0].z*vec.z,
          data_[1].x*vec.x + data_[1].y*vec.y + data_[1].z*vec.z,
          data_[2].x*vec.x + data_[2].y*vec.y + data_[2].z*vec.z );
    }

  template < typename typeA >
  template < typename typeB >
    inline Vec3<typeB> Matrix<typeA,3,3>::operator*( const Vec3<typeB> &vec ) const{
      return Vec3<typeB>(
          data_[0].x*vec.x + data_[0].y*vec.y + data_[0].z*vec.z,
          data_[1].x*vec.x + data_[1].y*vec.y + data_[1].z*vec.z,
          data_[2].x*vec.x + data_[2].y*vec.y + data_[2].z*vec.z );
    }

  template < typename type >
    inline Matrix<type,3,3> Matrix<type,3,3>::operator*( const Matrix<type,3,3> &rhs ) const{
      Matrix<type, 3, 3> lhs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            lhs[i][j] += data_[i][k]*rhs[k][j];
          }
        }
      }
      return lhs;
    }


  template < typename type >
    inline double Matrix<type,3,3>::max_norm() const {
      double max = DBL_MIN;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (std::abs(data_[i][j]) > max) {
            max = std::abs(data_[i][j]);
          }
        }
      }
      return max;
    }

  template < typename type >
    inline double Matrix<type,3,3>::determinant() const {
      return ( data_[0].x*(data_[1].y*data_[2].z-data_[1].z*data_[2].y)
          +data_[0].y*(data_[1].z*data_[2].x-data_[1].x*data_[2].z)
          +data_[0].z*(data_[1].x*data_[2].y-data_[1].y*data_[2].x) );
    }

  template < typename type >
    inline Matrix<type,3,3> Matrix<type,3,3>::inverse() const {

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

      assert( !floats_are_equal(det,0.0) );

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
  template < typename type >
    inline Matrix<type,3,3> Matrix<type,3,3>::transpose() const {
      return Matrix<type, 3, 3>(
        data_[0][0], data_[1][0], data_[2][0],
        data_[0][1], data_[1][1], data_[2][1],
        data_[0][2], data_[1][2], data_[2][2]
      );
    }

}

#endif  // JBLIB_CONTAINERS_MATRIX_H
