// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JBLIB_CONTAINERS_VEC3_H
#define JBLIB_CONTAINERS_VEC3_H

// TODO: remove output to another header
#include <ostream>
#include <cmath>
#include <algorithm>

#include "jblib/sys/define.h"
#include "jblib/sys/types.h"

#include "jblib/math/equalities.h"

namespace jblib {
  template< typename type >
    class Vec3 {

      public:
        type x;
        type y;
        type z;

        // Don't give default arguments of zero because this could be miss
        // interpreted if just one argument is given
        // default constructors
        Vec3() : x(0), y(0), z(0) {}
        Vec3(const type ix, const type iy, const type iz) : x(ix), y(iy), z(iz) {}
        //Vec3( const Vec3<type> &other ) : x(other.x), y(other.y), z(other.z) {}

        void set(const type ix, const type iy, const type iz);

        type	    operator[]( const int32 i ) const;
        type &	    operator[]( const int32 i );
        Vec3<type>	operator-() const;
        //Vec3<type> &	operator=( const Vec3 &a );
        Vec3<type>	operator*( const type a ) const;
        Vec3<type>	operator/( const type a ) const;
        Vec3<type>	operator+( const Vec3 &a ) const;
        Vec3<type>	operator-( const Vec3 &a ) const;
        Vec3<type> &	operator+=( const Vec3 &a );
        Vec3<type> &	operator-=( const Vec3 &a );
        Vec3<type> &	operator/=( const double a );
        Vec3<type> &	operator*=( const double a );

        inline type norm() const {
          return sqrt(x * x + y * y + z * z);
        }

        inline type norm_sq() const {
          return (x * x + y * y + z * z);
        }

        template< typename ftype >
          friend void swap( const Vec3<ftype> &a, const Vec3<ftype> &b);

        template< typename ftype >
          friend ftype dot( const Vec3<ftype> &a, const Vec3<ftype> &b);
        template< typename ftype >
          friend Vec3<ftype> cross( const Vec3<ftype> &a, const Vec3<ftype> &b);
        template< typename ftype >
          friend double abs( const Vec3<ftype> &a);
        template< typename ftype >
          friend double angle( const Vec3<ftype> &a, const Vec3<ftype> &b);
        template< typename ftype >
          friend ftype sum( const Vec3<ftype> &a);
        template< typename ftype >
          friend ftype product( const Vec3<ftype> &a);

        template< typename ftype >
          friend std::ostream& operator<<(std::ostream& os, const Vec3<ftype> &vec);

        //Vec3<type>& operator=(Vec3<type> rhs);

    };

  template< typename type >
    inline void Vec3<type>::set(const type ix, const type iy, const type iz){
      x = ix; y = iy; z = iz;
    }

  template< typename type >
    inline type Vec3<type>::operator[]( const int32 i ) const{
      switch(i) {
        case(0):
          return x;
        case(1):
          return y;
        case(2):
          return z;
        default:
          throw std::runtime_error("invalid Vec3 index");
      }
    }

  template< typename type >
    inline type & Vec3<type>::operator[]( const int32 i ){
      switch(i) {
        case(0):
          return x;
        case(1):
          return y;
        case(2):
          return z;
        default:
          throw std::runtime_error("invalid Vec3 index");
      }
    }

  template< typename type >
    inline Vec3<type> Vec3<type>::operator-() const {
      return Vec3<type>( -x, -y, -z);
    }

  //template< typename type >
  //inline Vec3<type> & Vec3<type>::operator=(const Vec3<type> &a) {
  //x = a.x; y = a.y, z = a.z;
  //return *this;
  //}

  template< typename type >
    inline Vec3<type> Vec3<type>::operator*(const type a) const{
      return Vec3<type>( a*x, a*y, a*z );
    }

  template< typename type >
    inline Vec3<type> Vec3<type>::operator/(const type a) const{
      const double norm = 1.0/a;
      return Vec3<type>( norm*x, norm*y, norm*z );
    }

  template< typename type >
    inline Vec3<type> Vec3<type>::operator+(const Vec3<type> &a) const{
      return Vec3<type>( x+a.x, y+a.y, z+a.z );
    }

  template< typename type >
    inline Vec3<type> Vec3<type>::operator-(const Vec3<type> &a) const{
      return Vec3<type>( x-a.x, y-a.y, z-a.z );
    }

  template< typename type >
    inline Vec3<type> &Vec3<type>::operator+=(const Vec3<type> &a) {
      x += a.x; y += a.y; z += a.z;
      return *this;
    }

  template< typename type >
    inline Vec3<type> &Vec3<type>::operator-=(const Vec3<type> &a) {
      x -= a.x; y -= a.y; z -= a.z;
      return *this;
    }

  template< typename type >
    inline Vec3<type> &Vec3<type>::operator/=(const double a) {
      x /= a; y /= a; z /= a;
      return *this;
    }

  template< typename type >
    inline Vec3<type> &Vec3<type>::operator*=(const double a) {
      x *= a; y *= a; z *= a;
      return *this;
    }

  // Friends
  template< typename type >
    inline void swap( const Vec3<type> &a, const Vec3<type> &b){
      std::swap(a.x,b.x); std::swap(a.y,b.y); std::swap(a.z,b.z);
    }

  template< typename type >
    inline type dot( const Vec3<type> &a, const Vec3<type> &b){
      return (a.x*b.x + a.y*b.y + a.z*b.z);
    }

  template< typename type >
    inline Vec3<type> cross( const Vec3<type> &a, const Vec3<type> &b){
      return Vec3<type>( (a.y*b.z-a.z*b.y), (a.z*b.x-a.x*b.z), (a.x*b.y - a.y*b.x) );
    }

  template< typename type >
    inline double abs( const Vec3<type> &a){
      return sqrt( dot(a,a) );
    }

  template< typename type >
    inline double angle( const Vec3<type> &a, const Vec3<type> &b){
      return acos( dot(a,b) / ( mod(a)*mod(b) ) );
    }

  template< typename ftype >
    inline ftype sum( const Vec3<ftype> &a) {
      return a.x + a.y + a.z;
    }

  template< typename ftype >
    inline ftype product( const Vec3<ftype> &a) {
      return a.x*a.y*a.z;
    }


  ///
  /// @brief returns an integer count of the number of zero elements in (x,y,z)
  ///
  /// @note bool to int conversion is defined in the C++ standard.
  ///       See http://stackoverflow.com/questions/5369770/bool-to-int-conversion
  ///
  inline int vecCountZeros(const Vec3<double> &a){
    return ( floats_are_equal(a.x,0.0) + floats_are_equal(a.y,0.0) + floats_are_equal(a.z,0.0) );
  }

  template< typename type >
    std::ostream& operator<<(std::ostream& os, const Vec3<type> &vec)
    {
      return os << vec.x << "\t" << vec.y << "\t" << vec.z;
    }
}

#endif  // JBLIB_CONTAINERS_VEC3_H
